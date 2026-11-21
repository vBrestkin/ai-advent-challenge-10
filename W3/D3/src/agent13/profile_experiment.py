"""Compare two profiles and check ten consecutive turns."""
import json
from pathlib import Path
import re

import httpx
from fastapi.testclient import TestClient

from .log import Journal, LoggingParser, identifier
from .metrics import Budget, TokenCounter
from .scenario import csv_text
from .transport import Client, preflight
from .web import ROOT, create_app, load_settings

BRIEF = {'length':'brief', 'code':'without'}
DETAILED = {'length':'detailed', 'code':'with'}
QUESTION = 'Как посчитать, сколько раз каждое слово встречается в списке?'
UPDATE = 'Теперь отвечай кратко и без кода.'
SERIES = [
    'Что такое словарь в Python и для чего он нужен?',
    'Чем список отличается от множества?',
    'Как удалить дубликаты из списка, сохранив порядок?',
    'Как работает сортировка по ключу?',
    'Чем отличается поверхностная копия от глубокой?',
    'Как организовать проверку входных данных?',
    'Что такое исключение и когда его перехватывать?',
    'Зачем в тесте подменять внешнее API?',
    'Что стоит записывать в журнал обработки запроса?',
    'Как проверить, что функция одинаково работает после перезапуска?',
]

FENCED_BLOCK = re.compile(r'^ {0,3}(?P<fence>`{3,}|~{3,})[^\n]*\n.*?^ {0,3}(?P=fence)[ \t]*$', re.MULTILINE | re.DOTALL)
WORD = re.compile(r'[^\W_]+(?:[-’][^\W_]+)*', re.UNICODE)


def assess(answer, profile):
    has_code = bool(FENCED_BLOCK.search(answer))
    words = len(WORD.findall(FENCED_BLOCK.sub('', answer)))
    length_ok = {'brief': words <= 45, 'detailed': words >= 100}[profile['length']]
    code_ok = has_code if profile['code'] == 'with' else not has_code
    return {'words':words,'has_code':has_code,'length_ok':length_ok,'code_ok':code_ok,
            'passed':length_ok and code_ok}


def first_violation(rows):
    return next((row['number'] for row in rows if not row['passed']), None)


def run_matrix(web, model, on_result):
    def create(title, preferences):
        sid = web.post('/api/layers', json={'title':title}).raise_for_status().json()['id']
        web.put(f'/api/layers/{sid}/profile', json=preferences).raise_for_status()
        return sid

    def ask(sid, question, phase, number):
        path = f'/api/layers/{sid}'
        before = web.get(path).raise_for_status().json()
        start = len(model.records)
        response = web.post(path+'/chat', json={'message':question})
        result = response.json()
        row = {'id':sid, 'phase':phase, 'number':number, 'question':question, 'before':before['profile'],
               'after':result.get('profile'), 'profile_changed':result.get('profile_changed',False),
               'answer':result.get('answer'), 'http_status':response.status_code,
               'calls':model.records[start:], 'steps':result.get('steps',[]), 'error':result.get('detail')}
        if response.is_success:
            row.update(assess(row['answer'],row['after']))
            persisted = web.get(path).raise_for_status().json()
            if persisted['profile'] != row['after'] or persisted['memory']['short'][-1]['content'] != row['answer']:
                raise RuntimeError('Ответ или профиль не сохранился.')
        on_result(row)
        response.raise_for_status()
        return row

    brief = create('1. Кратко и без кода', BRIEF)
    detailed = create('2. Подробно и с кодом', DETAILED)
    ask(brief, QUESTION, 'compare', 1)
    ask(detailed, QUESTION, 'compare', 2)

    changed = create('3. Смена профиля в диалоге', DETAILED)
    update = ask(changed, UPDATE, 'update', 1)
    if not update['profile_changed'] or update['after'] != BRIEF:
        raise RuntimeError('Реплика не обновила профиль.')
    ask(changed, QUESTION, 'follow_up', 1)

    series = create('4. Серия из 10 вопросов', BRIEF)
    for number, question in enumerate(SERIES, 1):
        ask(series, question, 'series', number)

    return {'brief':brief,'detailed':detailed,'changed':changed,'series':series}


def write_results(directory, data):
    (directory/'experiment.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    fields = ('phase','number','question','answer','words','has_code','length_ok','code_ok','passed','http_status')
    (directory/'details.csv').write_text(csv_text([{key:row.get(key) for key in fields} for row in data['rows']]),
                                         encoding='utf-8-sig')


def run(root=ROOT, *, budget_rub=20):
    root = Path(root)
    directory = root/'results'/identifier()
    directory.mkdir(parents=True)
    budget = Budget(budget_rub)
    key = (load_settings(root).get('POLZA_API_KEY') or '').strip()
    data = {'status':'running','rows':[],'thresholds':{'brief_max':45,'detailed_min':100},
            'budget':budget.accounting()}
    with Journal(root/'logs',key) as journal, httpx.Client(transport=httpx.HTTPTransport(retries=0)) as http:
        try:
            if not key:
                raise ValueError('Укажите POLZA_API_KEY.')
            counter = TokenCounter(root/'data/tokenizer.json')
            model = Client(key,journal,http,counter=counter,budget=budget)
            data['catalog'] = preflight(model)

            def save(row):
                data['rows'].append(row)
                data['budget'] = budget.accounting()
                write_results(directory,data)
                summary = f"{row['phase']} {row['number']}: слов {row.get('words')}, код {row.get('has_code')}, критерии {row.get('passed')}"
                journal.record('PROFILE_CHECK',summary)
                print(summary,flush=True)

            db = directory/'data'
            with TestClient(create_app(root=root,model=model,data_dir=db)) as web:
                data['sessions'] = run_matrix(web,model,save)
            with TestClient(create_app(root=root,model=model,data_dir=db)) as web:
                expected = {'brief':BRIEF, 'detailed':DETAILED, 'changed':BRIEF, 'series':BRIEF}
                data['restart_verified'] = all(web.get(f'/api/layers/{sid}').raise_for_status().json()['profile']==expected[name]
                                               for name,sid in data['sessions'].items())
            series = [row for row in data['rows'] if row['phase']=='series']
            data['series_first_violation'] = first_violation(series)
            data['status'] = 'complete' if data['restart_verified'] else 'restore_failed'
        except (httpx.HTTPError, RuntimeError, ValueError, OSError, KeyError) as exc:
            data.update(status='stopped',error=journal.scrub(str(exc)))
            journal.record('ERROR',data['error'])
        finally:
            data['budget'] = budget.accounting()
            write_results(directory,data)
            trace = []
            for line in journal.path.read_text(encoding='utf-8').splitlines():
                timestamp,event,payload = line.split(' ',2)
                trace.append({'timestamp':timestamp,'event':event,'data':json.loads(payload)})
            (directory/'trace.json').write_text(json.dumps(trace,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return directory,data['status']


def main():
    key = (load_settings(ROOT).get('POLZA_API_KEY') or '').strip()
    with Journal(ROOT/'logs',key) as journal:
        parser = LoggingParser(journal=journal,description='День 12. Проверить профиль пользователя')
        parser.add_argument('--budget-rub',type=float,default=20)
        args = parser.parse_args()
        path,status = run(budget_rub=args.budget_rub)
        message = f'{status}: {path}'
        journal.record('OUTPUT',message)
        print(message)
        return 0 if status=='complete' else 1

if __name__ == '__main__':
    raise SystemExit(main())
