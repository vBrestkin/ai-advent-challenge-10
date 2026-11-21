"""Compare each removed layer against the same original context."""
import json
from pathlib import Path
import httpx
from fastapi.testclient import TestClient
from .agent import SYSTEM
from .layers import Layers, LAYER_NAMES
from .log import Journal, LoggingParser, identifier
from .metrics import Budget, TokenCounter
from .scenario import grade, csv_text
from .tools import TOOLS
from .transport import Client, preflight
from .web import ROOT, create_app, load_settings

QUESTIONS = {
    'short': ('Какую фразу мы только что выбрали для начала видео?', 'Зажигаем маяк'),
    'working': ('Какой шаг подготовки сейчас выполняем?', 'проверить микрофон'),
    'long': ('Кто руководит проектом «Маяк»?', 'Ирина'),
}
TITLES = {'all': 'Маяк — все слои', 'short': 'Маяк — без диалога',
          'working': 'Маяк — без рабочей памяти', 'long': 'Маяк — без знаний проекта'}


def run_matrix(web, model, project_facts, on_result):
    seed = {'short': [
        {'role': 'user', 'content': 'Для начала видео выбрали фразу «Зажигаем маяк».'},
        {'role': 'assistant', 'content': 'Принято.'}],
        'working': {'current_step': 'проверить микрофон'}, 'long': project_facts}
    source = web.post('/api/layers', json={'title': TITLES['all'], 'memory': seed}).raise_for_status().json()['id']
    states = {}
    for mode in ('all', *LAYER_NAMES):
        sid = source
        if mode != 'all':
            path = f'/api/layers/{source}/copy'
            sid = web.post(path).raise_for_status().json()['id']
            web.post(f'/api/layers/{sid}/clear/{mode}').raise_for_status()
            with web.app.state.store.transaction() as db:
                db.execute('UPDATE layer_workspaces SET title=? WHERE id=?', (TITLES[mode], sid))
            web.app.state.journal.record('LAYERS_LABEL', {'id':sid,'title':TITLES[mode]})
        states[mode] = web.get(f'/api/layers/{sid}').raise_for_status().json()

    for key, (text, value) in QUESTIONS.items():
        question = text + ' Ответь только JSON с полем value, без Markdown. Если данных нет, укажи «неизвестно».'
        for mode, state in states.items():
            first = len(model.records)
            url = f"/api/layers/{state['id']}"
            response = web.post(url+'/chat?save=false', json={'message': question})
            answer = response.json()
            calls = model.records[first:]
            expected = 'неизвестно' if mode == key else value
            messages = [{'role': 'system', 'content': SYSTEM}] + Layers.context(
                state['memory'],state['memory']['short']+[{'role':'user','content':question}])
            unchanged = web.get(url).raise_for_status().json() == state
            row = {'mode':mode,'question_id':key,'question':question,'expected':expected,
                   'answer':answer.get('answer'),'passed': response.is_success and not answer['steps'] and grade(answer['answer'], [expected]),
                   'unchanged':unchanged,'prompt_tokens':calls[0]['prompt_tokens'] if calls else None,
                   'local_estimate':model.counter.estimate(messages, TOOLS),
                   'http_status':response.status_code,'error':answer.get('detail'),
                   'messages':messages,'calls':calls,'steps':answer.get('steps',[])}
            on_result(row)
            response.raise_for_status()
            if not unchanged:
                raise RuntimeError('Контрольный вопрос изменил память.')
    return states


def write_results(directory, data):
    (directory/'experiment.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    fields = ('mode','question_id','expected','answer','passed','unchanged','prompt_tokens','local_estimate')
    (directory/'details.csv').write_text(csv_text([{k:r[k] for k in fields} for r in data['rows']]), encoding='utf-8-sig')


def run(root=ROOT, *, budget_rub=20):
    root = Path(root)
    source = root.parents[1]/'W2/D5/submission/experiment.json'
    directory = root/'results'/identifier()
    directory.mkdir(parents=True)
    budget = Budget(budget_rub)
    key = (load_settings(root).get('POLZA_API_KEY') or '').strip()
    data = {'status':'running','rows':[],'source':'W2/D5/submission/experiment.json',
            'states':{},'budget':budget.accounting()}
    with Journal(root/'logs',key) as journal, httpx.Client(transport=httpx.HTTPTransport(retries=0)) as http:
        try:
            source_data = json.loads(source.read_text(encoding='utf-8'))
            stored_facts = source_data['states']['facts']['facts_state']['values']
            project_facts = {name: stored_facts[name] for name in ('project_name','project_leader')}
            if project_facts['project_leader'] != QUESTIONS['long'][1] or not key:
                raise ValueError('Проверьте факт руководителя из дня 10 и настройку POLZA_API_KEY.')
            counter = TokenCounter(root/'data/tokenizer.json')
            if not counter.exact_text:
                raise ValueError('Установите токенизатор через agent11.setup_tokenizer.')
            model = Client(key,journal,http,counter=counter,budget=budget)
            data['catalog'] = preflight(model)
            data['tokenizer_method'] = counter.method
            def save_row(row):
                data['rows'].append(row)
                data['budget'] = budget.accounting()
                write_results(directory,data)
                message = f"{row['mode']} / {row['question_id']}: {row['answer']} | вход API {row['prompt_tokens']}"
                journal.record('MATRIX_RESULT', row)
                print(message, flush=True)
            with TestClient(create_app(root=root,model=model,data_dir=directory/'data')) as web:
                data['states'] = run_matrix(web,model,project_facts,save_row)
            with TestClient(create_app(root=root,model=model,data_dir=directory/'data')) as web:
                data['restart_verified'] = all(web.get(f"/api/layers/{row['id']}").json()==row
                                               for row in data['states'].values())
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
        parser = LoggingParser(journal=journal, description='День 11. Проверить три слоя памяти')
        parser.add_argument('--budget-rub',type=float,default=20)
        args = parser.parse_args()
        path,status = run(budget_rub=args.budget_rub)
        message = f'{status}: {path}'
        journal.record('OUTPUT',message)
        print(message)
        return 0 if status=='complete' else 1

if __name__ == '__main__':
    raise SystemExit(main())
