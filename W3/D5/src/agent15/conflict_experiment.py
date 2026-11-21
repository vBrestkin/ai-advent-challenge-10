"""Dialogue with four corrections, request inspection and atomic task cleanup."""
import json

import httpx
from fastapi.testclient import TestClient

from .log import Journal, LoggingParser, identifier
from .metrics import Budget, TokenCounter
from .transport import Client, preflight
from .web import ROOT, create_app, load_settings

INITIAL = ('Срок: 2026-12-18', 'Версия: v1.2.0', 'Требование: формат 4K',
           'Подход: заранее записанный скринкаст',
           'Решение: Проверить звук перед демонстрацией', 'Решение: Подготовить запасной микрофон')
CHANGES = (
    ('Срок: 2026-12-22', 'Какой сейчас срок демонстрации? Ответь датой.', '2026-12-22', '2026-12-18'),
    ('Версия: v1.3.0', 'Какая версия сейчас утверждена? Ответь номером версии.', 'v1.3.0', 'v1.2.0'),
    ('Отменить требование', 'Какое сейчас требование к записи? Если оно отменено, так и скажи.', 'отменено', 'формат 4K'),
    ('Подход: живая демонстрация', 'Какой подход сейчас выбран?', 'живая демонстрация', 'заранее записанный скринкаст'),
)


def run_scenario(app, model, record, requests):
    def action(web, name, note):
        task = web.get(base+'/task').raise_for_status().json()
        return web.post(base+'/task/actions',json={'action':name,'note':note,
                        'revision':task['revision']}).raise_for_status().json()

    def update(web, command):
        response = web.post(base+'/chat',json={'message':command}).raise_for_status().json()
        if not response['fact_change']['applied']:
            raise RuntimeError('Команда не была сохранена.')
        record({'kind':'change','command':command,'change':response['fact_change']})

    with app() as web:
        sid = web.post('/api/layers',json={'title':'Маяк · актуальные факты'}).raise_for_status().json()['id']
        base = f'/api/layers/{sid}'
        web.post(base+'/task',json={'title':'Подготовить демонстрацию «Маяк»'}).raise_for_status()
        web.post(base+'/write/working',json={'key':'step','value':'Проверить микрофон'}).raise_for_status()
        web.post(base+'/write/long',json={'key':'leader','value':'Ирина'}).raise_for_status()
        action(web,'approve_plan','Проверить звук и подготовить демонстрацию.')
        for command in INITIAL:
            update(web,command)
        stale = []
        for command,question,expected,old in CHANGES:
            update(web,command)
            stale.append(old)
            start = len(model.records)
            request_count = len(requests())
            response = web.post(base+'/chat',json={'message':question})
            body = response.json()
            sent = requests()[request_count:]
            serialized = json.dumps(sent,ensure_ascii=False).casefold()
            answer = body.get('answer','')
            outdated = [value for value in stale if value.casefold() in answer.casefold()]
            stale_in_request = [value for value in stale if value.casefold() in serialized]
            record({'kind':'question','command':command,'question':question,'answer':answer,
                    'expected':expected,'status':response.status_code,'correct':expected.casefold() in answer.casefold() and not outdated,
                    'outdated_answers':outdated,'stale_in_request':stale_in_request,'requests':sent,
                    'calls':model.records[start:]})
            response.raise_for_status()
            if not sent or stale_in_request:
                raise RuntimeError('Не удалось подтвердить чистоту запроса к модели.')
        action(web,'submit_result','Демонстрация подготовлена.')
        before = web.get(base).raise_for_status().json()
        action(web,'accept_validation','Звук и экран проверены.')
        after = web.get(base).raise_for_status().json()
        if after['memory']['working'] or before['facts'] != after['facts'] or before['memory']['long'] != after['memory']['long']:
            raise RuntimeError('Нарушена очистка рабочей памяти.')

    with app() as web:
        restored = web.get(base).raise_for_status().json()
        if restored != after:
            raise RuntimeError('После перезапуска состояние не совпало.')
    record({'kind':'finish','stage':restored['task']['stage'],'working_before':before['memory']['working'],
            'working_after':restored['memory']['working'],'facts_preserved':before['facts']==restored['facts'],
            'long_before':before['memory']['long'],'long_after':restored['memory']['long'],
            'facts':restored['facts'],'restart_verified':True})
    return sid


def run(*,budget_rub=20):
    directory = ROOT/'results'/identifier()
    directory.mkdir(parents=True)
    key = (load_settings(ROOT).get('POLZA_API_KEY') or '').strip()
    budget = Budget(budget_rub)
    data = {'status':'running','events':[]}
    with Journal(ROOT/'logs',key) as journal, httpx.Client(transport=httpx.HTTPTransport(retries=0)) as http:
        def trace():
            output=[]
            for line in journal.path.read_text(encoding='utf-8').splitlines():
                timestamp,event,payload = line.split(' ',2)
                output.append({'timestamp':timestamp,'event':event,'data':json.loads(payload)})
            return output

        def requests():
            return [row['data'] for row in trace() if row['event']=='REQUEST']

        def save():
            data['budget'] = budget.accounting()
            (directory/'experiment.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

        def record(event):
            data['events'].append(event)
            journal.record('FACT_CASE',event)
            summary = f"{event['kind']}: {event.get('command',event.get('stage',''))} {event.get('answer','')}"
            journal.record('OUTPUT',summary)
            print(summary,flush=True)
            save()

        try:
            if not key:
                raise ValueError('Укажите POLZA_API_KEY.')
            model=Client(key,journal,http,counter=TokenCounter(ROOT/'data/tokenizer.json'),budget=budget)
            data['catalog']=preflight(model)
            def app():
                return TestClient(create_app(root=ROOT,model=model,data_dir=directory/'data'))
            data['workspace']=run_scenario(app,model,record,requests)
            data['status']='complete'
        except (httpx.HTTPError,RuntimeError,ValueError,OSError,KeyError) as exc:
            data.update(status='stopped',error=journal.scrub(str(exc)))
            journal.record('ERROR',data['error'])
        finally:
            save()
            (directory/'trace.json').write_text(json.dumps(trace(),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return directory,data['status']


def main():
    key=(load_settings(ROOT).get('POLZA_API_KEY') or '').strip()
    with Journal(ROOT/'logs',key) as journal:
        parser=LoggingParser(journal=journal,description='День 15. Проверка конфликтов памяти')
        parser.add_argument('--budget-rub',type=float,default=20)
        args=parser.parse_args()
        path,status=run(budget_rub=args.budget_rub)
        message=f'{status}: {path}'
        journal.record('OUTPUT',message)
        print(message)
        return 0 if status=='complete' else 1


if __name__=='__main__':
    raise SystemExit(main())
