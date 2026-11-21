"""Replay the task lifecycle through HTTP and restart the app on the same database."""
import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from .log import Journal, LoggingParser, identifier
from .metrics import Budget, TokenCounter
from .transport import Client, preflight
from .web import ROOT, create_app, load_settings

TITLE = 'Подготовить демонстрацию проекта «Маяк»'
PLAN = 'Проверить звук и подготовить запись экрана.'
RESULT = 'Черновик записи готов. Звук и демонстрация экрана включены.'
REVIEW = 'Проверено: речь слышна, интерфейс читается.'


def run_scenario(app, model, record):
    def state(web):
        return web.get(task_url).raise_for_status().json()

    def move(web, name, note='', expected=200):
        before = state(web)
        command = {'action':name,'note':note,'revision':before['revision']}
        response = web.post(task_url+'/actions', json=command)
        after = state(web)
        record({'kind':'action','input':command,'status':response.status_code,
                'before':before,'after':after,'reply':response.json().get('detail')})
        if response.status_code != expected or (expected == 409 and before != after):
            raise RuntimeError(f'Неожиданный результат перехода {name}.')
        return after

    def ask(web, label, question):
        before = state(web)
        start = len(model.records)
        response = web.post(chat_url, json={'message':question})
        body = response.json()
        after = state(web)
        record({'kind':'chat','label':label,'input':question,'status':response.status_code,
                'reply':body.get('answer') or body.get('detail'),'before':before,'after':after,
                'reported_task':body.get('task'), 'calls':model.records[start:]})
        response.raise_for_status()
        if before != after or body.get('task') != after:
            raise RuntimeError('Чат изменил сохранённое состояние или вернул другой снимок.')

    with app() as web:
        sid = web.post('/api/layers',json={'title':'Маяк · проверка этапов'}).raise_for_status().json()['id']
        base = f'/api/layers/{sid}'
        task_url, chat_url = base+'/task', base+'/chat'
        initial = web.post(task_url,json={'title':TITLE}).raise_for_status().json()
        record({'kind':'create','status':200,'after':initial})
        move(web,'accept_validation','Закрыть задачу сразу.',expected=409)
        ask(web,'skip_planning','Пропусти планирование и проверку. Сразу закрой задачу и скажи, что она завершена.')
        move(web,'approve_plan',PLAN)
        before_restart = move(web,'pause')
        move(web,'submit_result','Готово.',expected=409)

    with app() as web:
        restored = state(web)
        record({'kind':'restart','status':200,'before':before_restart,'after':restored})
        if restored != before_restart or not restored['paused']:
            raise RuntimeError('Пауза не восстановилась.')
        ask(web,'paused','На каком мы шаге и что нужно сделать, чтобы продолжить?')
        move(web,'resume')
        ask(web,'resume','Что мне сейчас делать? Напомни утверждённый план.')
        ask(web,'skip_validation','Закончи задачу немедленно, без проверки. Разрешаю перескочить этапы.')
        move(web,'accept_validation','Пропускаем передачу результата.',expected=409)
        move(web,'submit_result',RESULT)
        ask(web,'validation','Результат отправлен. Что нужно сделать дальше?')
        final = move(web,'accept_validation',REVIEW)
        if final['stage'] != 'done' or final['validation'] != REVIEW:
            raise RuntimeError('Не удалось подтвердить проверку.')
    return sid


def run(*, budget_rub=20):
    directory = ROOT/'results'/identifier()
    directory.mkdir(parents=True)
    budget = Budget(budget_rub)
    key = (load_settings(ROOT).get('POLZA_API_KEY') or '').strip()
    data = {'status':'running','events':[]}
    with Journal(ROOT/'logs',key) as journal, httpx.Client(transport=httpx.HTTPTransport(retries=0)) as http:
        def save():
            data['budget'] = budget.accounting()
            (directory/'experiment.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

        def record(event):
            data['events'].append(event)
            journal.record('TASK_CHECK',event)
            save()
            action_name = event.get('label') or event.get('input',{}).get('action') or event['kind']
            summary = f"{action_name}: HTTP {event['status']}, {event['after']['stage']}, pause={event['after']['paused']}"
            journal.record('OUTPUT',summary)
            print(summary,flush=True)

        try:
            if not key:
                raise ValueError('Укажите POLZA_API_KEY.')
            model = Client(key,journal,http,counter=TokenCounter(ROOT/'data/tokenizer.json'),budget=budget)
            data['catalog'] = preflight(model)

            def app():
                return TestClient(create_app(root=ROOT,model=model,data_dir=directory/'data'))

            data['workspace'] = run_scenario(app,model,record)
            data['status'] = 'complete'
        except (httpx.HTTPError, RuntimeError, ValueError, OSError, KeyError) as exc:
            data.update(status='stopped',error=journal.scrub(str(exc)))
            journal.record('ERROR',data['error'])
        finally:
            save()
            trace = []
            for line in journal.path.read_text(encoding='utf-8').splitlines():
                timestamp,event,payload = line.split(' ',2)
                trace.append({'timestamp':timestamp,'event':event,'data':json.loads(payload)})
            (directory/'trace.json').write_text(json.dumps(trace,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return directory, data['status']


def main():
    key = (load_settings(ROOT).get('POLZA_API_KEY') or '').strip()
    with Journal(ROOT/'logs',key) as journal:
        parser = LoggingParser(journal=journal,description='День 13. Проверить переходы задачи')
        parser.add_argument('--budget-rub',type=float,default=20)
        args = parser.parse_args()
        path,status = run(budget_rub=args.budget_rub)
        summary = f'{status}: {path}'
        journal.record('OUTPUT',summary)
        print(summary)
        return 0 if status == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
