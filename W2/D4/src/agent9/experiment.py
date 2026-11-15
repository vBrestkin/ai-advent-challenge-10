"""One canonical dialogue, progressive summary updates, independent paired recall quizzes."""
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import re
import httpx
from fastapi.testclient import TestClient
from .agent import Agent, AgentError, SYSTEM
from .memory import Memory, compact, context
from .log import Journal, LoggingParser, identifier
from .metrics import Budget, TokenCounter, aggregate, TOKENIZER_SHA256, TOKENIZER_URL
from .transport import Client, preflight, MODEL, PROVIDER
from .tools import TOOLS
from .web import ROOT, create_app, load_settings, PROFILE

# Frozen before any model request. Expected answers are never included in summary or quiz inputs.
PROMPTS = [
 'Проект называется «Маяк», руководитель — Ирина. Найди сумму заказа A-104 и вычисли цену после скидки 15%. Это контрольная сумма проекта. Ответь кратко.',
 'Запомни дату демонстрации: 18 декабря 2026 года. Бюджет проекта сейчас 180000 рублей. Подтверди кратко.',
 'Для хранения диалогов выбрана SQLite. Развёртывание разрешено только в России. Подтверди кратко.',
 'Ограничение: нельзя отправлять персональные данные во внешние облака. Решение по интерфейсу: локальный веб-чат. Подтверди кратко.',
 'Перейдём к оформлению: фон светлый, крупный текст, понятные подписи кнопок. Обсуди одним предложением.',
 'Для показа нужны примеры простого запроса и запроса с инструментом. Предложи порядок в одном предложении.',
 'Добавим раздел документации о запуске и остановке сервера. Подтверди одним предложением.',
 'Важное исправление: бюджет увеличен с 180000 до 210000 рублей. Актуальный бюджет — 210000 рублей. Остальные договорённости прежние. Подтверди кратко.',
 'На репетиции проверим восстановление диалога после перезапуска. Ответь одним предложением.',
 'Для записи видео подготовим чистое окно браузера и терминал. Подтверди одним предложением.',
 'Перед показом прогоним автотесты и проверим логи. Подтверди одним предложением.',
 'Подготовка завершена. Дальше я задам вопросы о конкретных прежних договорённостях. Ответь только «Готово».',
]
FACTS = [
 {'id':'project','question':'Как называется проект?','answers':['Маяк']},
 {'id':'manager','question':'Как зовут руководителя проекта?','answers':['Ирина']},
 {'id':'deadline','question':'Какова дата демонстрации? Используй YYYY-MM-DD.','answers':['2026-12-18']},
 {'id':'budget','question':'Каков актуальный бюджет в рублях? Ответь числом без валюты.','answers':['210000']},
 {'id':'storage','question':'Какую базу данных выбрали для хранения диалогов?','answers':['SQLite']},
 {'id':'region','question':'В какой стране разрешено развёртывание?','answers':['Россия','в России','России']},
 {'id':'constraint','question':'Какие данные запрещено отправлять во внешние облака?','answers':['персональные данные']},
 {'id':'interface','question':'Какой интерфейс решили сделать?','answers':['локальный веб-чат','локальный веб чат']},
 {'id':'order_amount','question':'Какая исходная сумма заказа A-104 уже была получена в этой беседе? Только число.','answers':['2390','2390.00']},
 {'id':'discount_amount','question':'Какая контрольная сумма после скидки 15% уже была вычислена? Только число.','answers':['2031.5','2031.50']},
]


def grade(answer, expected):
    try:
        value=json.loads(answer)
        if not isinstance(value,dict) or set(value)!={'value'} or not isinstance(value['value'],(str,int,float)) or isinstance(value['value'],bool): return False
        normalize=lambda s: re.sub(r'\s+',' ',str(s).strip().lower().replace('ё','е'))
        return normalize(value['value']) in {normalize(item) for item in expected}
    except (ValueError,TypeError): return False


def csv_text(rows):
    if not rows:return ''
    stream=io.StringIO(newline=''); writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(rows);return stream.getvalue()


def write_report(directory,data):
    directory=Path(directory)
    (directory/'experiment.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    summaries=[]
    maintenance=aggregate([call for call in data['calls'] if call.get('purpose')=='summary'])
    preparation=aggregate([call for call in data['calls'] if call.get('phase')=='dialogue'])
    for mode in ('full','summary'):
        rows=[row for row in data['quizzes'] if row['mode']==mode]
        metrics=aggregate([call for call in data['calls'] if call.get('phase')=='quiz' and call.get('mode')==mode])
        summaries.append({'mode':mode,'details_correct':sum(row['passed'] for row in rows),'details_tested':len(rows),
            'quiz_input_tokens':metrics['prompt_tokens'],'quiz_output_tokens':metrics['completion_tokens'],
            'quiz_total_tokens':metrics['total_tokens'],'quiz_calls':metrics['calls'], 'quiz_cost_rub':metrics['cost_rub'],
            'maintenance_input_tokens':maintenance['prompt_tokens'] if mode=='summary' else 0,
            'maintenance_output_tokens':maintenance['completion_tokens'] if mode=='summary' else 0,
            'maintenance_cost_rub':maintenance['cost_rub'] if mode=='summary' else 0,
            'cost_with_maintenance_rub':(metrics['cost_rub'] or 0)+((maintenance['cost_rub'] or 0) if mode=='summary' else 0),
            'unknown_costs':metrics['unknown_costs']+(maintenance['unknown_costs'] if mode=='summary' else 0)})
    (directory/'summary.csv').write_text(csv_text(summaries),encoding='utf-8-sig')
    quiz_rows=[{key:row.get(key) for key in ('mode','id','question','answer','passed','error')} for row in data['quizzes']]
    (directory/'details.csv').write_text(csv_text(quiz_rows),encoding='utf-8-sig')
    budget=data['budget']
    lines=['# День 9 — сжатие истории', '',f"Статус: {data['status']}. Модель `{MODEL}`, запрошенный маршрут `{PROVIDER}`.", '',
        'Один общий диалог; ожидаемые ответы зафиксированы в FACTS до запуска. Каждый вопрос проверяется независимо от той же исходной истории.', '',
        f"Архив: {len(data['archive'])} сообщений всех ролей без system. Summary покрывает {data['memory']['covered']} сообщений; минимум 6 последних остаются целиком. Порог обновления — 10 несжатых сообщений.", '',
        '| Режим | Верно / проверено | Вход тестов API | Выход тестов API | Токены сжатия (вход + выход) | Цена тестов, ₽ | Цена сжатия, ₽ | Всего с обслуживанием, ₽ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        display=lambda value:'—' if value is None else str(round(value,8)) if isinstance(value,float) else str(value)
        lines.append('| '+ ' | '.join([row['mode'],f"{row['details_correct']} / {row['details_tested']}",display(row['quiz_input_tokens']),display(row['quiz_output_tokens']),
            f"{display(row['maintenance_input_tokens'])} + {display(row['maintenance_output_tokens'])}",display(row['quiz_cost_rub']),display(row['maintenance_cost_rub']),display(row['cost_with_maintenance_rub'])])+' |')
    lines+=['',f"Общая подготовка исходного диалога (одна на оба режима): {preparation['calls']} вызовов, вход {preparation['prompt_tokens']}, выход {preparation['completion_tokens']} токенов, {preparation['cost_rub']} ₽.", '',
        '## Размер контекста до и после', '', '```json',json.dumps(data['context_sizes'],ensure_ascii=False,indent=2), '```', '',
        'Это локальная оценка JSON со схемами инструментов и system; серверное оформление отличается. В таблице выше — usage API, включая все инструментальные раунды.', '',
        '## Что потерялось', '']
    for mode in ('full','summary'):
        rows=[row for row in data['quizzes'] if row['mode']==mode]
        lost=[row['id'] for row in rows if not row['passed']]
        lines.append(f"- {mode}: "+(', '.join(lost) if lost else 'ошибок среди завершённых проверок не обнаружено')+f"; проверено {len(rows)}/10.")
    if len(data['quizzes'])==20:
        full,compressed=summaries
        if full['quiz_input_tokens'] and compressed['quiz_input_tokens'] is not None:
            percent=(1-compressed['quiz_input_tokens']/full['quiz_input_tokens'])*100
            lines+=['',f"Входные токены проверок сократились на {percent:.2f}%."]
        if full['cost_with_maintenance_rub'] and not any(row['unknown_costs'] for row in summaries):
            percent=(1-compressed['cost_with_maintenance_rub']/full['cost_with_maintenance_rub'])*100
            lines+=[f"Изменение цены проверок с учётом сжатия: экономия {percent:.2f}%. Общая подготовка диалога в это сравнение не включается."]
        for row in data['quizzes']:
            if not row['passed']:
                lines+=['',f"Ответ {row['mode']} / {row['id']}: `{row['answer']}`. Эталон: "+json.dumps(next(f['answers'] for f in FACTS if f['id']==row['id']),ensure_ascii=False)+'.']
    lines+=['', 'Полные ответы и ожидаемые варианты — experiment.json и details.csv. Оценка строгая: JSON с единственным полем value и точное совпадение после нормализации регистра и пробелов. Разбор ошибок по смыслу следует сверять с ответами.', '',
        '## Условия и выводы', '',
        '- Полный архив остаётся в SQLite. Сжатие меняет только отправляемый контекст; summary и граница покрытого префикса сохраняются отдельно.',
        '- При повторном сжатии в модель поступают прежняя сводка и только новый старый префикс. Вызовы инструментов с результатами не разрываются.',
        '- Сжатие требует дополнительных запросов и токенов. На короткой беседе затраты на сводки могут превышать экономию последующих запросов.',
        '- Это один фиксированный сценарий и один прогон, а не гарантия сохранения любых деталей. Проверки независимы: ответы не добавляются в контекст следующего вопроса.',
        '- Проверки исходной суммы и скидки допускают инструменты. trace и calls показывают, использовала ли модель память или повторно вычислила ответ.',
        '- Неизвестный usage не считается нулём; неизвестная стоимость удерживается резервом бюджета. Автоматических повторов и fallback нет.', '',
        f"Бюджет: известно {budget['known_cost_rub']} ₽; резерв неизвестных списаний {budget['unknown_reserve_rub']} ₽; учтено {budget['accounted_rub']} / {budget['limit_rub']} ₽.", '']
    if data.get('error'):lines+=['Ошибка остановки: '+data['error'],'']
    (directory/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def run(root=ROOT,*,budget_rub=20):
    root=Path(root); budget=Budget(budget_rub); directory=root/'results'/identifier();directory.mkdir(parents=True)
    data={'status':'initializing','calls':[],'quizzes':[],'archive':[],'memory':Memory().to_dict(),'context_sizes':{},
          'budget':budget.accounting(),'facts':FACTS,'prompts':PROMPTS,'summaries':[]}
    def save():
        data['budget']=budget.accounting();write_report(directory,data)
    save()
    key=(load_settings(root).get('POLZA_API_KEY') or '').strip()
    with Journal(root/'logs',key) as journal,httpx.Client(transport=httpx.HTTPTransport(retries=0)) as http:
        model=Client(key,journal,http,counter=TokenCounter(root/'data/tokenizer.json'),budget=budget)
        def record_new(start,phase,**labels):
            data['calls'].extend([{**deepcopy(call),'phase':phase,**labels} for call in model.records[start:]])
            save()
        try:
            if not key or not model.counter.exact_text:raise RuntimeError('Нужны POLZA_API_KEY и фиксированный токенизатор.')
            data['catalog']=preflight(model)
            data['tokenizer']={'url':TOKENIZER_URL,'sha256':TOKENIZER_SHA256,'method':model.counter.method}
            database=directory/'data'
            with TestClient(create_app(root=root,model=model,data_dir=database)) as client:
                sid=''; snapshots=[]
                for index,prompt in enumerate(PROMPTS):
                    start=len(model.records)
                    response=client.post('/api/chat',json={'message':prompt,'session_id':sid,'mode':'full'})
                    record_new(start,'dialogue',turn=index+1)
                    response.raise_for_status();result=response.json();sid=result['session_id']
                    state=client.get('/api/sessions/'+sid).json();snapshots.append(deepcopy(state));data['archive']=state['messages'];save()
                    if index==0 and [step['tool'] for step in result['steps']]!=['find_order','calculate']:
                        raise RuntimeError('Не пройден контроль двух инструментов.')
                    journal.record('PROGRESS',f'dialogue {index+1}/12');print(f'dialogue {index+1}/12',flush=True)
                archive=state['messages']; data['archive']=archive
                if not 20<=len(archive)<=30:raise RuntimeError('Число сообщений вне диапазона 20–30.')
                store=client.app.state.store; summary_sid=store.create('Сжатая копия: Маяк',PROFILE);memory=Memory()
                for index,snapshot in enumerate(snapshots):
                    attempt=store.begin(summary_sid,PROMPTS[index]);start=len(model.records)
                    def observe(event,record,call_id):
                        if event=='start':return store.begin_call(attempt,record)
                        store.finish_call(call_id,record)
                    model.observer=observe
                    try:updated=compact(snapshot['messages'],memory,model)
                    finally:
                        model.observer=None;record_new(start,'maintenance')
                    if updated!=memory:
                        data['summaries'].append({'after_messages':len(snapshot['messages']),**updated.to_dict()})
                    memory=updated
                    store.complete(attempt,snapshot['messages'],snapshot['turns'][-1],memory=memory.to_dict())
                    data['memory']=memory.to_dict();save()
                data['session_ids']={'full':sid,'summary':summary_sid}
            # Reopen SQLite through a fresh app lifespan; assert exact memory and archive restoration.
            with TestClient(create_app(root=root,model=model,data_dir=database)) as client:
                restored=client.get('/api/sessions/'+summary_sid).json()
                if restored['memory']!=memory.to_dict() or restored['messages']!=archive:raise RuntimeError('Память не восстановлена.')
                data['restart_verified']=True
                system=[{'role':'system','content':SYSTEM}]
                data['context_sizes']={'full_estimate':model.counter.estimate(system+archive,TOOLS),
                    'summary_estimate':model.counter.estimate(system+context(archive,memory),TOOLS),
                    'archive_messages':len(archive),'tail_messages':len(archive)-memory.covered}
                for fact in FACTS:
                    for mode in ('full','summary'):
                        agent=Agent(model,context_builder=(lambda messages:context(messages,memory)) if mode=='summary' else None)
                        agent.messages=deepcopy(archive);start=len(model.records)
                        question=fact['question']+' Верни только JSON вида {"value":"ответ"}, без других полей и Markdown. Если не помнишь, value="неизвестно".'
                        row={'mode':mode,'id':fact['id'],'question':question,'answer':'','passed':False,'error':None}
                        try:
                            result=agent.run(question);row.update(answer=result['answer'],steps=result['steps'])
                            row['passed']=grade(result['answer'],fact['answers'])
                        except AgentError as exc:row['error']=str(exc)
                        finally:
                            data['quizzes'].append(row);record_new(start,'quiz',mode=mode,fact_id=fact['id'])
                        journal.record('QUIZ',row);print(f"{mode} {fact['id']}: {row['passed']}",flush=True)
                        if row['error']:raise RuntimeError(row['error'])
            data['status']='complete'
        except (RuntimeError,AgentError,OSError,ValueError,httpx.HTTPError) as exc:
            data['status']='stopped';data['error']=journal.scrub(str(exc));journal.record('ERROR',data['error'])
        finally:
            save()
            # Redacted request/response trace for inspectable results, no .log required in Git.
            trace=[]
            for line in journal.path.read_text(encoding='utf-8').splitlines():
                timestamp,event,payload=line.split(' ',2)
                trace.append({'timestamp':timestamp,'event':event,'data':json.loads(payload)})
            (directory/'trace.json').write_text(json.dumps(trace,ensure_ascii=False,indent=2),encoding='utf-8')
    return directory


def main():
    key=(load_settings(ROOT).get('POLZA_API_KEY') or '').strip()
    with Journal(ROOT/'logs',key) as journal:
        parser=LoggingParser(journal=journal,description='День 9: общий диалог, summary и независимая проверка 10 деталей')
        parser.add_argument('--budget-rub',type=float,default=20);args=parser.parse_args()
        try:
            directory=run(budget_rub=args.budget_rub);status=json.loads((directory/'experiment.json').read_text())['status']
            message=f'Отчёт: {directory / "report.md"}; статус: {status}'
            journal.record('STDOUT',message);print(message);return 0 if status=='complete' else 1
        except (ValueError,OSError) as exc:
            journal.record('ERROR',str(exc));print('Эксперимент остановлен, проверьте журнал.');return 1

if __name__=='__main__':raise SystemExit(main())
