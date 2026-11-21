"""Paired strategy comparison using the exact frozen day-9 dialogue, plus real independent branches."""
from copy import deepcopy
import ast
import hashlib
import json
from pathlib import Path
import httpx
from fastapi.testclient import TestClient
from .agent import Agent,AgentError,SYSTEM
from .memory import Memory,compact
from .strategies import Facts,update_facts,select_context
from .store import StoreError,validate_messages
from .log import Journal,LoggingParser,identifier
from .metrics import Budget,TokenCounter,aggregate,TOKENIZER_SHA256,TOKENIZER_URL
from .transport import Client,preflight,MODEL,PROVIDER
from .tools import TOOLS
from .web import ROOT,create_app,load_settings,PROFILE
from .scenario import PROMPTS,FACTS,grade,csv_text

MODES=('window','facts','summary')


def complexity():
    groups={'window':('strategies.py',{'window'}),
            'facts':('strategies.py',{'Facts','FACTS_SYSTEM','FACTS_RESPONSE_FORMAT','validate_values','validate_facts','update_facts'}),
            'summary':('memory.py',{'Memory','SUMMARY_SYSTEM','validate_memory','context','compact'})}
    result={}
    for mode,(filename,names) in groups.items():
        source=Path(__file__).with_name(filename).read_text();lines=source.splitlines();selected=set()
        for node in ast.parse(source).body:
            name=getattr(node,'name',None)
            if isinstance(node,ast.Assign):name=getattr(node.targets[0],'id',None)
            if name in names:
                start=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])
                selected.update(range(start-1,node.end_lineno))
        result[mode]={'nonblank_lines':sum(bool(lines[i].strip()) for i in selected),'file':filename,
            'complexity':{'window':'низкая','facts':'высокая','summary':'средняя'}[mode]}
    return result


def load_source(path):
    payload=Path(path).read_bytes();source=json.loads(payload);archive=source['archive'];validate_messages(archive)
    if not 20<=len(archive)<=30:raise ValueError('Нужен исходный диалог на 20–30 сообщений.')
    if [m['content'] for m in archive if m['role']=='user']!=PROMPTS:raise ValueError('Сценарий отличается от дня 9.')
    if source.get('facts')!=FACTS:raise ValueError('Эталоны отличаются от дня 9.')
    return source,hashlib.sha256(payload).hexdigest()


def turn_view(chunk):
    calls={};steps=[]
    for message in chunk:
        for call in message.get('tool_calls') or []:
            calls[call['id']]={'id':call['id'],'tool':call['function']['name'],'arguments':json.loads(call['function']['arguments'])}
        if message['role']=='tool':steps.append({**calls[message['tool_call_id']],'result':json.loads(message['content'])})
    return {'user':chunk[0]['content'],'answer':chunk[-1]['content'],'steps':steps,'imported':True}


def summary_rows(data):
    code=complexity();rows=[]
    for mode in MODES:
        quizzes=[q for q in data['quizzes'] if q['mode']==mode]
        calls=[c for c in data['calls'] if c.get('mode')==mode and c.get('phase')!='branch']
        answers=aggregate([c for c in calls if c.get('purpose')=='chat'])
        maintenance=aggregate([c for c in calls if c.get('purpose')!='chat'])
        total=aggregate(calls)
        maintenance_value=lambda field:0 if maintenance['calls']==0 else maintenance[field]
        rows.append({'mode':mode,'correct':sum(q['passed'] for q in quizzes),'tested':len(quizzes),
            'correct_without_tools':sum(q['passed'] and not q.get('steps') for q in quizzes),
            'correct_with_tools':sum(q['passed'] and bool(q.get('steps')) for q in quizzes),
            'answer_input_tokens':answers['prompt_tokens'],'answer_output_tokens':answers['completion_tokens'],
            'maintenance_calls':maintenance['calls'],'maintenance_input_tokens':maintenance_value('prompt_tokens'),
            'maintenance_output_tokens':maintenance_value('completion_tokens'),'maintenance_cost_rub_known':maintenance_value('cost_rub'),
            'all_calls':total['calls'],'all_tokens_known':total['total_tokens'],'all_cost_rub_known':total['cost_rub'],
            'unknown_costs':total['unknown_costs'],'unknown_total_tokens':total['unknown_total_tokens'],
            **code[mode]})
    return rows


def write_report(directory,data):
    directory=Path(directory)
    (directory/'experiment.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=summary_rows(data)
    (directory/'summary.csv').write_text(csv_text(rows),encoding='utf-8-sig')
    details=[{key:q.get(key) for key in ('mode','id','question','answer','passed','error')} for q in data['quizzes']]
    (directory/'details.csv').write_text(csv_text(details),encoding='utf-8-sig')
    show=lambda value:'—' if value is None else str(round(value,8)) if isinstance(value,float) else str(value)
    lines=['# День 10 — стратегии контекста и ветвление','',f"Статус: {data['status']}. Модель `{MODEL}`, запрошенный маршрут `{PROVIDER}`.",'',
        f"Исходный диалог дня 9: {len(data['archive'])} сообщений всех ролей без system. Он импортирован без новых генераций, поэтому исходные ответы идентичны для всех стратегий.",'',
        '| Стратегия | Верно / проверено | Верно без инструментов | Вход ответов API | Выход ответов API | Обслуживание: вызовов | Всего токенов с обслуживанием, известно | Цена с обслуживанием, известно, ₽ | Код стратегии, строк | Сложность |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in rows:
        lines.append('| '+' | '.join([row['mode'],f"{row['correct']} / {row['tested']}"]+[show(row[k]) for k in ('correct_without_tools','answer_input_tokens','answer_output_tokens','maintenance_calls','all_tokens_known','all_cost_rub_known','nonblank_lines','complexity')])+' |')
    lines+=['','## Цена обслуживания','',
        '| Стратегия | Вход | Выход | Цена, известно, ₽ | Неизвестных списаний всего |', '|---|---:|---:|---:|---:|']
    for row in rows:lines.append('| '+' | '.join(show(row[k]) for k in ('mode','maintenance_input_tokens','maintenance_output_tokens','maintenance_cost_rub_known','unknown_costs'))+' |')
    lines+=['', 'В обслуживание включены построение памяти на исходном диалоге и её обновления на проверочных вопросах. Sticky Facts обновляются перед **каждым** пользовательским сообщением, в том числе вопросом проверки. Нулевое число вызовов означает отсутствие расходов; отсутствующий usage не считается нулём.', '',
        '## Сохранённые детали и ошибки','']
    for mode in MODES:
        quizzes=[q for q in data['quizzes'] if q['mode']==mode]
        lost=[q['id'] for q in quizzes if not q['passed']]
        assisted=[q['id'] for q in quizzes if q['passed'] and q.get('steps')]
        lines.append(f"- {mode}: неверные ответы — {', '.join(lost) or 'нет среди завершённых'}; верные с инструментами — {', '.join(assisted) or 'нет'}.")
    for q in data['quizzes']:
        if not q['passed']:lines.append(f"- {q['mode']} / {q['id']}: `{q['answer']}`; ошибка API: {q.get('error') or 'нет'}.")
    lines+=['', 'Верный ответ после повторного вызова инструмента не доказывает сохранение детали в памяти. Поэтому верные без инструментов показаны отдельно. Оценка строгая: JSON с единственным value и совпадение с заранее заданными вариантами; исходные ответы и эталоны доступны в experiment.json.', '',
        '## Размер контекста: локальная оценка','', '```json',json.dumps(data['context_sizes'],ensure_ascii=False,indent=2),'```','',
        'Оценка JSON с system и схемами отличается от chat template провайдера. Таблицы расходов используют usage API; reasoning не прибавляется второй раз.', '',
        '## Checkpoint и две ветки','', '```json',json.dumps(data['branches'],ensure_ascii=False,indent=2),'```','',
        'Checkpoint содержит неизменяемый снимок архива, profile, summary и facts. Обе ветки начинаются от одной точки. Новые вызовы и их стоимость не копируются в соседнюю ветку. Архив и память родителя проверяются после продолжения обеих веток; перезапуск приложения должен восстановить все снимки.', '',
        '## Сложность и применимость','',
        '- Sliding Window: минимальный код и нет LLM-вызовов обслуживания. Подходит коротким локальным обсуждениям; ранние детали из запроса исчезают.',
        '- Sticky Facts: явные актуальные значения полезны для бюджетов, сроков и договорённостей. Экстракция после каждого пользовательского сообщения добавляет задержку и расходы; ошибки извлечения тоже возможны.',
        '- Summary: сохраняет общий ход обсуждения и требует меньше обновлений, но может смешивать или ослаблять отдельные детали.',
        '- Количество строк измерено по AST: непустые строки функций, классов и промпта конкретной стратегии. Общие Agent, SQLite, API, интерфейс, fingerprint и boundaries исключены. Это воспроизводимая оценка локальной сложности, а не полный размер реализации.',
        '- Все вопросы независимы: берётся одна исходная история, ответы предыдущих проверок не добавляются. Эталоны не передаются в память или проверочные запросы. Порядок стратегий циклически меняется между вопросами.',
        '- Один фиксированный сценарий и один прогон не дают статистической гарантии. Модель, параметры и запрошенный маршрут одинаковы; автоматических повторов и fallback нет. Полный архив сохраняется при всех стратегиях.', '']
    branch_metrics=aggregate([c for c in data['calls'] if c.get('phase')=='branch'])
    budget=data['budget']
    lines+=[f"Проверка веток отдельно: {branch_metrics['calls']} вызовов, известная стоимость {branch_metrics['cost_rub']} ₽.", '',
        f"Общий бюджет: известно {budget['known_cost_rub']} ₽; резерв неизвестных списаний {budget['unknown_reserve_rub']} ₽; учтено {budget['accounted_rub']} из {budget['limit_rub']} ₽.",'']
    if data.get('error'):lines+=['Ошибка остановки: '+data['error'],'']
    (directory/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def run(root=ROOT,*,budget_rub=20,source_path=None):
    root=Path(root);budget=Budget(budget_rub);directory=root/'results'/identifier();directory.mkdir(parents=True)
    data={'status':'initializing','calls':[],'quizzes':[],'archive':[],'states':{},'context_sizes':{},'branches':{},
          'budget':budget.accounting(),'facts':FACTS,'prompts':PROMPTS,'code':complexity()}
    def save():data['budget']=budget.accounting();write_report(directory,data)
    save();key=(load_settings(root).get('POLZA_API_KEY') or '').strip()
    with Journal(root/'logs',key) as journal,httpx.Client(transport=httpx.HTTPTransport(retries=0)) as http:
        model=Client(key,journal,http,counter=TokenCounter(root/'data/tokenizer.json'),budget=budget)
        def record(start,phase,**labels):
            data['calls'].extend([{**deepcopy(c),'phase':phase,**labels} for c in model.records[start:]]);save()
        def progress(message):journal.record('PROGRESS',message);print(message,flush=True)
        try:
            path=Path(source_path or root.parent/'D4/submission/experiment.json')
            source,digest=load_source(path);archive=source['archive'];data['archive']=archive
            data['source']={'path':str(path),'sha256':digest,'original_cost_rub':source.get('budget',{}).get('known_cost_rub'),'new_preparation_calls':0}
            journal.record('SOURCE_IMPORT',data['source'])
            if not key or not model.counter.exact_text:raise RuntimeError('Нужны ключ и фиксированный токенизатор Qwen.')
            data['catalog']=preflight(model);data['tokenizer']={'url':TOKENIZER_URL,'sha256':TOKENIZER_SHA256,'method':model.counter.method}
            data['status']='running';save();database=directory/'data';session_ids={}
            starts=[i for i,m in enumerate(archive) if m['role']=='user'];ends=starts[1:]+[len(archive)]
            with TestClient(create_app(root=root,model=model,data_dir=database)) as web:
                store=web.app.state.store
                for mode in MODES:
                    sid=store.create('Маяк · '+mode,PROFILE);session_ids[mode]=sid;memory=Memory();facts=Facts()
                    for turn,(start,end) in enumerate(zip(starts,ends),1):
                        prefix=archive[:end];attempt=store.begin(sid,archive[start]['content']);first=len(model.records)
                        def observe(event,call,call_id):
                            if event=='start':return store.begin_call(attempt,call)
                            store.finish_call(call_id,call)
                        model.observer=observe
                        try:
                            if mode=='summary':memory=compact(prefix,memory,model)
                            if mode=='facts':facts=update_facts(prefix,facts,model)
                            store.complete(attempt,prefix,turn_view(archive[start:end]),memory=memory.to_dict(),facts_state=facts.to_dict())
                        except (AgentError,StoreError) as exc:store.fail(attempt,str(exc));raise
                        finally:model.observer=None;record(first,'build',mode=mode,turn=turn)
                        progress(f'{mode}: подготовка {turn}/12')
                    data['states'][mode]={'memory':memory.to_dict(),'facts_state':facts.to_dict()}
                    selected=select_context(archive,mode,memory,facts)
                    data['context_sizes'][mode]=model.counter.estimate([{'role':'system','content':SYSTEM}]+selected,TOOLS)
                    save()
            # Reopen the app and verify every strategy's archive and memory.
            with TestClient(create_app(root=root,model=model,data_dir=database)) as web:
                store=web.app.state.store
                for mode,sid in session_ids.items():
                    restored=store.load(sid)
                    if restored['messages']!=archive or restored['memory']!=data['states'][mode]['memory'] or restored['facts_state']!=data['states'][mode]['facts_state']:
                        raise RuntimeError('Не восстановлено состояние стратегии.')
                data['restart_verified']=True;data['session_ids']=session_ids
                for index,fact in enumerate(FACTS):
                    order=MODES[index%3:]+MODES[:index%3]
                    for mode in order:
                        first=len(model.records);state=data['states'][mode];memory=Memory(**deepcopy(state['memory']));facts=Facts(**deepcopy(state['facts_state']))
                        question=fact['question']+' Верни только JSON вида {"value":"ответ"}, без других полей и Markdown. Если не помнишь, value="неизвестно".'
                        row={'mode':mode,'id':fact['id'],'question':question,'answer':'','passed':False,'steps':[],'error':None}
                        try:
                            if mode=='facts':facts=update_facts(archive+[{'role':'user','content':question}],facts,model)
                            agent=Agent(model,context_builder=lambda messages:select_context(messages,mode,memory,facts));agent.messages=deepcopy(archive)
                            result=agent.run(question);row.update(answer=result['answer'],steps=result['steps']);row['passed']=grade(result['answer'],fact['answers'])
                            row['facts_after_question']=facts.to_dict() if mode=='facts' else None
                        except AgentError as exc:row['error']=str(exc)
                        finally:data['quizzes'].append(row);record(first,'quiz',mode=mode,fact_id=fact['id'])
                        journal.record('QUIZ',row);progress(f"{mode} / {fact['id']}: {row['passed']}")
                        if row['error']:raise RuntimeError(row['error'])
                parent=session_ids['facts'];before=store.load(parent);checkpoint=store.save_checkpoint(parent,'Маяк: общий checkpoint')
                journal.record('CHECKPOINT',{'checkpoint_id':checkpoint,'session_id':parent})
                data['branches']={'checkpoint_id':checkpoint,'parent_session_id':parent,'variants':{}}
                for label,budget_value,date in [('A','150000','2026-12-20'),('B','320000','2026-12-25')]:
                    sid=store.fork(checkpoint,'Ветка '+label);data['branches']['variants'][label]={'session_id':sid,'expected_budget':budget_value,'expected_date':date}
                    journal.record('FORK',{'checkpoint_id':checkpoint,'session_id':sid,'label':label})
                    for question in [f'Только для этой ветки: бюджет теперь {budget_value} рублей, дата демонстрации {date}. Подтверди кратко.',
                        'Каковы актуальные бюджет и дата демонстрации в этой ветке? Верни только JSON с полями budget_rub (число строкой) и deadline (YYYY-MM-DD).']:
                        first=len(model.records);response=web.post('/api/chat',json={'message':question,'session_id':sid,'mode':'facts'})
                        record(first,'branch',mode='facts',branch=label);response.raise_for_status()
                        result=response.json()
                    try:
                        answer=json.loads(result['answer']);passed=answer=={'budget_rub':budget_value,'deadline':date}
                    except (ValueError,TypeError):passed=False
                    data['branches']['variants'][label].update(answer=result['answer'],passed=passed)
                    progress(f'Ветка {label}: {passed}');save()
                after=store.load(parent)
                data['branches']['parent_unchanged']=all(after[k]==before[k] for k in ('messages','memory','facts_state','revision'))
                data['branch_snapshots']={label:store.load(value['session_id']) for label,value in data['branches']['variants'].items()}
            with TestClient(create_app(root=root,model=model,data_dir=database)) as web:
                data['branches']['restart_verified']=all(web.app.state.store.load(v['session_id'])['facts_state']==data['branch_snapshots'][label]['facts_state'] for label,v in data['branches']['variants'].items())
            checks=[data['branches']['parent_unchanged'],data['branches']['restart_verified']]+[v['passed'] for v in data['branches']['variants'].values()]
            data['status']='complete' if all(checks) else 'observed_branch_error'
        except (RuntimeError,AgentError,StoreError,OSError,ValueError,KeyError,httpx.HTTPError) as exc:
            data['status']='stopped';data['error']=journal.scrub(str(exc));journal.record('ERROR',data['error'])
        finally:
            save();trace=[]
            for line in journal.path.read_text(encoding='utf-8').splitlines():
                timestamp,event,payload=line.split(' ',2);trace.append({'timestamp':timestamp,'event':event,'data':json.loads(payload)})
            (directory/'trace.json').write_text(json.dumps(trace,ensure_ascii=False,indent=2),encoding='utf-8')
    return directory


def main():
    key=(load_settings(ROOT).get('POLZA_API_KEY') or '').strip()
    with Journal(ROOT/'logs',key) as journal:
        parser=LoggingParser(journal=journal,description='День 10: три стратегии на диалоге дня 9 и две независимые ветки')
        parser.add_argument('--budget-rub',type=float,default=20);parser.add_argument('--source',type=Path)
        args=parser.parse_args()
        try:
            directory=run(budget_rub=args.budget_rub,source_path=args.source);status=json.loads((directory/'experiment.json').read_text())['status']
            message=f'Отчёт: {directory / "report.md"}; статус: {status}'
            journal.record('STDOUT',message);print(message);return 0 if status=='complete' else 1
        except (ValueError,OSError) as exc:journal.record('ERROR',str(exc));print('Эксперимент остановлен. Проверьте журнал.');return 1

if __name__=='__main__':raise SystemExit(main())
