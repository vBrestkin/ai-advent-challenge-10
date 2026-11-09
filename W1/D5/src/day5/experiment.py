"""Три модели, общий бюджет, пилот и воспроизводимая локальная оценка."""
from copy import deepcopy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
from statistics import median

from jsonschema import Draft202012Validator

from .core import ApiError, RequestInterrupted, atomic_json, identifier, numeric
from .models import MODELS, PROVIDER, PARAMETERS, preflight
from .tasks import TASKS, verify_answer

SCHEMA={'type':'object','properties':{'answer':{'type':'string'}},
        'required':['answer'],'additionalProperties':False}
SYSTEM='Реши задачу. Верни только JSON по схеме API: answer — строка с ответом в формате условия, без объяснения.'
METRICS=('prompt_tokens','completion_tokens','total_tokens','reasoning_tokens','cost_rub','duration_seconds')


def messages_for(task):
    return [{'role':'system','content':SYSTEM},{'role':'user','content':task['question']}]


def parameters():
    return {**deepcopy(PARAMETERS),'response_format':{'type':'json_schema','json_schema':{
        'name':'solution','strict':True,'schema':deepcopy(SCHEMA)}}}


def assess(task,content):
    result={'format_valid':False,'answer_correct':None,'parsed':None}
    def reject(_):raise ValueError
    def unique(items):
        obj={}
        for k,v in items:
            if k in obj:raise ValueError
            obj[k]=v
        return obj
    try:
        obj=json.loads(content,parse_constant=reject,object_pairs_hook=unique)
        if not Draft202012Validator(SCHEMA).is_valid(obj):return result
    except (ValueError,TypeError,RecursionError):return result
    return {**result,'format_valid':True,'answer_correct':verify_answer(task,obj['answer']),'parsed':obj}


class BudgetStop(Exception):pass


class Budget:
    def __init__(self,limit):self.limit,self.spent,self.reserve=limit,0.,0.

    def before(self,api,messages,params):
        # Байты UTF-8 + служебный запас — консервативная оценка входа, не точный токенизатор.
        size=len(json.dumps({'model':api.model,'messages':messages,**params},ensure_ascii=False).encode())+2048
        amount=(size*api.price_limits['prompt']+params['max_tokens']*api.price_limits['completion'])/1e6
        if self.spent+self.reserve+amount>self.limit:raise BudgetStop
        return amount

    def after(self,row,amount):
        if numeric(row.get('cost_rub')) is None:
            self.reserve+=amount; row['budget_reserve_rub']=amount
        else:self.spent+=row['cost_rub']

    def restore(self,rows):
        self.spent,self.reserve=0.,0.
        for row in rows:
            if numeric(row.get('cost_rub')) is not None:self.spent+=row['cost_rub']
            elif numeric(row.get('budget_reserve_rub')) is not None:self.reserve+=row['budget_reserve_rub']
            else:raise ApiError('Нет сохранённого резерва для неизвестного списания.')

    def accounting(self):
        return {'limit_rub':self.limit,'known_cost_rub':round(self.spent,10),
                'unknown_reserve_rub':round(self.reserve,10),'accounted_rub':round(self.spent+self.reserve,10)}


def totals(rows):
    data={}
    for field in METRICS:
        known=[r[field] for r in rows if numeric(r.get(field)) is not None]
        data[field]=round(sum(known),10) if known else None
        data['unknown_costs' if field=='cost_rub' else 'unknown_'+field]=len(rows)-len(known)
    return data


def summarize(records):
    result=[]
    for tier,model in MODELS.items():
        rows=[r for r in records if r['tier']==tier]
        if not rows:continue
        received=[r for r in rows if r['received']]
        correct=sum(r['answer_correct'] is True for r in received)
        metrics=totals(rows)
        times=[r['duration_seconds'] for r in received if numeric(r.get('duration_seconds')) is not None]
        result.append(dict(tier=tier,model=model,attempts=len(rows),received=len(received),correct=correct,
            accuracy_attempts=correct/len(rows),accuracy_received=correct/len(received) if received else None,
            wrong_answers=sum(r['answer_correct'] is False for r in received),
            format_errors=sum(not r['format_valid'] for r in received),
            api_errors=sum(r['failure'] in ('api_error','interrupted','uncertain') for r in rows),
            incomplete=sum(r['failure']=='incomplete' for r in rows),
            model_mismatches=sum(r['failure']=='model_mismatch' for r in rows),
            duration_median_seconds=median(times) if times else None,
            cost_per_correct_rub=metrics['cost_rub']/correct if correct and not metrics['unknown_costs'] else None,
            **metrics))
    return result


def comparisons(summary,records=None):
    base=next((r for r in summary if r['tier']=='small'),None)
    if base is None:return []
    rows=[]
    for r in summary:
        if r is base:continue
        # Фактический расход сравниваем только на одинаковом числе попыток без неизвестных списаний.
        cost_ok=not r['unknown_costs'] and not base['unknown_costs'] and r['attempts']==base['attempts']
        lookup={(x['task_id'],x['repeat'],x['tier']):x for x in (records or [])}
        pairs=[]
        for small in (records or []):
            if small['tier']!='small':continue
            other=lookup.get((small['task_id'],small['repeat'],r['tier']))
            if other and small['received'] and other['received'] and all(numeric(x.get('cost_rub')) is not None for x in (small,other)):
                pairs.append((small['cost_rub'],other['cost_rub']))
        matched_base=sum(a for a,b in pairs)
        rows.append(dict(tier=r['tier'],cost_ratio=r['cost_rub']/base['cost_rub']
            if cost_ok and base['cost_rub'] else None,
            matched_cost_pairs=len(pairs),matched_cost_ratio=sum(b for a,b in pairs)/matched_base if matched_base else None,
            accuracy_delta_pp=(r['accuracy_attempts']-base['accuracy_attempts'])*100,
            equal_attempts=r['attempts']==base['attempts'],
            median_time_ratio=r['duration_median_seconds']/base['duration_median_seconds']
                if r['duration_median_seconds'] is not None and base['duration_median_seconds'] else None))
    return rows


def task_summary(records):
    result=[]
    for task in TASKS:
        for tier in MODELS:
            rows=[r for r in records if r['tier']==tier and r['task_id']==task['id']]
            if not rows:continue
            result.append(dict(task_id=task['id'],title=task['title'],tier=tier,attempts=len(rows),
                received=sum(r['received'] for r in rows),correct=sum(r['answer_correct'] is True for r in rows),
                failures=sum(r['failure'] is not None for r in rows)))
    return result


def csv_text(rows):
    if not rows:return ''
    out=io.StringIO(newline=''); writer=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return out.getvalue()


def show(value):
    return '—' if value is None else f'{value:.6f}'.rstrip('0').rstrip('.') if isinstance(value,float) else str(value)


def write_reports(directory,data):
    atomic_json(directory/'results.json',data)
    summary=summarize(data['records']); comp=comparisons(summary,data['records']); tasks=task_summary(data['records'])
    (directory/'summary.csv').write_text(csv_text(summary),encoding='utf-8-sig')
    (directory/'comparisons.csv').write_text(csv_text(comp),encoding='utf-8-sig')
    (directory/'tasks.csv').write_text(csv_text(tasks),encoding='utf-8-sig')
    fields=('tier','model','task_id','repeat','received','format_valid','answer_correct','failure','api_error',
            'finish_reason','content','budget_reserve_rub',*METRICS)
    (directory/'details.csv').write_text(csv_text([{k:r.get(k) for k in fields} for r in data['records']]),encoding='utf-8-sig')
    budget=data['budget']
    lines=['# День 5 — версии моделей','',f"Статус: **{data['config']['status']}**. Маршрут: `{PROVIDER}`.",'',
        '## Пилот','',f"Проверка возможностей: {data['pilot_verified']}. Пилотные вызовы не входят в основную таблицу, но входят в бюджет.",'',
        '| Модель | Завершён | Схема | Reasoning-токены | Ошибки проверки |','|---|---:|---:|---:|---|']
    for r in data['pilot']:
        lines.append('| '+' | '.join([r['model'],str(r['complete']),str(r['format_valid']),show(r.get('reasoning_tokens')),', '.join(r.get('pilot_errors',[])) or 'нет'])+' |')
    lines+=['','## Итоговая таблица','',
        '| Модель | Верных / попыток | Получено | Неверных | Формат | API | Незавершённых | Медиана ответа, с | Все токены, известные | Стоимость, ₽ (известная) | Неизв. списаний |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append('| '+' | '.join([r['model'],f"{r['correct']}/{r['attempts']}"]+[show(r[k]) for k in
            ('received','wrong_answers','format_errors','api_errors','incomplete','duration_median_seconds','total_tokens','cost_rub','unknown_costs')])+' |')
    lines+=['','Медиана времени рассчитана по завершённым ответам выбранной модели, включая неверный ответ и ошибку формата. '
        'В details.csv сохранено время всех HTTP-попыток. Сетевая ошибка не считается неверным математическим ответом. '
        'Основная доля успехов — верные / все попытки; accuracy_received отдельно показывает верные / полученные.', '',
        '## Цена и точность относительно маленькой модели','']
    for r in comp:
        lines.append(f"- {MODELS[r['tier']]}: отношение фактической стоимости = {show(r['cost_ratio'])}; "
            f"разница точности = {show(r['accuracy_delta_pp'])} процентных пунктов; отношение медиан времени = {show(r['median_time_ratio'])}. "
            f"Одинаковое число попыток: {r['equal_attempts']}.")
        lines.append(f"  На {r['matched_cost_pairs']} одинаковых парах задача/повтор с завершёнными ответами и известными ценами "
            f"отношение суммарных расходов = {show(r['matched_cost_ratio'])}. Это стоимость сопоставимой части, а не всего прогона.")
    lines+=['','Если хотя бы одно списание неизвестно или число попыток разное, отношение фактических затрат не вычисляется. '
        'Резерв бюджета не подставляется вместо реальной цены. Разница точности включает доступность API; '
        'для качества рассуждений см. accuracy_received и попарные результаты ниже.', '',
        '## Результаты по задачам','', '| Задача | Маленькая | Средняя | Флагманская |','|---|---:|---:|---:|']
    for task in TASKS:
        cells=[]
        for tier in MODELS:
            row=next((r for r in tasks if r['task_id']==task['id'] and r['tier']==tier),None)
            cells.append(f"{row['correct']}/{row['attempts']} (ответов {row['received']})" if row else '—')
        lines.append('| '+task['id']+' '+task['title']+' | '+' | '.join(cells)+' |')
    lines+=['','## Выводы','']
    small=[r for r in tasks if r['tier']=='small']
    enough=[r['task_id']+' '+r['title'] for r in small if r['attempts']==data['config']['repeats'] and r['correct']==r['attempts']]
    lines.append('Маленькая модель решила все повторы следующих задач: '+('; '.join(enough) if enough else 'пока ни одной')+'.')
    for tier in ('medium','flagship'):
        pairs=[]
        lookup={(r['task_id'],r['repeat'],r['tier']):r for r in data['records']}
        for r in data['records']:
            if r['tier']!='small':continue
            other=lookup.get((r['task_id'],r['repeat'],tier))
            if other and r['format_valid'] and other['format_valid'] and r['received'] and other['received']:
                pairs.append((r['answer_correct'],other['answer_correct']))
        lines.append(f"- {tier}: среди {len(pairs)} пар с пригодными ответами обеих моделей улучшений "
            f"{sum(not a and b for a,b in pairs)}, ухудшений {sum(a and not b for a,b in pairs)} относительно маленькой.")
    lines+=['','Восемь задач и три повтора дают учебное наблюдение. Повторы одних задач зависимы, '
        'поэтому 24 попытки не равны 24 независимым задачам. Выводы ограничены этим набором, '
        'запрошенным reasoning-бюджетом и выбранным маршрутом. Результат не доказывает превосходство на всех задачах.', '',
        '## Тарифы и условия','', '| Модель | Вход, ₽/1M | Выход, ₽/1M |','|---|---:|---:|']
    for tier,check in data['catalog'].items():
        lines.append('| '+MODELS[tier]+' | '+show(check['rates']['prompt_per_million'])+' | '+show(check['rates']['completion_per_million'])+' |')
    small_rates=data['catalog']['small']['rates']; flagship_rates=data['catalog']['flagship']['rates']
    lines+=['',f"Тариф флагманской модели выше маленькой в {show(flagship_rates['prompt_per_million']/small_rates['prompt_per_million'])} "
        f"раза на входе и в {show(flagship_rates['completion_per_million']/small_rates['completion_per_million'])} раза на выходе. "
        'Отношение реальной стоимости зависит также от количества использованных токенов.']
    lines+=['','- Идентичные system/user сообщения, строгая JSON Schema с одним строковым answer, температура 0.7.',
        '- Всем запрошены reasoning.enabled=true и reasoning.max_tokens=1024, общий max_tokens=4096. '
        '1024 — запрошенный бюджет, не обещание одинакового фактического расхода или жёсткого внутреннего предела.',
        '- У Gemini 2.5 Pro рассуждение обязательно; у остальных включено явно. Проверяется положительный reasoning-счётчик в пилоте.',
        '- Каждая попытка начинает чистый контекст; в интерактивном chat история сохраняется. Промпт не содержит эталона.',
        '- Порядок 24 комбинаций перемешан внутри каждого повтора с локальным seed=42; seed модели не передаётся.',
        '- Маршрут закреплён, fallback и повторы отключены. Поле provider в ответе сохраняется как сообщённое шлюзом.',
        '- Тарифы и capabilities сохранены в catalog.json; ограничение max_price = снимок тарифа +1%.',
        '- Reasoning уже входит в completion/total и не прибавляется повторно. Неизвестные метрики не превращаются в нули.',
        '- Стоимость берётся из usage.cost_rub. При разрыве связи списание неизвестно; резерв не равен расходу.',
        '- Резерв перед вызовом: верхний лимит выхода и оценка входа по UTF-8 байтам с запасом 2048. '
        'Это локальная защита бюджета, а не банковская гарантия.', '',
        '## Общий бюджет, включая пилот','',
        f"Известно: {show(budget['known_cost_rub'])} ₽; резерв неизвестных списаний: {show(budget['unknown_reserve_rub'])} ₽; "
        f"учтено: {show(budget['accounted_rub'])} / {show(budget['limit_rub'])} ₽.", '',
        '## Источники','',
        '- [Каталог Polza.ai](https://polza.ai/docs/api-reference/models/list)',
        '- [Reasoning Polza.ai](https://polza.ai/docs/osobennosti/reasoning-tokens)',
        '- [Бюджеты Gemini 2.5](https://ai.google.dev/gemini-api/docs/generate-content/thinking?hl=en)', '',
        'Полные данные: results.json, details.csv, tasks.csv, comparisons.csv и raw/*.json. '
        'Эталоны и правила проверки — TASKS.md и src/day5/tasks.py.']
    (directory/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def configuration(repeats,budget_rub):
    return dict(version=1,models=deepcopy(MODELS),provider=PROVIDER,parameters=parameters(),
        system=SYSTEM,repeats=repeats,budget_rub=budget_rub,order_seed=42,
        task_fingerprint=hashlib.sha256(json.dumps(TASKS,sort_keys=True,ensure_ascii=False).encode()).hexdigest())


def pilot_errors(row):
    errors=[]
    if not row['received'] or not row['format_valid']:errors.append('Не получен завершённый ответ по схеме от выбранной модели.')
    if any(numeric(row.get(k)) is None for k in METRICS):errors.append('Не все метрики доступны.')
    if numeric(row.get('reasoning_tokens')) is None or row['reasoning_tokens']<=0:errors.append('Не подтверждён ненулевой reasoning.')
    if all(numeric(row.get(k)) is not None for k in ('prompt_tokens','completion_tokens','total_tokens','reasoning_tokens')):
        if row['total_tokens']!=row['prompt_tokens']+row['completion_tokens'] or row['reasoning_tokens']>row['completion_tokens']:
            errors.append('Несогласованные счётчики токенов.')
    return errors


def run_experiment(clients,output_dir,*,repeats=3,budget_rub=50,notify=None,resume_dir=None):
    if type(repeats) is not int or not 1<=repeats<=100 or numeric(budget_rub) is None or budget_rub<=0:
        raise ApiError('Некорректное число повторов или бюджет.')
    if set(clients)!=set(MODELS) or any(api.model!=MODELS[tier] for tier,api in clients.items()):
        raise ApiError('Нужны ровно три согласованных модели.')
    cfg=configuration(repeats,budget_rub)
    data=None
    if resume_dir is not None:
        directory=Path(resume_dir)
        try:
            data=json.loads((directory/'results.json').read_text(encoding='utf-8'))
            if any(data['config'][k]!=v for k,v in cfg.items()):raise ValueError
            if not all(isinstance(data[k],list) for k in ('records','pilot')):raise ValueError
            seen=set()
            for group in ('pilot','records'):
                for r in data[group]:
                    key=(group,r['tier'],r['task_id'],r['repeat'])
                    if key in seen or r['tier'] not in MODELS:raise ValueError
                    if r['model']!=MODELS[r['tier']] or r['task_id'] not in {t['id'] for t in TASKS}:raise ValueError
                    if type(r['repeat']) is not int:raise ValueError
                    if group=='pilot' and (r['repeat']!=0 or r['task_id']!='T8'):raise ValueError
                    if group=='records' and not 1<=r['repeat']<=repeats:raise ValueError
                    seen.add(key)
            if len(data['pilot'])>3 or len(data['records'])>len(TASKS)*3*repeats:raise ValueError
        except (OSError,ValueError,TypeError,KeyError):raise ApiError('Прогон повреждён или его настройки отличаются; файл не изменён.') from None
    # Проверяем все три маршрута до первого платного запроса.
    checked={tier:preflight(api) for tier,api in clients.items()}
    if data is None:
        directory=Path(output_dir)/identifier(); directory.mkdir(parents=True,exist_ok=False)
        data={'config':{**cfg,'status':'ready'},'pilot':[],'records':[],'pilot_verified':False,'catalog':checked}
    atomic_json(directory/('catalog-resume-'+identifier()+'.json' if resume_dir else 'catalog.json'),checked)
    budget=Budget(budget_rub); budget.restore(data['pilot']+data['records'])
    journal=clients['small'].journal

    def save():
        data['budget']=budget.accounting(); write_reports(directory,data)

    def execute(group,tier,task,repeat):
        api=clients[tier]; messages=messages_for(task); params=parameters()
        reservation=budget.before(api,messages,params)
        # До отправки оставляем durable-маркер: аварийный выход не приведёт к повторному списанию при resume.
        pending=dict(tier=tier,model=api.model,task_id=task['id'],repeat=repeat,
            request={'model':api.model,'messages':messages,**params},content='',complete=False,received=False,
            actual_model=None,http_status=None,api_error='Неизвестный исход активного запроса.',failure='uncertain',
            format_valid=False,answer_correct=None,parsed=None,**{k:None for k in METRICS})
        budget.after(pending,reservation); data[group].append(pending)
        atomic_json(directory/'raw'/f'{group}_{tier}_{task["id"]}_{repeat:02d}.json',pending)
        save()
        interrupted=False
        try:row=api.send(messages,params)
        except RequestInterrupted as exc:row=exc.result; interrupted=True
        except ApiError as exc:
            if not exc.result:raise
            row=exc.result
        row.update(tier=tier,model=api.model,task_id=task['id'],repeat=repeat)
        row['received']=bool(row['complete'] and not row['api_error'] and row['actual_model']==api.model)
        row.update(assess(task,row['content']) if row['received'] else {'format_valid':False,'answer_correct':None,'parsed':None})
        row['failure']=('interrupted' if interrupted else 'api_error' if row['api_error'] else
            'model_mismatch' if row['actual_model']!=api.model else 'incomplete' if not row['complete'] else
            'format_error' if not row['format_valid'] else 'wrong_answer' if row['answer_correct'] is False else None)
        if group=='pilot':row['pilot_errors']=pilot_errors(row)
        if numeric(row.get('cost_rub')) is None:row['budget_reserve_rub']=reservation
        data[group][-1]=journal.scrub(row); budget.restore(data['pilot']+data['records'])
        atomic_json(directory/'raw'/f'{group}_{tier}_{task["id"]}_{repeat:02d}.json',data[group][-1]); save()
        journal.record('EXPERIMENT_RESULT',row)
        if notify:notify(f"{group}: {tier}, {task['id']}, повтор {repeat}; {row['failure'] or 'верно'}; "
                         f"расход {show(budget.accounting()['accounted_rub'])}/{show(budget_rub)} ₽")
        if interrupted:raise KeyboardInterrupt
        return row

    save()
    try:
        data['config']['status']='pilot'
        for tier in MODELS:
            if any(r['tier']==tier for r in data['pilot']):continue
            row=execute('pilot',tier,TASKS[-1],0)
            if row['http_status'] in (400,401,402,403,404,413,422):break
        data['pilot_verified']=len(data['pilot'])==3 and all(not r.get('pilot_errors',['unknown']) for r in data['pilot'])
        if not data['pilot_verified']:
            data['config']['status']='pilot_failed'; return directory
        if any(r.get('http_status') in (400,401,402,403,404,413,422) for r in data['records']):
            data['config']['status']='request_rejected'; return directory
        data['config']['status']='running'; save()
        done={(r['tier'],r['task_id'],r['repeat']) for r in data['records']}
        rng=random.Random(42)
        for repeat in range(1,repeats+1):
            order=[(tier,task) for tier in MODELS for task in TASKS]; rng.shuffle(order)
            for tier,task in order:
                if (tier,task['id'],repeat) in done:continue
                row=execute('records',tier,task,repeat)
                if row['http_status'] in (400,401,402,403,404,413,422):
                    data['config']['status']='request_rejected'; return directory
        data['config']['status']='complete'
    except BudgetStop:data['config']['status']='budget_stopped'
    except BaseException:
        data['config']['status']='interrupted'; raise
    finally:
        save(); journal.record('EXPERIMENT_END',{'directory':str(directory),'status':data['config']['status'],'budget':data['budget']})
    return directory
