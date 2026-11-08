"""Проверка возможностей, пилот и эксперимент без смешивания контекста задач."""
from copy import deepcopy
import csv
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import random

from jsonschema import Draft202012Validator

from .core import ApiError, atomic_json, identifier
from .history import MODES as HISTORY_MODES
from .tasks import TASKS, verify_answer

METHODS = ['direct', 'stepwise', 'metaprompt', 'experts']
MODES = {key: deepcopy(HISTORY_MODES[key]) for key in ('min', 'max')}
LABELS = {'direct':'Прямой ответ','stepwise':'Пошаговое объяснение',
          'metaprompt':'Составление промпта','experts':'Группа экспертов'}
MAX_TOKENS = 32768
TEMPERATURE = .7
FINAL_SCHEMA = {'type':'object','properties':{'answer':{'type':'string'},
    'explanation':{'type':'string'},'experts':{'type':'array','items':{'type':'object',
        'properties':{'role':{'type':'string','enum':['analyst','engineer','critic']},
            'answer':{'type':'string'},'explanation':{'type':'string'}},
        'required':['role','answer','explanation'],'additionalProperties':False}}},
    'required':['answer','explanation','experts'],'additionalProperties':False}
PLAN_SCHEMA = {'type':'object','properties':{'prompt':{'type':'string'}},
               'required':['prompt'],'additionalProperties':False}
SYSTEM = ('Выполняй текущий запрос пользователя. Отвечай JSON по схеме API без Markdown. '
          'В решении поле answer содержит только ответ в формате условия; explanation — краткое '
          'объяснение до 150 слов, если оно запрошено, иначе пустую строку. '
          'experts — пустой массив, если роли экспертов не запрошены. Не используй внешние инструменты.')


def config(api):
    return dict(version=2,model=api.model,provider=api.provider,modes=deepcopy(MODES),
                max_tokens=MAX_TOKENS,temperature=TEMPERATURE,schema=FINAL_SCHEMA,
                task_fingerprint=hashlib.sha256(json.dumps(TASKS,sort_keys=True,ensure_ascii=False).encode()).hexdigest())


def preflight(api):
    if api.model != 'google/gemini-2.5-flash-lite':
        raise ApiError('Профиль бюджета 24576 описан только для google/gemini-2.5-flash-lite; для другой модели нужна отдельная проверка.')
    catalog=api.catalog()
    providers=catalog.get('providers',[])
    provider=next((p for p in providers if p.get('name')==api.provider),None)
    if provider is None:
        raise ApiError('У выбранной модели нет запрошенного маршрута провайдера.')
    required={'reasoning','response_format','structured_outputs','max_tokens','temperature'}
    if not required <= set(provider.get('supported_parameters',[])):
        raise ApiError('Маршрут не заявляет все необходимые параметры reasoning и формата.')
    caps=provider.get('capabilities') or {}
    if any(caps.get(name,{}).get('status')!='supported' for name in ('reasoning_toggle','json_schema_strict')):
        raise ApiError('Каталог не подтверждает переключение reasoning или строгую схему для маршрута.')
    if provider.get('max_completion_tokens',catalog.get('max_completion_tokens',0)) < MAX_TOKENS:
        raise ApiError('Лимит модели меньше общего бюджета 32768 токенов.')
    pricing=provider.get('pricing') or {}
    try:
        rates={k:float(pricing[k]) for k in ('prompt_per_million','completion_per_million')}
        if pricing.get('currency')!='RUB' or any(not math.isfinite(v) or v<=0 for v in rates.values()):
            raise ValueError
    except (KeyError,TypeError,ValueError):
        raise ApiError('Неизвестная рублёвая цена выбранного маршрута.') from None
    # Допускаем небольшое округление. Более дорогой маршрут отклоняет сам шлюз.
    api.price_limits={'prompt':rates['prompt_per_million']*1.01,
                      'completion':rates['completion_per_million']*1.01}
    return {'catalog':catalog,'provider':provider,'rates':rates,'price_limits':deepcopy(api.price_limits)}


def response_format(schema, name):
    return {'type':'json_schema','json_schema':{'name':name,'strict':True,'schema':deepcopy(schema)}}


def parse(content, schema):
    def constant(_):
        raise ValueError
    def pairs(items):
        result={}
        for key,value in items:
            if key in result: raise ValueError
            result[key]=value
        return result
    try:
        data=json.loads(content,parse_constant=constant,object_pairs_hook=pairs)
        if not Draft202012Validator(schema).is_valid(data): return None
        return data
    except (ValueError,TypeError,RecursionError):
        return None


METRIC_KEYS=('prompt_tokens','completion_tokens','total_tokens','reasoning_tokens','cost_rub','duration_seconds')


def aggregate(stages):
    metrics={'calls':len(stages)}
    for key in METRIC_KEYS:
        known=[stage.get(key) for stage in stages if stage.get(key) is not None]
        metrics[key]=round(sum(known),10) if known else None
        metrics['unknown_'+('costs' if key=='cost_rub' else key)]=len(stages)-len(known)
    return metrics


class BudgetStop(Exception):
    row = None


def solve_one(api, task, method, mode, *, checkpoint=None, before=None, after=None):
    row=dict(task_id=task['id'],task_title=task['title'],method=method,mode=mode,
             stages=[],parsed=None,format_valid=False,method_valid=False,
             answer_correct=False,correct=False,failure=None,metrics=aggregate([]))

    def save():
        row['metrics']=aggregate(row['stages'])
        if checkpoint: checkpoint(deepcopy(row))

    def call(messages, schema, name):
        parameters={'max_tokens':MAX_TOKENS,'temperature':TEMPERATURE,'reasoning':deepcopy(MODES[mode]),
                    'response_format':response_format(schema,name)}
        if before: before(messages,parameters)
        try:
            stage=api.send(messages,parameters)
        except ApiError as exc:
            stage=exc.result or {'api_error':str(exc),'content':'','complete':False}
        row['stages'].append(stage)
        if after: after(stage)
        save()
        if stage.get('api_error'):
            row['failure']='api_error'; return None
        if not stage.get('complete'):
            row['failure']='incomplete'; return None
        if stage.get('actual_model')!=api.model:
            row['failure']='model_mismatch'; return None
        data=parse(stage['content'],schema)
        if data is None:
            row['failure']='format_error'
        return data

    messages=[{'role':'system','content':SYSTEM}]
    question=task['question']
    try:
        if method=='metaprompt':
            messages.append({'role':'user','content':'Составь промпт для решения следующей задачи. '
                             'Пока не решай задачу. Верни JSON с полем prompt. Задача:\n'+question})
            plan=call(messages,PLAN_SCHEMA,'solution_prompt')
            if plan is None or not plan['prompt'].strip():
                row['failure']=row['failure'] or 'format_error'
                return row
            messages.append({'role':'assistant','content':row['stages'][-1]['content']})
            messages.append({'role':'user','content':'Теперь реши исходную задачу, используя составленный промпт. '
                'Промпт:\n'+plan['prompt']+'\nИсходная задача:\n'+question+
                '\nВерни ответ по схеме solution; explanation и experts оставь пустыми.'})
        else:
            instructions={
                'direct':'Дай прямой ответ. explanation — пустая строка, experts — пустой массив.',
                'stepwise':'ПОШАГОВО: реши задачу с кратким пошаговым объяснением в explanation; experts — пустой массив.',
                'experts':'ЭКСПЕРТЫ: аналитик (analyst), инженер (engineer) и критик (critic). '
                    'В experts дай отдельный ответ и краткое объяснение каждого. '
                    'В answer дай общий итог, в explanation кратко обоснуй его.'}
            messages.append({'role':'user','content':question+'\n'+instructions[method]})
        parsed=call(messages,FINAL_SCHEMA,'solution')
        row['parsed']=parsed
        if parsed is None: return row
        row['format_valid']=True
        if method=='experts':
            roles=[expert['role'] for expert in parsed['experts']]
            row['method_valid']=len(roles)==3 and set(roles)=={'analyst','engineer','critic'} and all(
                e['answer'].strip() and e['explanation'].strip() for e in parsed['experts'])
        elif method=='stepwise':
            row['method_valid']=bool(parsed['explanation'].strip()) and not parsed['experts']
        else:
            row['method_valid']=not parsed['experts'] and not parsed['explanation'].strip()
        row['answer_correct']=verify_answer(task,parsed['answer'])
        row['correct']=bool(row['answer_correct'] and row['method_valid'])
        row['failure']=None if row['correct'] else ('wrong_answer' if not row['answer_correct'] else 'method_error')
        return row
    except BudgetStop as exc:
        row['failure']='budget_stopped'
        exc.row=row
        raise
    except KeyboardInterrupt as exc:
        row['failure']='interrupted'
        exc.row=row
        raise
    finally:
        save()


def pilot_checks(row, api):
    errors=[]
    if row.get('mode') not in MODES:
        return ['Пилот не относится к согласованным режимам min/max.']
    if not row['format_valid'] or not row['method_valid']:
        errors.append('Ответ не завершён или не соответствует схеме/методу.')
    if len(row['stages'])!=1:
        errors.append('Ожидался один вызов для режима.')
    else:
        stage=row['stages'][0]
        if stage.get('actual_model')!=api.model: errors.append('Модель ответа отличается.')
        request=stage.get('request',{})
        if (request.get('reasoning')!=MODES[row['mode']] or request.get('max_tokens')!=MAX_TOKENS
                or request.get('provider',{}).get('only')!=[api.provider]):
            errors.append('Настройки запроса отличаются от согласованных.')
        metrics=row['metrics']
        if any(metrics.get(k) is None for k in METRIC_KEYS): errors.append('Отсутствуют обязательные метрики.')
        reason=metrics.get('reasoning_tokens')
        if reason is None or reason<=0:
            errors.append('Не подтверждено использование reasoning-токенов.')
        completion=metrics.get('completion_tokens')
        total=metrics.get('total_tokens')
        prompt=metrics.get('prompt_tokens')
        if reason is not None and completion is not None and reason>completion:
            errors.append('Reasoning-токены не укладываются в completion_tokens; семантику usage нужно уточнить.')
        if None not in (total,prompt,completion) and total!=prompt+completion:
            errors.append('Не подтверждён обычный учёт total=prompt+completion.')
    return errors


def show(number):
    return '—' if number is None else f'{number:.8f}'.rstrip('0').rstrip('.') if isinstance(number,float) else str(number)


def run_pilot(api, output_dir, notify=None):
    checked=preflight(api)
    directory=Path(output_dir)/identifier()
    directory.mkdir(parents=True,exist_ok=False)
    atomic_json(directory/'catalog.json',checked)
    data={'config':config(api),'started_at':datetime.now().astimezone().isoformat(timespec='seconds'),
          'records':[],'verified':False,'metrics':aggregate([]),'checks':{}}
    path=directory/'pilot.json'
    atomic_json(path,data)
    try:
        for mode in MODES:
            def checkpoint(row):
                atomic_json(directory/(mode+'.json'),row)
            row=solve_one(api,TASKS[-1],'direct',mode,checkpoint=checkpoint)
            data['records'].append(row)
            data['checks'][mode]=pilot_checks(row,api)
            atomic_json(path,data)
            if notify: notify(f"Пилот {mode}: reasoning_tokens={row['metrics']['reasoning_tokens']}; цена={show(row['metrics']['cost_rub'])} ₽")
        if len(data['records'])==2:
            counts={r['mode']:r['metrics']['reasoning_tokens'] for r in data['records']}
            if (counts['min'] is not None and counts['max'] is not None and counts['max']<=counts['min']):
                data['checks']['max'].append('На проверочной задаче не наблюдается большего reasoning-расхода max по сравнению с min.')
        data['verified']=len(data['records'])==2 and not any(data['checks'].values())
    finally:
        stages=[s for row in data['records'] for s in row['stages']]
        data['metrics']=aggregate(stages)
        atomic_json(path,data)
        lines=['# Пилот D3','',f"Модель: `{api.model}`. Маршрут: `{api.provider}`.",'',
               f"Режимы подтверждены: {'да' if data['verified'] else 'нет'}.",'',
               '| Режим | Reasoning-токены | Все токены | Секунды | ₽ | Ответ T8 | Проверки |',
               '|---|---:|---:|---:|---:|---|---|']
        for row in data['records']:
            m=row['metrics']; answer=row['parsed']['answer'] if row['parsed'] else '—'
            lines.append('| '+' | '.join([row['mode']]+[show(m[k]) for k in ('reasoning_tokens','total_tokens','duration_seconds','cost_rub')]+[
                answer, '; '.join(data['checks'].get(row['mode'],[])) or 'пройдены'])+' |')
        lines += ['',f"Стоимость двух запросов: {show(data['metrics']['cost_rub'])} ₽.",'',
            'Использована одна задача T8 (эталон 14); это проверка API, а не итог сравнения методов. '
            'Неверный ответ сам по себе не означает отсутствие поддержки reasoning.', '',
            'Положительный счётчик подтверждает использование thinking, но не доказывает, что провайдер '
            'реализовал максимальный бюджет точно или израсходовал его целиком. Закреплён запрошенный маршрут; '
            'независимой аттестации внутренней реализации backend нет.', '',
            'Основной эксперимент не запускается автоматически. Для него нужны путь к пилоту и бюджет.', '']
        (directory/'report.md').write_text('\n'.join(lines),encoding='utf-8')
        api.journal.record('PILOT_END',{'path':str(path),'verified':data['verified'],'metrics':data['metrics']})
    return path


class Budget:
    def __init__(self, api, limit):
        self.api,self.limit,self.spent=api,limit,0.
        self.unknown_reserve=0.
        self.pending_reserve=0.

    def reservation(self, request):
        input_bound=len(json.dumps(request,ensure_ascii=False).encode())+2048
        old_rates=request.get('provider',{}).get('max_price',{})
        rates={key:max(self.api.price_limits[key],old_rates.get(key,0)) for key in ('prompt','completion')}
        return (input_bound*rates['prompt']+request.get('max_tokens',MAX_TOKENS)*rates['completion'])/1_000_000

    def before(self,messages,parameters):
        # Запас по входу: байты текста/схемы плюс 2048 на служебное обрамление.
        # Это консервативная оценка, не официальный счётчик токенизатора.
        request={'model':self.api.model,'messages':messages,**parameters}
        reserved=self.reservation(request)
        if self.spent+self.unknown_reserve+reserved>self.limit:
            raise BudgetStop('Не хватает бюджета с учётом резервов на запросы с неизвестным расходом.')
        self.pending_reserve=reserved

    def after(self,stage):
        if stage.get('cost_rub') is None:
            self.unknown_reserve+=self.pending_reserve
            stage['budget_reserve_rub']=self.pending_reserve
        else: self.spent+=stage['cost_rub']

    def restore(self,records):
        for row in records:
            for stage in row['stages']:
                if stage.get('cost_rub') is None:
                    if not stage.get('request'):raise ApiError('Нельзя восстановить резерв: отсутствует исходный запрос.')
                    self.unknown_reserve+=stage.get('budget_reserve_rub') or self.reservation(stage['request'])
                else:self.spent+=stage['cost_rub']

    def accounting(self):
        return {'known_cost_rub':round(self.spent,10),'unknown_reserve_rub':round(self.unknown_reserve,10),
                'accounted_rub':round(self.spent+self.unknown_reserve,10),'limit_rub':self.limit}


def summarize(records):
    result=[]
    for method in METHODS:
        for mode in MODES:
            rows=[r for r in records if r['method']==method and r['mode']==mode]
            if not rows: continue
            result.append(dict(method=method,mode=mode,tasks=len(rows),correct=sum(r['correct'] for r in rows),
                answer_correct=sum(r['answer_correct'] for r in rows),
                answers_received=sum(bool(r['format_valid']) for r in rows),
                wrong_answers=sum(bool(r['format_valid'] and not r['answer_correct']) for r in rows),
                api_errors=sum(r['failure']=='api_error' for r in rows),
                format_errors=sum(r['failure']=='format_error' for r in rows),
                incomplete=sum(r['failure']=='incomplete' for r in rows),
                method_errors=sum(r['failure']=='method_error' for r in rows),
                metrics=aggregate([stage for row in rows for stage in row['stages']])))
    return result


def comparable(row):
    return row['format_valid'] and row['failure'] not in {
        'api_error','incomplete','model_mismatch','budget_stopped','interrupted','format_error'}


def comparisons(records):
    result=[]
    for mode in MODES:
        direct={r['task_id']:r for r in records if r['mode']==mode and r['method']=='direct'}
        for method in METHODS[1:]:
            other={r['task_id']:r for r in records if r['mode']==mode and r['method']==method}
            all_common=sorted(set(direct)&set(other))
            if not all_common:continue
            common=[t for t in all_common if comparable(direct[t]) and comparable(other[t])]
            base=sum(bool(direct[t]['answer_correct']) for t in common)
            score=sum(bool(other[t]['answer_correct']) for t in common)
            gains=sum(not direct[t]['answer_correct'] and other[t]['answer_correct'] for t in common)
            losses=sum(direct[t]['answer_correct'] and not other[t]['answer_correct'] for t in common)
            base_metrics=aggregate([s for t in common for s in direct[t]['stages']])
            other_metrics=aggregate([s for t in common for s in other[t]['stages']])
            ratios={}
            for key in ('total_tokens','duration_seconds','cost_rub'):
                unknown='unknown_'+('costs' if key=='cost_rub' else key)
                ratios[key+'_ratio']=(other_metrics[key]/base_metrics[key] if base_metrics[key]
                    and other_metrics[key] is not None and not base_metrics[unknown] and not other_metrics[unknown] else None)
            result.append(dict(method=method,mode=mode,common_tasks=len(common),direct_correct=base,
                excluded_pairs=len(all_common)-len(common),method_correct=score,net_gain=score-base,gains=gains,losses=losses,**ratios))
    return result


def interactions(records):
    index={(r['task_id'],r['method'],r['mode']):r for r in records}
    result=[]
    for method in METHODS[1:]:
        complete=[t['id'] for t in TASKS if all((t['id'],m,mode) in index
                  for m in ('direct',method) for mode in MODES)]
        if not complete:continue
        common=[t for t in complete if all(comparable(index[t,m,mode]) for m in ('direct',method) for mode in MODES)]
        row=dict(method=method,common_tasks=len(common),excluded_tasks=len(complete)-len(common))
        for mode in MODES:
            row['direct_'+mode]=sum(index[t,'direct',mode]['answer_correct'] for t in common)
            row['method_'+mode]=sum(index[t,method,mode]['answer_correct'] for t in common)
            row['gain_'+mode]=row['method_'+mode]-row['direct_'+mode]
        row['gain_change']=row['gain_max']-row['gain_min']
        result.append(row)
    return result


def csv_content(rows):
    if not rows: return ''
    file=io.StringIO(newline='')
    writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return file.getvalue()


def write_reports(directory,data):
    atomic_json(directory/'results.json',data)
    summary=summarize(data['records'])
    flattened=[{**{k:v for k,v in r.items() if k!='metrics'},**r['metrics']} for r in summary]
    (directory/'summary.csv').write_text(csv_content(flattened),encoding='utf-8-sig')
    details=[{**{k:r[k] for k in ('task_id','method','mode','correct','answer_correct','format_valid','failure')},
              'answer':r['parsed']['answer'] if r['parsed'] else '',**r['metrics']} for r in data['records']]
    (directory/'details.csv').write_text(csv_content(details),encoding='utf-8-sig')
    paired=comparisons(data['records'])
    (directory/'comparisons.csv').write_text(csv_content(paired),encoding='utf-8-sig')
    interaction=interactions(data['records'])
    (directory/'interactions.csv').write_text(csv_content(interaction),encoding='utf-8-sig')
    lines=['# День 3 — способы рассуждения','',f"Статус: {data['config']['status']}. Модель: `{data['config']['model']}`.",'',
        '| Способ | Reasoning | Верные ответы | Полный успех | Вызовы | Токены | Reasoning-токены | Время, с | Известная стоимость, ₽ | Ошибки API / формата / обрезания | Нарушения метода |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|']
    for r in summary:
        m=r['metrics']
        values=[('≥' if m.get('unknown_'+k,0) else '')+show(m[k]) for k in ('total_tokens','reasoning_tokens')]
        lines.append('| '+' | '.join([LABELS[r['method']],r['mode'],f"{r['answer_correct']}/{r['tasks']}",f"{r['correct']}/{r['tasks']}"]+
            [show(m['calls'])]+values+[show(m[k]) for k in ('duration_seconds','cost_rub')]+
            [f"{r['api_errors']} / {r['format_errors']} / {r['incomplete']}",str(r['method_errors'])])+' |')
    all_metrics=aggregate([s for r in data['records'] for s in r['stages']])
    lines += ['',f"Всего решений: {len(data['records'])}; API-вызовов: {all_metrics['calls']}; "
        f"известная стоимость: {show(all_metrics['cost_rub'])} ₽; неизвестных стоимостей: {all_metrics['unknown_costs']}; "
        f"неизвестных значений токенов: {all_metrics['unknown_total_tokens']}.", '']
    direct_max=next((r for r in summary if r['method']=='direct' and r['mode']=='max'),None)
    if direct_max:
        lines += [f"У прямого ответа в max получено {direct_max['answers_received']} из {direct_max['tasks']} ответов; "
            f"верны {direct_max['answer_correct']} из {direct_max['answers_received']}. Ошибок API: {direct_max['api_errors']}. "
            'Сбой API не является неверным решением модели. Доля от всех попыток в таблице и точность полученных ответов — разные показатели.', '']
    accounting=data['config'].get('budget_accounting')
    if accounting:
        lines += [f"В бюджете учтено {show(accounting['accounted_rub'])} ₽ из {show(accounting['limit_rub'])} ₽, "
                  f"включая резерв {show(accounting['unknown_reserve_rub'])} ₽ на неизвестные списания. "
                  'Резерв — верхняя оценка, а не фактический расход.', '']
    lines += ['','## Выводы','']
    for r in paired:
        lines.append(f"- {r['mode']}, {LABELS[r['method']]}: на {r['common_tasks']} сопоставимых задачах "
            f"верных ответов {r['method_correct']}, прямой ответ — {r['direct_correct']}; "
            f"разница {r['net_gain']:+d}, исправлено {r['gains']}, ухудшено {r['losses']}. "
            f"Исключено пар с отсутствующим/невалидным ответом: {r['excluded_pairs']}. "
            f"Токены относительно прямого ответа: ×{show(r['total_tokens_ratio'])}; "
            f"время: ×{show(r['duration_seconds_ratio'])}; стоимость: ×{show(r['cost_rub_ratio'])}.")
    lines += ['', '## Что меняется при увеличении reasoning', '',
        'Для каждого приёма здесь используется один и тот же набор задач в min и max; '
        'пары с отсутствующим или невалидным ответом исключены. Сетевой сбой не считается доказательством ухудшения промпта.', '']
    for r in interaction:
        if not r['common_tasks']:continue
        text=(f"- **{LABELS[r['method']]}:** на {r['common_tasks']} общих задачах прирост относительно прямого ответа "
              f"в min {r['gain_min']:+d}, в max {r['gain_max']:+d}; исключено задач: {r['excluded_tasks']}. ")
        if r['gain_min']>0 and r['gain_max']<=0:
            text+='Наблюдаемый выигрыш в min не сохраняется в max на этой выборке.'
        elif r['gain_min']==0 and r['gain_max']==0:
            text+='Чистого прироста числа верных ответов нет в обоих режимах; это не доказывает, что приём перестал работать именно из-за reasoning.'
        elif r['gain_max']>0:
            text+='В max остаётся положительный прирост; утверждать, что приём перестал помогать, нельзя.'
        else:
            text+='Данные не подтверждают положительного прироста в max; вывод ограничен этой выборкой.'
        lines.append(text)
    lines += ['', 'Сравнение включает только наблюдения: нулевая разница на восьми задачах не доказывает бесполезность приёма. '
        'Одна попытка на комбинацию не измеряет устойчивость результата. Эксперты — роли одной модели, а не независимые агенты.', '',
        '## Условия и учёт', '',
        '- 8 фиксированных задач, 4 метода, 2 режима. Контекст разных решений не смешивается.',
        '- Метапромпт — два хода одного разговора; в итог входят оба запроса, их время и стоимость.',
        '- Reasoning-токены уже входят в completion/total по проверке пилота; повторно к total не прибавляются.',
        '- min — включённое thinking с минимальным бюджетом 512, а не отключённое мышление; видимое объяснение считается отдельно от thinking.',
        '- max означает бюджет 24576 thinking-токенов, не обещание потратить их целиком. Общий max_tokens=32768, temperature=0.7.',
        '- Время — сумма HTTP-операций метода, не время ввода пользователя и не вся длительность программы.',
        '- Финальная схема одинакова; верно учитывается только завершённый структурированный ответ, соответствующий методу и эталону.',
        '- Ошибка API, формат, обрезание и нарушение метода не считаются просто неправильным решением: подробности в details.csv.',
        '- Неизвестные метрики не заменяются нулями; суммы при пропусках содержат только известные значения, счётчики unknown_* — в CSV.',
        '- Стоимость берётся из usage; вход, выход и reasoning сохранены отдельно. Сырые сообщения reasoning не интерпретируются как точная внутренняя цепочка.',
        '- Запрошенный маршрут закреплён, fallback и HTTP-повторы отключены. Метаданные backend не являются независимой аттестацией.',
        '- Для запросов без usage стоимость не считается нулевой: резервируется консервативная верхняя оценка. '
        'Продолжение не повторяет завершённые попытки, включая ошибки. Контроль бюджета локальный, не банковская гарантия биллинга.',
        '- Порядок 64 решений перемешан с seed=42. Эталоны не включены в сообщения модели.', '',
        '## Подробные результаты', '', '| Задача | Метод | Режим | Ответ | Ответ верен | Полный успех | Ошибка |', '|---|---|---|---|---|---|---|']
    for r in data['records']:
        answer=(r['parsed']['answer'] if r['parsed'] else '—').replace('|','\\|').replace('\n',' ')
        lines.append(f"| {r['task_id']} | {r['method']} | {r['mode']} | {answer} | {'да' if r['answer_correct'] else 'нет'} | {'да' if r['correct'] else 'нет'} | {r['failure'] or '—'} |")
    (directory/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run_experiment(api,output_dir,*,pilot_path,budget_rub,notify=None,resume_dir=None):
    if not math.isfinite(budget_rub) or budget_rub<=0: raise ApiError('Бюджет должен быть положительным.')
    try:
        pilot=json.loads(Path(pilot_path).read_text(encoding='utf-8'))
        valid=(pilot['verified'] is True and pilot['config']==config(api) and len(pilot['records'])==2
               and {r['mode'] for r in pilot['records']}==set(MODES)
               and not any(pilot_checks(r,api) for r in pilot['records']))
    except (OSError,ValueError,KeyError,TypeError): valid=False
    if not valid: raise ApiError('Нужен успешный пилот с теми же моделью, маршрутом и настройками.')
    checked=preflight(api)
    if resume_dir is None:
        directory=Path(output_dir)/identifier(); directory.mkdir(parents=True,exist_ok=False)
        (directory/'raw').mkdir()
        atomic_json(directory/'catalog.json',checked)
        atomic_json(directory/'tasks.json',TASKS)
        data={'config':{**config(api),'status':'running','budget_rub':budget_rub,'order_seed':42,
            'pilot_path':str(pilot_path),'started_at':datetime.now().astimezone().isoformat(timespec='seconds')},'records':[]}
    else:
        directory=Path(resume_dir)
        try:
            data=json.loads((directory/'results.json').read_text(encoding='utf-8'))
            if any(data['config'].get(k)!=v for k,v in config(api).items()) or data['config']['budget_rub']!=budget_rub:
                raise ValueError
            keys=[(r['task_id'],r['method'],r['mode']) for r in data['records']]
            valid_keys={(t['id'],method,mode) for t in TASKS for method in METHODS for mode in MODES}
            if len(keys)!=len(set(keys)) or not set(keys)<=valid_keys or any(not r['stages'] for r in data['records']):
                raise ValueError
        except (OSError,ValueError,KeyError,TypeError):
            raise ApiError('Продолжение требует неповреждённые результаты с прежними настройками и тем же бюджетом.') from None
        data['config'].setdefault('resumes',[]).append(datetime.now().astimezone().isoformat(timespec='seconds'))
        data['config']['status']='running'
        atomic_json(directory/('catalog-resume-'+identifier()+'.json'),checked)
    budget=Budget(api,budget_rub)
    budget.restore(data['records'])
    data['config']['budget_accounting']=budget.accounting()
    work=[(task,method,mode) for task in TASKS for method in METHODS for mode in MODES]
    random.Random(42).shuffle(work)
    completed={(r['task_id'],r['method'],r['mode']) for r in data['records']}
    work=[(t,m,r) for t,m,r in work if (t['id'],m,r) not in completed]
    write_reports(directory,data)
    try:
        for task,method,mode in work:
            raw=directory/'raw'/f"{task['id']}_{method}_{mode}.json"
            try:
                row=solve_one(api,task,method,mode,checkpoint=lambda r:atomic_json(raw,r),
                              before=budget.before,after=budget.after)
            except BudgetStop as exc:
                if exc.row and exc.row['stages']: data['records'].append(exc.row)
                data['config']['status']='budget_stopped'
                break
            data['records'].append(row)
            data['config']['budget_accounting']=budget.accounting()
            write_reports(directory,data)
            if notify: notify(f"{len(data['records'])}/64: {task['id']} {method} {mode}; верно={row['correct']}; ₽={show(row['metrics']['cost_rub'])}")
        else: data['config']['status']='complete'
    except BaseException as exc:
        partial=getattr(exc,'row',None)
        if partial and partial['stages']:
            data['records'].append(partial)
        data['config']['status']='interrupted'
        raise
    finally:
        data['config']['budget_accounting']=budget.accounting()
        write_reports(directory,data)
        api.journal.record('EXPERIMENT_END',{'path':str(directory),'status':data['config']['status'],'spent':budget.spent})
    return directory
