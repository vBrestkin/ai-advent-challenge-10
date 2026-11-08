"""Один фактор temperature; строгий анализ без исправления ответов модели."""
from copy import deepcopy
import csv
from datetime import datetime
from itertools import combinations
import io
import json
import math
from pathlib import Path
import random
import re
import unicodedata

from jsonschema import Draft202012Validator

from .core import ApiError,RequestInterrupted,atomic_json,identifier

TEMPERATURES=(0,.7,1.2)
REASONING={'enabled':True,'max_tokens':512}
MAX_TOKENS=4096
SYSTEM='Следуй формату пользователя. Верни ответ без Markdown, кодовых блоков и пояснений.'
TASKS={
    'logic':{'title':'Задача с одним ответом','prompt':
        'Сколько двоичных строк длины 12 содержат ровно пять единиц, причём никакие две единицы '
        'не стоят рядом? Ведущие нули разрешены. Ответь только JSON-объектом с единственным '
        'полем answer: целое число. Не заключай JSON в тройные обратные кавычки. '
        'Ответ начинается символом { и заканчивается }.',
        'schema':{'type':'object','properties':{'answer':{'type':'integer'}},
                  'required':['answer'],'additionalProperties':False}},
    'names':{'title':'Генерация названий','prompt':
        'Придумай пять разных коротких названий приложения для изучения Python. '
        'Ответь только JSON-объектом с единственным полем names: массив из пяти строк-названий. '
        'Никаких описаний и других полей. Не заключай JSON в тройные обратные кавычки. '
        'Ответ начинается символом { и заканчивается }.',
        'schema':{'type':'object','properties':{'names':{'type':'array','items':{'type':'string'}}},
                  'required':['names'],'additionalProperties':False}}
}


def normalize_name(value):
    return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',value).casefold()).strip()


def strict_json(content):
    def constant(_):raise ValueError
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError
            result[key]=value
        return result
    return json.loads(content,parse_constant=constant,object_pairs_hook=pairs)


def assess(task_id,content):
    result=dict(json_valid=False,schema_valid=False,requirements_met=False,
        answer_correct=None,semantic_key=None,normalized_names=None,parsed=None,validation_error=None)
    try:data=strict_json(content)
    except (ValueError,TypeError,RecursionError):
        result['validation_error']='Исходный текст не разбирается как строгий JSON.'
        return result
    result['json_valid']=True; result['parsed']=data
    if not Draft202012Validator(TASKS[task_id]['schema']).is_valid(data):
        result['validation_error']='JSON не соответствует ожидаемым полям и типам.'
        return result
    result['schema_valid']=True
    if task_id=='logic':
        result['answer_correct']=data['answer']==56
        result['requirements_met']=result['answer_correct']
        result['semantic_key']=str(int(data['answer']))
    else:
        names=[normalize_name(v) for v in data['names']]
        result['normalized_names']=names
        result['requirements_met']=len(names)==5 and all(names) and len(set(names))==5
        result['semantic_key']=json.dumps(sorted(names),ensure_ascii=False)
    if not result['requirements_met']:result['validation_error']='Неверный ответ или нарушены требования к пяти разным непустым названиям.'
    return result


def preflight(api):
    if api.model!='google/gemini-2.5-flash-lite':
        raise ApiError('Текущий профиль reasoning-бюджета проверяется для google/gemini-2.5-flash-lite.')
    catalog=api.catalog()
    provider=next((p for p in catalog.get('providers',[]) if p.get('name')==api.provider),None)
    if provider is None:raise ApiError('Модель не имеет выбранного маршрута.')
    if not {'temperature','reasoning','max_tokens'}<=set(provider.get('supported_parameters',[])):
        raise ApiError('Маршрут не заявляет необходимые temperature, reasoning и max_tokens.')
    if provider.get('max_completion_tokens',catalog.get('max_completion_tokens',0))<MAX_TOKENS:
        raise ApiError('У модели недостаточный лимит ответа.')
    pricing=provider.get('pricing') or {}
    try:
        rates={k:float(pricing[k]) for k in ('prompt_per_million','completion_per_million')}
        if pricing.get('currency')!='RUB' or any(not math.isfinite(v) or v<=0 for v in rates.values()):raise ValueError
    except (KeyError,ValueError,TypeError):raise ApiError('Неизвестна рублёвая цена маршрута.') from None
    api.price_limits={'prompt':rates['prompt_per_million']*1.01,'completion':rates['completion_per_million']*1.01}
    return {'catalog':catalog,'provider':provider,'rates':rates,'price_limits':deepcopy(api.price_limits)}


METRICS=('total_tokens','prompt_tokens','completion_tokens','reasoning_tokens','cost_rub','duration_seconds')


def totals(records):
    result={'calls':len(records)}
    for key in METRICS:
        known=[r.get(key) for r in records if r.get(key) is not None]
        result[key]=round(sum(known),10) if known else None
        result['unknown_'+('costs' if key=='cost_rub' else key)]=len(records)-len(known)
    return result


def summarize(records):
    result=[]
    for task_id in TASKS:
        for temperature in TEMPERATURES:
            rows=[r for r in records if r['task_id']==task_id and r['temperature']==temperature]
            if not rows:continue
            received=[r for r in rows if r['received']]
            name_rows=[r for r in received if task_id=='names' and r['requirements_met']]
            sets=[set(r['normalized_names']) for r in name_rows]
            jaccard=[len(a&b)/len(a|b) for a,b in combinations(sets,2)]
            pool=set().union(*sets) if sets else set()
            slots=sum(len(r['normalized_names']) for r in name_rows)
            result.append(dict(task_id=task_id,temperature=temperature,attempts=len(rows),
                answers_received=len(received),json_valid=sum(r['json_valid'] for r in received),
                schema_valid=sum(r['schema_valid'] for r in received),
                format_errors=sum(not r['schema_valid'] for r in received),
                wrong_answers=sum(r['answer_correct'] is False for r in received),
                correct_answers=sum(r['answer_correct'] is True for r in received),
                requirements_met=sum(r['requirements_met'] for r in received),
                requirements_failures=sum(r['schema_valid'] and not r['requirements_met'] for r in received),
                api_errors=sum(bool(r['api_error']) for r in rows),
                model_mismatches=sum(r.get('failure')=='model_mismatch' for r in rows),
                incomplete=sum(not r['complete'] and not r['api_error'] for r in rows),
                unique_raw=len({r['content'] for r in received}),
                unique_semantic=len({r['semantic_key'] for r in received if r['schema_valid']}),
                valid_name_responses=len(name_rows),unique_names=len(pool),total_name_slots=slots,
                name_uniqueness_ratio=len(pool)/slots if slots else None,
                mean_jaccard=sum(jaccard)/len(jaccard) if jaccard else None,
                mean_jaccard_distance=1-sum(jaccard)/len(jaccard) if jaccard else None,
                **totals(rows)))
    return result


def zero_findings(summary):
    lines=[]
    for row in summary:
        if row['temperature']!=0:continue
        prefix=TASKS[row['task_id']]['title']+': '
        if not row['answers_received']:lines.append(prefix+'не получено завершённых ответов; повторяемость не оценивалась.')
        elif row['unique_raw']==1:
            lines.append(prefix+f"из {row['answers_received']} полученных ответов все тексты одинаковы. Это наблюдение, а не гарантия для следующих запросов.")
        else:
            lines.append(prefix+f"получены различающиеся тексты: {row['unique_raw']} уникальных из {row['answers_received']}. Temperature = 0 не дала одинаковый текст во всех этих повторах.")
        if row['schema_valid']:
            lines[-1]+=f" После нормализации: {row['unique_semantic']} уникальных наборов данных среди {row['schema_valid']} структурно корректных ответов."
    return '\n'.join(lines) or 'Temperature = 0 пока не проверена.'


def temperature_conclusions(summary):
    lines=[]
    logic=[r for r in summary if r['task_id']=='logic' and r['schema_valid']]
    if logic:
        lines.append('Правильность математического ответа среди структурно проверяемых ответов: '+
            '; '.join(f"T={r['temperature']}: {r['correct_answers']}/{r['schema_valid']}" for r in logic)+'. '
            'API-сбои и ошибки формата в эти знаменатели не включены.')
    names={r['temperature']:r for r in summary if r['task_id']=='names'}
    if .7 in names and 1.2 in names:
        middle,high=names[.7],names[1.2]
        if middle['valid_name_responses'] and middle['valid_name_responses']==high['valid_name_responses']:
            increased=high['unique_names']>middle['unique_names']
            lines.append(f"Повышение температуры с 0.7 до 1.2 {'увеличило' if increased else 'не увеличило'} "
                f"пул уникальных названий: {middle['unique_names']} → {high['unique_names']} при одинаковом числе "
                f"пригодных ответов ({middle['valid_name_responses']}). Рост температуры не обязан давать монотонный рост разнообразия.")
        elif middle['valid_name_responses']!=high['valid_name_responses']:
            lines.append('Число пригодных списков при 0.7 и 1.2 различается; напрямую сравнивать размер общего пула названий нельзя.')
    return '\n\n'.join(lines)


class BudgetStop(Exception):pass


class Budget:
    def __init__(self,api,limit):
        self.api,self.limit,self.spent,self.unknown_reserve,self.pending=api,limit,0.,0.,0.

    def reservation(self,request):
        size=len(json.dumps(request,ensure_ascii=False).encode())+2048
        old=request.get('provider',{}).get('max_price',{})
        rates={k:max(self.api.price_limits[k],old.get(k,0)) for k in ('prompt','completion')}
        return (size*rates['prompt']+request.get('max_tokens',MAX_TOKENS)*rates['completion'])/1_000_000

    def before(self,messages,parameters):
        self.pending=self.reservation({'messages':messages,'model':self.api.model,**parameters})
        if self.spent+self.unknown_reserve+self.pending>self.limit:raise BudgetStop

    def after(self,row):
        if row['cost_rub'] is None:
            self.unknown_reserve+=self.pending; row['budget_reserve_rub']=self.pending
        else:self.spent+=row['cost_rub']

    def restore(self,rows):
        for row in rows:
            if row['cost_rub'] is None:
                if not row.get('request'):raise ApiError('Нельзя восстановить резерв без запроса.')
                self.unknown_reserve+=row.get('budget_reserve_rub') or self.reservation(row['request'])
            else:self.spent+=row['cost_rub']

    def accounting(self):
        return dict(known_cost_rub=round(self.spent,10),unknown_reserve_rub=round(self.unknown_reserve,10),
                    accounted_rub=round(self.spent+self.unknown_reserve,10),limit_rub=self.limit)


def config(api,repeats):
    return dict(version=1,model=api.model,provider=api.provider,repeats=repeats,
        temperatures=list(TEMPERATURES),reasoning=deepcopy(REASONING),max_tokens=MAX_TOKENS,
        system=SYSTEM,tasks=deepcopy(TASKS),order_seed=42)


def csv_text(rows):
    if not rows:return ''
    file=io.StringIO(newline=''); writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return file.getvalue()


def show(value):
    if value is None:return '—'
    return f'{value:.6f}'.rstrip('0').rstrip('.') if isinstance(value,float) else str(value)


def write_reports(directory,data):
    atomic_json(directory/'results.json',data)
    summary=summarize(data['records'])
    (directory/'summary.csv').write_text(csv_text(summary),encoding='utf-8-sig')
    fields=('task_id','temperature','repeat','received','complete','failure','json_valid','schema_valid',
        'answer_correct','requirements_met','api_error','finish_reason','content',*METRICS)
    details=[{k:r.get(k) for k in fields} for r in data['records']]
    (directory/'details.csv').write_text(csv_text(details),encoding='utf-8-sig')
    lines=['# День 4 — температура','',f"Модель: `{data['config']['model']}`; маршрут: `{data['config']['provider']}`. Статус: {data['config']['status']}.",'',
        '## Сводная таблица','',
        '| Задача | T | Ответов | JSON/схема | Верно / требования выполнены | Ошибки формата | Неверные ответы | API | Уникальных текстов | Уникальных данных |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        score=r['correct_answers'] if r['task_id']=='logic' else r['requirements_met']
        lines.append('| '+' | '.join([TASKS[r['task_id']]['title'],show(r['temperature']),
            f"{r['answers_received']}/{r['attempts']}",str(r['schema_valid']),str(score),str(r['format_errors']),
            str(r['wrong_answers']) if r['task_id']=='logic' else 'не применимо',str(r['api_errors']),
            str(r['unique_raw']),str(r['unique_semantic'])])+' |')
    lines+=['','Ошибка формата не приравнивается к математически неверному ответу: при неразбираемом JSON правильность неизвестна. '
        'Для генерации проверяются количество и различие названий; художественное качество не оценивается.', '',
        '## Разнообразие названий','',
        '| T | Пригодных списков | Уникальных названий | Всего позиций | Доля уникальных | Среднее сходство Жаккара |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['task_id']=='names':
            lines.append('| '+' | '.join(show(r[k]) for k in ('temperature','valid_name_responses','unique_names',
                'total_name_slots','name_uniqueness_ratio','mean_jaccard'))+' |')
    lines+=['','Жаккар = пересечение / объединение пар наборов: 1 — одинаковые наборы, 0 — нет общих названий. '
        'При менее двух пригодных ответах среднее сходство не вычисляется. Это лексическая метрика, не оценка качества идей.', '',
        '## temperature = 0: одинаков ли ответ?', '',zero_findings(summary),'',
        '## Выводы','']
    for r in summary:
        if r['task_id']=='logic':
            lines.append(f"- T={r['temperature']}, логика: верных {r['correct_answers']}, неверных {r['wrong_answers']}, "
                f"форматных сбоев {r['format_errors']}, ошибок API {r['api_errors']} из {r['attempts']} попыток. "
                'Изменение единственного правильного числа не является полезным разнообразием.')
        else:
            lines.append(f"- T={r['temperature']}, названия: {r['unique_names']} различных названий в "
                f"{r['valid_name_responses']} пригодных списках, {r['unique_semantic']} различных наборов среди "
                f"{r['schema_valid']} структурно корректных ответов. Доля уникальных: {show(r['name_uniqueness_ratio'])}.")
    lines += ['',temperature_conclusions(summary),'', 'Практическое применение: 0 — отправная точка для задач с фиксированным ответом и повторяемой формой; '
        'правильность всё равно нужно проверять. 0.7 — кандидат для обычной генерации идей, '
        '1.2 — кандидат для поиска более необычных вариантов с последующим отбором. '
        'Это направления использования, а не гарантия качества или монотонного роста разнообразия. '
        'Для автоматической обработки JSON одной инструкции может быть недостаточно — формат проверяется кодом.', '',
        '## Ресурсы','', '| Задача | T | Все токены (известные) | Reasoning-токены | Время, с | Известная стоимость, ₽ | Неизвестных списаний |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append('| '+' | '.join([r['task_id']]+[show(r[k]) for k in ('temperature','total_tokens','reasoning_tokens','duration_seconds','cost_rub','unknown_costs')])+' |')
    metrics=totals(data['records']); budget=data['config'].get('budget_accounting',{})
    lines += ['',f"Вызовов: {metrics['calls']}. Известная стоимость: {show(metrics['cost_rub'])} ₽. "
        f"Неизвестных стоимостей: {metrics['unknown_costs']}; неизвестных значений токенов: {metrics['unknown_total_tokens']}.",
        f"Учёт бюджета: {show(budget.get('accounted_rub'))} ₽ из {show(budget.get('limit_rub'))} ₽, "
        f"в том числе резерв неизвестных списаний {show(budget.get('unknown_reserve_rub'))} ₽. Резерв не равен фактическому расходу.", '',
        '## Условия сравнения','',
        '- Один и тот же пользовательский запрос и системная инструкция для всех температур каждого типа задачи.',
        '- Пять повторов по умолчанию. Каждая попытка — пустая история с system/user, без переноса прошлых ответов.',
        '- Reasoning включён с одним бюджетом 512, max_tokens=4096. Меняется только temperature.',
        '- Нативная JSON Schema/JSON mode, stop и seed не передаются. Ожидаемый JSON задан только промптом.',
        '- Нормализация названий: Unicode NFKC, casefold, удаление крайних и схлопывание внутренних пробелов. Пунктуация не удаляется.',
        '- Уникальные тексты — точное совпадение исходных ответов. Уникальные данные — JSON после нормализации; порядок названий не учитывается.',
        '- Разнообразие названий считается только по пригодным спискам из пяти непустых различных названий.',
        '- API-, незавершённые и несовпадающие по модели ответы исключаются из оценки разнообразия. Ошибки JSON сохраняются отдельно.',
        '- Шлюз выбирает backend внутри запрошенного маршрута; параметры переданы явно, внутренняя реализация не аттестована независимо.',
        '- Порядок шести комбинаций перемешан внутри каждого повтора с seed=42 только на стороне программы; это не seed модели.',
        '- Разнообразие может быть немонотонным; пять наблюдений на одном запросе не дают универсального вывода о всех задачах.',
        '- Сырые ответы не исправляются: Markdown не вырезается. Thinking уже входит в total и не прибавляется второй раз.',
        '- Полные данные: results.json, raw/*.json, catalog*.json. Схема проверки локальная и не ограничивает генерацию на сервере.', '',
        '## Примеры исходных ответов','']
    for r in data['records']:
        text=r['content'] or r['api_error'] or '(пустой ответ)'
        if len(text)>1500:text=text[:1500]+'\n[Сокращено; полный текст в raw.]'
        lines += [f"### {r['task_id']}, T={r['temperature']}, повтор {r['repeat']}",'','````text',text,'````','']
    (directory/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def run_experiment(api,output_dir,*,repeats=5,budget_rub=5,notify=None,resume_dir=None):
    if type(repeats) is not int or not 1<=repeats<=100 or not math.isfinite(budget_rub) or budget_rub<=0:
        raise ApiError('Некорректное число повторов или бюджет.')
    checked=preflight(api)
    if resume_dir is None:
        directory=Path(output_dir)/identifier(); directory.mkdir(parents=True,exist_ok=False)
        (directory/'raw').mkdir(); atomic_json(directory/'catalog.json',checked)
        data={'config':{**config(api,repeats),'status':'running','budget_rub':budget_rub,
            'started_at':datetime.now().astimezone().isoformat(timespec='seconds')},'records':[]}
    else:
        directory=Path(resume_dir)
        try:
            data=json.loads((directory/'results.json').read_text())
            if any(data['config'].get(k)!=v for k,v in config(api,repeats).items()) or data['config']['budget_rub']!=budget_rub:
                raise ValueError
            keys=[(r['task_id'],r['temperature'],r['repeat']) for r in data['records']]
            valid={(task,t,n) for task in TASKS for t in TEMPERATURES for n in range(1,repeats+1)}
            if len(keys)!=len(set(keys)) or not set(keys)<=valid:raise ValueError
        except (OSError,ValueError,KeyError,TypeError):
            raise ApiError('Для продолжения нужны прежние настройки, бюджет и корректный файл результатов.') from None
        data['config']['status']='running'
        data['config'].setdefault('resumes',[]).append(datetime.now().astimezone().isoformat(timespec='seconds'))
        atomic_json(directory/('catalog-resume-'+identifier()+'.json'),checked)
    budget=Budget(api,budget_rub); budget.restore(data['records'])
    data['config']['budget_accounting']=budget.accounting(); write_reports(directory,data)
    done={(r['task_id'],r['temperature'],r['repeat']) for r in data['records']}
    randomizer=random.Random(42)
    try:
        stop=False
        for repeat in range(1,repeats+1):
            order=[(task,t) for task in TASKS for t in TEMPERATURES]; randomizer.shuffle(order)
            for task_id,temperature in order:
                if (task_id,temperature,repeat) in done:continue
                messages=[{'role':'system','content':SYSTEM},{'role':'user','content':TASKS[task_id]['prompt']}]
                parameters={'max_tokens':MAX_TOKENS,'temperature':temperature,'reasoning':deepcopy(REASONING)}
                try:budget.before(messages,parameters)
                except BudgetStop:
                    data['config']['status']='budget_stopped'; stop=True; break
                interrupted=False
                try:row=api.send(messages,parameters)
                except RequestInterrupted as exc:row=exc.result; interrupted=True
                except ApiError as exc:row=exc.result
                row.update(task_id=task_id,temperature=temperature,repeat=repeat)
                row['received']=bool(row['complete'] and not row['api_error'] and row['actual_model']==api.model)
                evaluated=assess(task_id,row['content']) if row['received'] else assess(task_id,'')
                row.update(evaluated); budget.after(row)
                row['failure']=('interrupted' if interrupted else 'api_error' if row['api_error'] else
                    'model_mismatch' if row['actual_model']!=api.model else 'incomplete' if not row['complete'] else
                    'format_error' if not row['schema_valid'] else 'wrong_answer' if row['answer_correct'] is False else
                    'requirements_error' if not row['requirements_met'] else None)
                row=api.journal.scrub(row); data['records'].append(row)
                atomic_json(directory/'raw'/f'{task_id}_{temperature}_{repeat:02d}.json',row)
                data['config']['budget_accounting']=budget.accounting(); write_reports(directory,data)
                api.journal.record('EXPERIMENT_RESULT',row)
                if notify:
                    formatted=str(row['schema_valid']) if row['received'] else 'не оценено'
                    notify(f"{len(data['records'])}/{repeats*6}: {task_id}, T={temperature}; JSON={formatted}; исход={row['failure'] or 'успех'}")
                if interrupted:raise KeyboardInterrupt
                if row['http_status'] in (400,401,402,403,404,413,422):
                    data['config']['status']='request_rejected'; stop=True; break
            if stop:break
        else:data['config']['status']='complete'
    except BaseException:
        data['config']['status']='interrupted'
        raise
    finally:
        data['config']['budget_accounting']=budget.accounting(); write_reports(directory,data)
        api.journal.record('EXPERIMENT_END',{'directory':str(directory),'status':data['config']['status'],
            'budget':budget.accounting(),'rows':len(data['records'])})
    return directory
