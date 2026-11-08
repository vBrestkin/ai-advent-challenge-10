import io
import json
from itertools import combinations, product
from fractions import Fraction
from pathlib import Path

import httpx
import pytest
from pytest_bdd import given, when, then, scenarios

from day3.core import ApiClient, ApiError, Journal
from day3.history import Conversation, HistoryError
from day3.tasks import TASKS, verify_answer
from day3.experiment import (MODES, METHODS, FINAL_SCHEMA, solve_one, run_pilot,
    run_experiment, aggregate, summarize, preflight)
from day3.cli import main

scenarios('features/day3.feature')
SECRET='private-test-key'
MODEL='google/gemini-2.5-flash-lite'
PROVIDER='google-ai-studio/flex'


def catalog():
    provider={'name':PROVIDER,'max_completion_tokens':65535,
        'supported_parameters':['reasoning','response_format','structured_outputs','temperature','max_tokens'],
        'capabilities':{'reasoning_toggle':{'status':'supported'},'json_schema_strict':{'status':'supported'}},
        'pricing':{'currency':'RUB','prompt_per_million':'6','completion_per_million':'24'}}
    return {'id':MODEL,'max_completion_tokens':65535,'providers':[provider],'top_provider':provider}


def completion(payload, *, reasoning=None, content=None, reason='stop'):
    schema=payload.get('response_format',{}).get('json_schema',{}).get('name')
    if schema=='solution_prompt':
        answer={'prompt':'Проверь ограничения и вычисли ответ исходной задачи.'}
    else:
        answer={'answer':'14','explanation':'','experts':[]}
        if schema=='solution':
            user=payload['messages'][-1]['content']
            if 'ПОШАГОВО' in user:
                answer['explanation']='Краткая проверка ограничений.'
            if 'ЭКСПЕРТЫ' in user:
                answer['explanation']='Общий итог.'
                answer['experts']=[{'role':r,'answer':'14','explanation':'Проверено.'} for r in ('analyst','engineer','critic')]
    enabled=payload.get('reasoning',{}).get('enabled',False)
    return {'model':MODEL,'provider':PROVIDER,'choices':[{'finish_reason':reason,'message':{
        'role':'assistant','content':json.dumps(answer,ensure_ascii=False) if content is None else content}}],
        'usage':{'prompt_tokens':10,'completion_tokens':20,'total_tokens':30,'cost_rub':.01,
                 'completion_tokens_details':{'reasoning_tokens':(12 if payload.get('reasoning',{}).get('max_tokens')==24576 else 2) if reasoning is None else reasoning}}}


@pytest.fixture
def state(tmp_path):
    journal=Journal(tmp_path/'logs',SECRET)
    data=dict(root=tmp_path,journal=journal,calls=[],fault=None)
    def respond(request):
        if request.method=='GET':
            return httpx.Response(200,json={'data':[catalog()]})
        payload=json.loads(request.content)
        data['calls'].append(payload)
        if data['fault']=='timeout':
            raise httpx.ReadTimeout(SECRET,request=request)
        if data['fault']=='http':
            return httpx.Response(503,text=SECRET)
        body=completion(payload)
        if data['fault']=='length': body['choices'][0]['finish_reason']='length'
        if data['fault']=='bad_plan': body['choices'][0]['message']['content']='{}'
        return httpx.Response(200,json=body)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        data['api']=ApiClient(SECRET,MODEL,journal,client,provider=PROVIDER)
        yield data
    journal.close()


@given('маршрут поддерживает reasoning и строгую схему')
def ready(state):
    pass


@when('выполнен пилот с минимальным и максимальным reasoning')
def pilot(state):
    state['pilot']=run_pilot(state['api'],state['root']/'pilots')


@then('сделаны ровно два запроса с разными бюджетами')
def modes(state):
    assert len(state['calls'])==2
    assert [p['reasoning'] for p in state['calls']]==[{'enabled':True,'max_tokens':512},{'enabled':True,'max_tokens':24576}]
    assert all(p['max_tokens']==32768 for p in state['calls'])
    assert all(p['provider']['only']==[PROVIDER] and not p['provider']['allow_fallbacks'] for p in state['calls'])
    assert all('reasoning_effort' not in p for p in state['calls'])


@then('сохранено подтверждение режимов и их стоимость')
def pilot_saved(state):
    data=json.loads(state['pilot'].read_text())
    assert data['verified'] is True
    assert data['metrics']['cost_rub']==pytest.approx(.02)
    assert [r['metrics']['reasoning_tokens'] for r in data['records']]==[2,12]
    assert (state['pilot'].parent/'report.md').exists()


@when('задача решена через составление промпта с максимальным reasoning')
def metaprompt(state):
    state['row']=solve_one(state['api'],TASKS[-1],'metaprompt','max')


@then('второй запрос содержит первый ход разговора')
def previous_turn(state):
    assert len(state['calls'])==2
    assert [m['role'] for m in state['calls'][1]['messages']]==['system','user','assistant','user']
    assert 'Проверь ограничения' in state['calls'][1]['messages'][-1]['content']
    assert TASKS[-1]['question'] in state['calls'][1]['messages'][-1]['content']
    assert all(p['reasoning']==MODES['max'] for p in state['calls'])


@then('ресурсы обоих этапов суммируются без повторного учёта reasoning')
def both_costs(state):
    metrics=state['row']['metrics']
    assert metrics['total_tokens']==60 and metrics['reasoning_tokens']==24
    assert metrics['cost_rub']==pytest.approx(.02)
    assert metrics['calls']==2 and state['row']['correct']


@when('разговор сохранён и восстановлен после первого ответа')
def restored(state):
    path=state['root']/'chat.json'
    conversation=Conversation.create(path,MODEL,provider=PROVIDER,mode='max')
    conversation.ask('Запомни слово Сосна.',state['api'])
    conversation=Conversation.load(path)
    conversation.ask('Какое слово?',state['api'])
    state['conversation']=conversation


@then('второй ход использует историю и прежний режим reasoning')
def history_kept(state):
    assert len(state['calls'][1]['messages'])==3
    assert state['calls'][1]['reasoning']==MODES['max']
    assert state['conversation'].data['mode']=='max'
    assert len(state['conversation'].messages)==4


@pytest.mark.parametrize('error',[httpx.ConnectError,httpx.ConnectTimeout,httpx.ReadTimeout,
    httpx.WriteTimeout,httpx.PoolTimeout,httpx.ReadError,httpx.WriteError,httpx.RemoteProtocolError])
def test_network_errors_safe_no_retries(state,error):
    calls=[]
    def fail(request):
        calls.append(request)
        raise error(SECRET,request=request)
    with httpx.Client(transport=httpx.MockTransport(fail)) as client:
        api=ApiClient(SECRET,MODEL,state['journal'],client,provider=PROVIDER)
        with pytest.raises(ApiError) as exc: api.send([{'role':'user','content':'hello'}])
    assert len(calls)==1 and SECRET not in str(exc.value)
    assert SECRET not in state['journal'].path.read_text()


@pytest.mark.parametrize('status',[301,400,401,402,403,404,408,413,429,500,502,503,504])
def test_http_errors_not_format_errors(state,status):
    with httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(status,text=SECRET))) as client:
        with pytest.raises(ApiError) as exc:
            ApiClient(SECRET,MODEL,state['journal'],client,provider=PROVIDER).send([{'role':'user','content':'hi'}])
    assert str(status) in str(exc.value) and SECRET not in state['journal'].path.read_text()


@pytest.mark.parametrize('mutation',['route','reasoning','schema','limit','currency'])
def test_preflight_rejects_bad_capabilities(state,monkeypatch,mutation):
    data=catalog()
    p=data['providers'][0]
    if mutation=='route': p['name']='elsewhere'
    if mutation=='reasoning': p['supported_parameters'].remove('reasoning')
    if mutation=='schema': p['capabilities']['json_schema_strict']['status']='unknown'
    if mutation=='limit': p['max_completion_tokens']=4096
    if mutation=='currency': p['pricing']['currency']='USD'
    monkeypatch.setattr(state['api'],'catalog',lambda:data)
    with pytest.raises(ApiError): preflight(state['api'])
    assert not state['calls']


@pytest.mark.parametrize('reasoning',[None,'missing'])
def test_missing_reasoning_is_unknown_not_zero(state,reasoning):
    def respond(request):
        body=completion(json.loads(request.content))
        if reasoning=='missing': body['usage'].pop('completion_tokens_details')
        else: body['usage']['completion_tokens_details']['reasoning_tokens']=None
        return httpx.Response(200,json=body)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result=ApiClient(SECRET,MODEL,state['journal'],client,provider=PROVIDER).send([{'role':'user','content':'hi'}])
    assert result['reasoning_tokens'] is None


def test_failed_pilot_cannot_authorize_full_run(state,monkeypatch):
    original=state['api'].send
    def send(*args,**kwargs):
        row=original(*args,**kwargs)
        row['reasoning_tokens']=None
        return row
    monkeypatch.setattr(state['api'],'send',send)
    path=run_pilot(state['api'],state['root']/'pilots')
    assert json.loads(path.read_text())['verified'] is False
    before=len(state['calls'])
    with pytest.raises(ApiError): run_experiment(state['api'],state['root']/'results',pilot_path=path,budget_rub=10)
    assert len(state['calls'])==before


@pytest.mark.parametrize('fault',['timeout','http','length','bad_plan'])
def test_bad_first_metaprompt_stage_does_not_send_second(state,fault):
    state['fault']=fault
    row=solve_one(state['api'],TASKS[0],'metaprompt','max')
    assert len(state['calls'])==1 and not row['correct']
    assert row['failure'] in {'api_error','incomplete','format_error'}
    assert row['metrics']['calls']==1


@pytest.mark.parametrize('method',METHODS)
@pytest.mark.parametrize('mode',MODES)
def test_methods_use_requested_mode_and_schema_without_oracle(state,method,mode):
    row=solve_one(state['api'],TASKS[-1],method,mode)
    assert row['correct'] is True
    assert len(state['calls'])==(2 if method=='metaprompt' else 1)
    for payload in state['calls']:
        assert payload['reasoning']==MODES[mode]
        assert payload['response_format']['json_schema']['strict'] is True
        assert 'expected' not in json.dumps(payload).lower()
        assert 'Эталон' not in json.dumps(payload,ensure_ascii=False)
    if method=='experts':
        assert len(row['parsed']['experts'])==3


def test_experts_must_return_three_roles(state,monkeypatch):
    original=state['api'].send
    def send(*args,**kwargs):
        row=original(*args,**kwargs)
        row['content']=json.dumps({'answer':'14','explanation':'итог','experts':[]})
        return row
    monkeypatch.setattr(state['api'],'send',send)
    row=solve_one(state['api'],TASKS[-1],'experts','min')
    assert row['answer_correct'] and not row['correct'] and row['failure']=='method_error'


def test_unknown_metrics_not_free_and_not_double_counted():
    rows=[dict(total_tokens=30,reasoning_tokens=12,cost_rub=.1,duration_seconds=1,
               prompt_tokens=10,completion_tokens=20),
          dict(total_tokens=None,reasoning_tokens=None,cost_rub=None,duration_seconds=2,
               prompt_tokens=None,completion_tokens=None)]
    metrics=aggregate(rows)
    assert metrics['total_tokens']==30 and metrics['reasoning_tokens']==12
    assert metrics['unknown_total_tokens']==1 and metrics['unknown_costs']==1
    assert metrics['cost_rub']==.1 and metrics['duration_seconds']==3


@pytest.mark.parametrize('task_id,answer,valid',[('T1','D, A',True),('T1','A',False),
    ('T2','56',True),('T2','56 слов',False),('T5','4/26',True),('T5','2/0',False),
    ('T5','0.1538',False),('T7','A,A,B',True),('T7','B,A,A',False)])
def test_answer_grader(task_id,answer,valid):
    assert verify_answer(next(t for t in TASKS if t['id']==task_id),answer) is valid


def test_task_oracles_independently_verified():
    expected={t['id']:t['expected'] for t in TASKS}
    knights=[]
    for A,B,C,D in product([False,True],repeat=4):
        if [A,B,C,D]==[not B,C==D,not A and not B,(A+B)==1]: knights.append((A,B,C,D))
    assert knights==[(True,False,False,True)] and expected['T1']=='A,D'
    strings=sum(sum(bits)==5 and all(not(bits[i] and bits[i+1]) for i in range(11)) for bits in product((0,1),repeat=12))
    assert strings==int(expected['T2'])==56
    assert sum(sum(n%d==0 for d in (6,10,15))==1 for n in range(1,1001))==int(expected['T3'])
    items=[(3,4),(4,5),(5,7),(8,12),(9,13)]
    best=max(sum(v for w,v in combo) for k in range(6) for combo in combinations(items,k) if sum(w for w,v in combo)<=15)
    assert best==int(expected['T4'])==21
    assert Fraction(1,100)*Fraction(9,10)/(Fraction(1,100)*Fraction(9,10)+Fraction(99,100)*Fraction(5,100))==Fraction(expected['T5'])
    a=[3,4,-1,0,6,2,3,7,5,8,9,1]
    best=max(len(seq) for n in range(len(a)+1) for seq in combinations(a,n) if all(x<y for x,y in zip(seq,seq[1:])))
    assert best==int(expected['T6'])==7
    assert [Fraction(81,87)>Fraction(234,270),Fraction(192,263)>Fraction(55,80),Fraction(273,350)>Fraction(289,350)]==[True,True,False]
    assert expected['T7']=='A,A,B' and expected['T8']=='14'


def test_full_run_keeps_64_results_and_80_calls(state):
    path=run_pilot(state['api'],state['root']/'pilots')
    state['calls'].clear()
    directory=run_experiment(state['api'],state['root']/'results',pilot_path=path,budget_rub=10)
    data=json.loads((directory/'results.json').read_text())
    assert len(data['records'])==64 and len(state['calls'])==80 and data['config']['status']=='complete'
    assert len(list((directory/'raw').glob('*.json')))==64
    summary=summarize(data['records'])
    assert len(summary)==8 and all(row['tasks']==8 for row in summary)
    assert all(row['metrics']['calls']==(16 if row['method']=='metaprompt' else 8) for row in summary)
    assert (directory/'summary.csv').exists() and (directory/'report.md').exists()
    assert 'reasoning' in (directory/'report.md').read_text()
    assert sum(row['metrics']['cost_rub'] for row in data['records'])==pytest.approx(.8)


def test_budget_guard_stops_before_any_generation(state):
    path=run_pilot(state['api'],state['root']/'pilots')
    state['calls'].clear()
    directory=run_experiment(state['api'],state['root']/'results',pilot_path=path,budget_rub=.001)
    data=json.loads((directory/'results.json').read_text())
    assert data['config']['status']=='budget_stopped' and not state['calls']


def test_pilot_model_or_config_mismatch_blocks_run(state):
    path=run_pilot(state['api'],state['root']/'pilots')
    data=json.loads(path.read_text())
    data['config']['model']='different/model'
    path.write_text(json.dumps(data))
    before=len(state['calls'])
    with pytest.raises(ApiError): run_experiment(state['api'],state['root']/'results',pilot_path=path,budget_rub=10)
    assert len(state['calls'])==before


@pytest.mark.parametrize('fault',['timeout','http','length'])
def test_failed_chat_does_not_mutate_history(state,fault):
    path=state['root']/'chat.json'
    chat=Conversation.create(path,MODEL,provider=PROVIDER,mode='max')
    original=path.read_bytes()
    state['fault']=fault
    with pytest.raises((ApiError,HistoryError)): chat.ask('hi',state['api'])
    assert path.read_bytes()==original and chat.messages==[]


@pytest.mark.parametrize('text',['broken','{}','[]'])
def test_corrupt_history_not_overwritten(tmp_path,text):
    path=tmp_path/'chat.json'; path.write_text(text)
    with pytest.raises(HistoryError): Conversation.load(path)
    assert path.read_text()==text


def test_cli_chat_resume_and_commands(state):
    def cli(args,lines):
        iterator=iter(lines); out=io.StringIO(); err=io.StringIO()
        code=main(args,root=state['root'],env={'POLZA_API_KEY':SECRET,'LLM_MODEL':MODEL},
                  client=state['api'].client,read=lambda prompt:next(iterator),stdout=out,stderr=err)
        assert code==0 and not err.getvalue()
        return out.getvalue()
    cli(['chat','--session','demo','--reasoning','max'],['hi','/exit'])
    out=cli(['chat','--session','demo'],['again','/history','/new','/exit'])
    assert state['calls'][-1]['reasoning']==MODES['max'] and len(state['calls'][-1]['messages'])==3
    assert 'user' in out and len(list((state['root']/'history').glob('*.json')))==2


@pytest.mark.parametrize('args',[[],['chat','--session','../bad'],['chat','--max-tokens','4096'],
    ['chat','--read-timeout','nan'],['chat','--reasoning','unlimited'],['experiment']])
def test_cli_bad_arguments_no_calls(state,args):
    err=io.StringIO()
    code=main(args,root=state['root'],env={'POLZA_API_KEY':SECRET,'LLM_MODEL':MODEL},
              client=state['api'].client,stdout=io.StringIO(),stderr=err)
    assert code==2 and 'Ошибка' in err.getvalue() and not state['calls']


def test_cli_help_logs_no_api(state):
    out=io.StringIO()
    assert main(['--help'],root=state['root'],env={},stdout=out,stderr=io.StringIO())==0
    assert 'usage:' in out.getvalue() and not state['calls']


def test_journal_unique_and_secret_removed(state):
    with Journal(state['root']/'logs',SECRET) as other:
        other.record('INPUT',SECRET)
        assert other.path!=state['journal'].path and SECRET not in other.path.read_text()


def test_empty_truncated_answer_preserves_billed_reasoning(state):
    def respond(request):
        body=completion(json.loads(request.content),reason='length')
        body['choices'][0]['message']['content']=None
        return httpx.Response(200,json=body)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        api=ApiClient(SECRET,MODEL,state['journal'],client,provider=PROVIDER)
        result=api.send([{'role':'user','content':'hi'}],{'reasoning':MODES['max']})
    assert not result['complete'] and result['finish_reason']=='length'
    assert result['cost_rub']==.01 and result['reasoning_tokens']==12


def test_second_stage_error_keeps_first_stage_cost(state,monkeypatch):
    original=state['api'].send
    def send(*args,**kwargs):
        if state['calls']: state['fault']='timeout'
        return original(*args,**kwargs)
    monkeypatch.setattr(state['api'],'send',send)
    checkpoints=[]
    row=solve_one(state['api'],TASKS[-1],'metaprompt','max',checkpoint=checkpoints.append)
    assert row['failure']=='api_error' and len(state['calls'])==2
    assert row['metrics']['cost_rub']==.01 and row['metrics']['unknown_costs']==1
    assert len(checkpoints[0]['stages'])==1


def test_interrupted_second_stage_keeps_checkpoint(state,monkeypatch):
    original=state['api'].send
    def send(*args,**kwargs):
        if state['calls']: raise KeyboardInterrupt
        return original(*args,**kwargs)
    monkeypatch.setattr(state['api'],'send',send)
    checkpoints=[]
    with pytest.raises(KeyboardInterrupt):
        solve_one(state['api'],TASKS[-1],'metaprompt','max',checkpoint=checkpoints.append)
    assert len(checkpoints[-1]['stages'])==1 and checkpoints[-1]['metrics']['cost_rub']==.01


def test_history_write_failure_preserves_previous_turn(state,monkeypatch):
    import day3.history as history
    path=state['root']/'chat.json'
    chat=Conversation.create(path,MODEL,provider=PROVIDER,mode='max')
    chat.ask('first',state['api']); snapshot=path.read_bytes()
    def fail(*args): raise OSError('disk full')
    monkeypatch.setattr(history.os,'replace',fail)
    with pytest.raises(HistoryError): chat.ask('second',state['api'])
    assert path.read_bytes()==snapshot and len(chat.messages)==2


def test_minimal_reasoning_preserves_nonzero_counter_and_label(state,monkeypatch):
    original=state['api'].send
    def send(*args,**kwargs):
        result=original(*args,**kwargs)
        if result['request']['reasoning']==MODES['min']:
            result['reasoning_tokens']=2
        return result
    monkeypatch.setattr(state['api'],'send',send)
    path=run_pilot(state['api'],state['root']/'pilots')
    data=json.loads(path.read_text())
    assert data['verified'] and not data['checks']['min']
    assert data['records'][0]['mode']=='min' and 'off' not in data['config']['modes']
    assert data['records'][0]['metrics']['reasoning_tokens']==2


def test_schedule_oracle_is_minimal_not_just_a_feasible_answer():
    durations=[3,5,4,6,4,3]; deps=[[],[],[0],[0],[1,2],[3,4]]
    def feasible(horizon):
        starts=[]; occupied=[0]*horizon
        def search(i):
            if i==6:return True
            earliest=max((starts[j]+durations[j] for j in deps[i]),default=0)
            for start in range(earliest,horizon-durations[i]+1):
                interval=range(start,start+durations[i])
                if any(occupied[t]>=2 for t in interval):continue
                starts.append(start)
                for t in interval:occupied[t]+=1
                if search(i+1):return True
                for t in interval:occupied[t]-=1
                starts.pop()
            return False
        return search(0)
    assert not feasible(13) and feasible(14)
    assert TASKS[-1]['expected']=='14'


def test_interruption_in_full_run_includes_partial_method_cost(state,monkeypatch):
    path=run_pilot(state['api'],state['root']/'pilots'); state['calls'].clear()
    import day3.experiment as experiment
    monkeypatch.setattr(experiment.random.Random,'shuffle',lambda self,work:None)
    original=state['api'].send
    def send(*args,**kwargs):
        if len(state['calls'])==5: raise KeyboardInterrupt
        return original(*args,**kwargs)
    monkeypatch.setattr(state['api'],'send',send)
    with pytest.raises(KeyboardInterrupt):
        run_experiment(state['api'],state['root']/'results',pilot_path=path,budget_rub=10)
    result=next((state['root']/'results').glob('*/results.json'))
    data=json.loads(result.read_text())
    assert data['config']['status']=='interrupted'
    assert len(data['records'])==5
    assert sum(r['metrics']['cost_rub'] for r in data['records'])==pytest.approx(.05)


def test_other_model_needs_its_own_verified_budget_profile(state,monkeypatch):
    state['api'].model='another/model'
    data=catalog(); data['id']='another/model'
    monkeypatch.setattr(state['api'],'catalog',lambda:data)
    with pytest.raises(ApiError):preflight(state['api'])
    assert not state['calls']


def test_equal_thinking_counts_do_not_confirm_comparison(state,monkeypatch):
    original=state['api'].send
    def send(*args,**kwargs):
        row=original(*args,**kwargs); row['reasoning_tokens']=2
        return row
    monkeypatch.setattr(state['api'],'send',send)
    path=run_pilot(state['api'],state['root']/'pilots')
    assert json.loads(path.read_text())['verified'] is False


def test_previous_off_history_remains_loadable(state):
    path=state['root']/'old-chat.json'
    Conversation.create(path,MODEL,provider=PROVIDER,mode='off')
    assert Conversation.load(path).data['parameters']['reasoning']=={'enabled':False}


def test_minimal_chat_has_explicit_budget_and_restores_it(state):
    path=state['root']/'min-chat.json'
    chat=Conversation.create(path,MODEL,provider=PROVIDER,mode='min')
    chat.ask('first',state['api'])
    Conversation.load(path).ask('second',state['api'])
    assert state['calls'][-1]['reasoning']=={'enabled':True,'max_tokens':512}
    assert len(state['calls'][-1]['messages'])==3


def test_paired_comparison_keeps_gains_and_losses_separate():
    from day3.experiment import comparisons
    rows=[]
    for mode in ('min','max'):
        direct=[True,False,True] if mode=='min' else [True,True,True]
        step=[True,True,False] if mode=='min' else [True,True,True]
        for method,answers in [('direct',direct),('stepwise',step)]:
            for i,answer in enumerate(answers):
                rows.append(dict(task_id=f'T{i+1}',method=method,mode=mode,correct=answer,
                    answer_correct=answer,stages=[dict(total_tokens=30,reasoning_tokens=2,cost_rub=.01,
                    prompt_tokens=10,completion_tokens=20,duration_seconds=1)],failure=None,
                    metrics=aggregate([]),parsed=None,format_valid=True,method_valid=True))
    result=comparisons(rows)
    minimum=next(r for r in result if r['mode']=='min')
    assert minimum['gains']==1 and minimum['losses']==1 and minimum['net_gain']==0
    maximum=next(r for r in result if r['mode']=='max')
    assert maximum['gains']==0 and maximum['losses']==0 and maximum['common_tasks']==3


def test_report_separates_answer_accuracy_from_method_compliance(state,monkeypatch):
    from day3.experiment import write_reports, config
    original=state['api'].send
    def send(*args,**kwargs):
        result=original(*args,**kwargs)
        result['content']=json.dumps({'answer':'14','explanation':'','experts':[]})
        return result
    monkeypatch.setattr(state['api'],'send',send)
    row=solve_one(state['api'],TASKS[-1],'experts','min')
    root=state['root']/'report'; root.mkdir()
    write_reports(root,{'config':{**config(state['api']),'status':'partial'},'records':[row]})
    text=(root/'report.md').read_text()
    assert 'Верные ответы' in text and 'Полный успех' in text and 'Нарушения метода' in text
    assert row['answer_correct'] and not row['correct']


def test_unknown_cost_is_reserved_instead_of_assumed_zero(state,monkeypatch):
    path=run_pilot(state['api'],state['root']/'pilots'); state['calls'].clear()
    original=state['api'].send
    def send(*args,**kwargs):
        state['fault']='http' if not state['calls'] else None
        return original(*args,**kwargs)
    monkeypatch.setattr(state['api'],'send',send)
    directory=run_experiment(state['api'],state['root']/'results',pilot_path=path,budget_rub=10)
    data=json.loads((directory/'results.json').read_text())
    assert data['config']['status']=='complete' and len(data['records'])==64
    assert len(state['calls'])==80
    budget=data['config']['budget_accounting']
    assert budget['unknown_reserve_rub']>0 and budget['accounted_rub']<=10
    assert sum(r['metrics']['unknown_costs'] for r in data['records'])==1


def test_resume_skips_all_finished_attempts_including_network_failure(state,monkeypatch):
    path=run_pilot(state['api'],state['root']/'pilots'); state['calls'].clear()
    original=state['api'].send
    def send(*args,**kwargs):
        if len(state['calls'])>=4:raise KeyboardInterrupt
        state['fault']='http' if len(state['calls'])==3 else None
        return original(*args,**kwargs)
    monkeypatch.setattr(state['api'],'send',send)
    with pytest.raises(KeyboardInterrupt):
        run_experiment(state['api'],state['root']/'results',pilot_path=path,budget_rub=10)
    folder=next((state['root']/'results').iterdir())
    before=json.loads((folder/'results.json').read_text())['records']
    assert len(before)==4 and before[-1]['failure']=='api_error'
    monkeypatch.setattr(state['api'],'send',original); state['fault']=None
    result=run_experiment(state['api'],state['root']/'results',pilot_path=path,budget_rub=10,resume_dir=folder)
    after=json.loads((result/'results.json').read_text())
    assert after['config']['status']=='complete' and len(after['records'])==64
    assert after['records'][:4]==before and len(state['calls'])==80
    assert len({(r['task_id'],r['method'],r['mode']) for r in after['records']})==64


def test_resume_cannot_silently_change_budget(state):
    path=run_pilot(state['api'],state['root']/'pilots'); state['calls'].clear()
    folder=run_experiment(state['api'],state['root']/'results',pilot_path=path,budget_rub=.001)
    assert not state['calls']
    with pytest.raises(ApiError):
        run_experiment(state['api'],state['root']/'results',pilot_path=path,budget_rub=10,resume_dir=folder)
    assert not state['calls']


def test_reasoning_comparison_does_not_blame_prompt_for_network_error(state):
    from day3.experiment import comparisons
    direct=solve_one(state['api'],TASKS[-1],'direct','min')
    state['fault']='http'
    experts=solve_one(state['api'],TASKS[-1],'experts','min')
    row=comparisons([direct,experts])[0]
    assert row['common_tasks']==0 and row['excluded_pairs']==1 and row['losses']==0


def test_interaction_uses_same_tasks_for_both_reasoning_levels(state,monkeypatch):
    from day3.experiment import interactions
    rows=[]
    for mode in MODES:
        for method in ('direct','stepwise'):
            rows.append(solve_one(state['api'],TASKS[-1],method,mode))
    result=interactions(rows)[0]
    assert result['common_tasks']==1 and result['gain_min']==result['gain_max']==0
    rows[-1]['format_valid']=False
    result=interactions(rows)[0]
    assert result['common_tasks']==0 and result['excluded_tasks']==1


def test_network_failure_does_not_reduce_accuracy_of_received_answers(state):
    good=solve_one(state['api'],TASKS[-1],'direct','max')
    state['fault']='http'
    failed=solve_one(state['api'],TASKS[0],'direct','max')
    row=summarize([good,failed])[0]
    assert row['tasks']==2 and row['answer_correct']==1 and row['answers_received']==1
    assert row['wrong_answers']==0 and row['api_errors']==1
