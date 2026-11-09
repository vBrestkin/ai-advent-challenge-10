from copy import deepcopy
import csv
import io
import json
import re

import httpx
import pytest
from pytest_bdd import scenarios, given, when, then

from day5.core import ApiClient, ApiError, Journal, RequestInterrupted, atomic_json
from day5.models import MODELS, PROVIDER, PARAMETERS, preflight
from day5.tasks import TASKS, verify_answer
from day5.experiment import run_experiment, assess, summarize, comparisons, Budget, BudgetStop
from day5.history import Conversation, HistoryError
from day5.cli import main

scenarios('features/day5.feature')
SECRET='test-secret-DO-NOT-LEAK'


def catalog(model):
    return {'id':model,'max_completion_tokens':65535,'providers':[{
        'name':PROVIDER,'max_completion_tokens':65535,
        'supported_parameters':['reasoning','temperature','max_tokens','response_format','structured_outputs'],
        'capabilities':{'json_schema_strict':{'status':'supported'}},
        'reasoning':{'mandatory':model.endswith('-pro')},
        'pricing':{'currency':'RUB','prompt_per_million':'10','completion_per_million':'20'}}]}


@pytest.fixture
def state(tmp_path):
    journal=Journal(tmp_path/'logs',SECRET)
    state={'root':tmp_path,'calls':[],'fault':None,'catalog_fault':None,'count':0}
    def respond(request):
        if request.method=='GET':
            models=[catalog(m) for m in MODELS.values()]
            if state['catalog_fault']:state['catalog_fault'](models)
            return httpx.Response(200,json={'data':models})
        payload=json.loads(request.content); state['calls'].append(payload)
        fault=state['fault']
        if callable(fault):return fault(request,payload)
        if fault=='interrupt' and len(state['calls'])==5:raise KeyboardInterrupt
        if fault=='network':raise httpx.ReadError(SECRET,request=request)
        if fault=='fatal':return httpx.Response(401,text=SECRET)
        question=payload['messages'][-1]['content']
        task=next((t for t in TASKS if t['question'] in question),TASKS[0])
        answer=task['expected']
        body={'model':payload['model'],'provider':'Google AI Studio',
              'choices':[{'message':{'content':json.dumps({'answer':answer})},'finish_reason':'stop'}],
              'usage':{'prompt_tokens':100,'completion_tokens':200,'total_tokens':300,
                       'completion_tokens_details':{'reasoning_tokens':170},'cost_rub':.01}}
        if fault=='length':body['choices'][0]['finish_reason']='length'
        if fault=='mismatch':body['model']='different'
        if fault=='no_metrics':body.pop('usage')
        return httpx.Response(200,json=body)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        state['connection']=client
        state['clients']={tier:ApiClient(SECRET,model,journal,client,provider=PROVIDER) for tier,model in MODELS.items()}
        yield state
    journal.close()


@given('три модели поддерживают общие параметры')
def ready(state):pass


@when('запущен эксперимент с тремя повторами')
def run(state):
    state['directory']=run_experiment(state['clients'],state['root']/'results',budget_rub=50)


@then('пилот содержит три проверки, а основной прогон 72 результата')
def all_saved(state):
    path=state['directory']; data=json.loads((path/'results.json').read_text())
    assert data['config']['status']=='complete'
    assert len(data['pilot'])==3 and len(data['records'])==72
    assert data['pilot_verified'] is True and len(state['calls'])==75
    assert len(list((path/'raw').glob('*.json')))==75
    assert data['budget']['known_cost_rub']==pytest.approx(.75)


@then('запрос каждой задачи одинаков для всех моделей')
def fair(state):
    for task in TASKS:
        calls=[p for p in state['calls'][3:] if task['question'] in p['messages'][-1]['content']]
        assert len(calls)==9
        base={k:v for k,v in calls[0].items() if k!='model'}
        assert all({k:v for k,v in p.items() if k!='model'}==base for p in calls)
        assert all(len(p['messages'])==2 for p in calls)
        assert all(p['reasoning']=={'enabled':True,'max_tokens':1024} and p['max_tokens']==4096 for p in calls)
        assert all(p['provider']['only']==[PROVIDER] and not p['provider']['allow_fallbacks'] for p in calls)
        assert all(p['response_format']['json_schema']['strict'] for p in calls)


@then('отчёт содержит три модели и сравнение цены с точностью')
def reports(state):
    path=state['directory']
    with (path/'summary.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    assert len(rows)==3 and all(r['correct']=='24' for r in rows)
    with (path/'details.csv').open(encoding='utf-8-sig') as f:assert len(list(csv.DictReader(f)))==72
    with (path/'tasks.csv').open(encoding='utf-8-sig') as f:assert len(list(csv.DictReader(f)))==24
    text=(path/'report.md').read_text()
    assert all(m in text for m in MODELS.values()) and 'процентных пункт' in text and 'Пилот' in text
    assert SECRET not in ''.join(p.read_text() for p in path.rglob('*') if p.is_file())


@when('активный запрос прерван и эксперимент продолжен')
def interrupt_resume(state):
    state['fault']='interrupt'
    with pytest.raises(KeyboardInterrupt):run(state)
    path=next((state['root']/'results').iterdir())
    data=json.loads((path/'results.json').read_text()); assert len(data['records'])==2
    assert data['records'][-1]['failure']=='interrupted'
    state['fault']=None
    state['directory']=run_experiment(state['clients'],state['root']/'results',budget_rub=50,resume_dir=path)


@then('прерванная попытка учтена, пилот и готовые запросы не отправлены повторно')
def no_retry(state):
    assert len(state['calls'])==75
    data=json.loads((state['directory']/'results.json').read_text())
    assert data['config']['status']=='complete' and data['records'][1]['failure']=='interrupted'
    assert data['budget']['unknown_reserve_rub']>0
    before=len(state['calls'])
    run_experiment(state['clients'],state['root']/'results',budget_rub=50,resume_dir=state['directory'])
    assert len(state['calls'])==before


@when('разговор средней модели сохранён и восстановлен')
def dialogue(state):
    path=state['root']/'history'/'demo.json'; api=state['clients']['medium']
    chat=Conversation.create(path,api.model,provider=PROVIDER)
    chat.ask('Первое сообщение',api); Conversation.load(path).ask('Второе сообщение',api)


@then('второй запрос содержит предыдущую пару ходов')
def stack(state):
    assert len(state['calls'][-1]['messages'])==3 and state['calls'][-1]['model']==MODELS['medium']
    assert all(state['calls'][-1][k]==v for k,v in PARAMETERS.items())


@pytest.mark.parametrize('error',[httpx.ConnectError,httpx.ReadError,httpx.WriteError,httpx.RemoteProtocolError,
    httpx.ConnectTimeout,httpx.ReadTimeout,httpx.WriteTimeout,httpx.PoolTimeout])
def test_network_errors_logged_redacted_without_retry(state,error):
    def fault(request,payload):raise error(SECRET,request=request)
    state['fault']=fault
    with pytest.raises(ApiError) as exc:state['clients']['small'].send([{'role':'user','content':'hi'}],PARAMETERS)
    assert len(state['calls'])==1 and exc.value.result['cost_rub'] is None
    assert SECRET not in str(exc.value)
    assert SECRET not in next((state['root']/'logs').glob('*.log')).read_text()


@pytest.mark.parametrize('status',[301,400,401,402,403,404,408,413,422,429,500,502,503,504])
def test_http_error_is_not_success(state,status):
    state['fault']=lambda request,payload:httpx.Response(status,text=SECRET)
    with pytest.raises(ApiError) as exc:state['clients']['small'].send([{'role':'user','content':'hi'}])
    assert exc.value.result['http_status']==status and len(state['calls'])==1
    assert SECRET not in json.dumps(exc.value.result)


@pytest.mark.parametrize('mutate',[
    lambda ms:ms[0]['providers'][0]['supported_parameters'].remove('reasoning'),
    lambda ms:ms[0]['providers'][0]['supported_parameters'].remove('response_format'),
    lambda ms:ms[0]['providers'][0]['capabilities']['json_schema_strict'].update(status='unknown'),
    lambda ms:ms[0]['providers'][0].update(name='other'),
    lambda ms:ms[0]['providers'][0]['pricing'].update(currency='USD'),
    lambda ms:ms[0]['providers'][0]['pricing'].update(completion_per_million='NaN'),
    lambda ms:ms[0]['providers'][0].update(max_completion_tokens=100),
])
def test_incompatible_catalog_prevents_paid_calls(state,mutate):
    state['catalog_fault']=mutate
    with pytest.raises(ApiError):run(state)
    assert not state['calls']


@pytest.mark.parametrize('fault',['network','fatal','length','mismatch','no_metrics'])
def test_pilot_failure_stops_main_and_keeps_evidence(state,fault):
    state['fault']=fault; run(state)
    data=json.loads((state['directory']/'results.json').read_text())
    assert not data['pilot_verified'] and not data['records'] and len(state['calls'])<=3
    assert data['config']['status']=='pilot_failed'
    assert (state['directory']/'report.md').exists()


@pytest.mark.parametrize('text,valid,correct',[
    ('{"answer":"56"}',True,True),('{"answer":"55"}',True,False),
    ('{"answer":56}',False,None),('{"answer":true}',False,None),
    ('```json\n{"answer":"56"}\n```',False,None),
    ('{"answer":"56","answer":"55"}',False,None),
    ('{"answer":"56","extra":1}',False,None),('NaN',False,None),('[]',False,None)])
def test_format_and_correctness_are_separate(text,valid,correct):
    data=assess(TASKS[1],text)
    assert data['format_valid'] is valid and data['answer_correct'] is correct


def metric_row(tier='small',**extra):
    return dict(tier=tier,model=MODELS[tier],task_id='T1',repeat=1,received=True,complete=True,
        format_valid=True,answer_correct=True,failure=None,api_error=None,total_tokens=300,
        prompt_tokens=100,completion_tokens=200,reasoning_tokens=170,cost_rub=.01,duration_seconds=2,**extra)


def test_missing_metrics_not_zero_and_thinking_not_double_counted():
    one=metric_row(); two=metric_row(); two.update(total_tokens=None,cost_rub=None,duration_seconds=4)
    s=summarize([one,two])[0]
    assert s['total_tokens']==300 and s['unknown_total_tokens']==1 and s['unknown_costs']==1
    assert s['duration_median_seconds']==3 and s['cost_per_correct_rub'] is None


def test_network_failure_not_counted_as_wrong_math_or_format():
    row=metric_row(); row.update(received=False,format_valid=False,answer_correct=None,
        failure='api_error',api_error='network',complete=False)
    s=summarize([metric_row(),row])[0]
    assert s['correct']==1 and s['api_errors']==1 and s['wrong_answers']==0 and s['format_errors']==0
    assert s['accuracy_attempts']==.5 and s['accuracy_received']==1


def test_comparisons_guard_unknown_costs_and_zero_denominator():
    small=metric_row(); pro=metric_row('flagship'); pro['cost_rub']=.25
    rows=summarize([small,pro]); comp=comparisons(rows)[0]
    assert comp['cost_ratio']==25 and comp['accuracy_delta_pp']==0
    pro['cost_rub']=None
    assert comparisons(summarize([small,pro]))[0]['cost_ratio'] is None


def test_matched_cost_ratio_uses_only_same_successfully_received_tasks():
    small=metric_row(); pro=metric_row('flagship'); pro['cost_rub']=.25
    lost=metric_row(); lost.update(repeat=2,cost_rub=None,received=False,answer_correct=None,failure='api_error')
    other=metric_row('flagship'); other.update(repeat=2,cost_rub=.5)
    records=[small,pro,lost,other]
    result=comparisons(summarize(records),records)[0]
    assert result['cost_ratio'] is None
    assert result['matched_cost_ratio']==25 and result['matched_cost_pairs']==1


def test_budget_stops_before_spending_and_reserves_unknown(state):
    checked=preflight(state['clients']['small']); b=Budget(50)
    amount=b.before(state['clients']['small'],[{'role':'user','content':'hi'}],PARAMETERS)
    row={'cost_rub':None}; b.after(row,amount)
    assert b.accounting()['unknown_reserve_rub']>0 and row['budget_reserve_rub']==amount
    b.restore([row]); assert b.accounting()['unknown_reserve_rub']==pytest.approx(amount)
    with pytest.raises(BudgetStop):Budget(.00001).before(state['clients']['small'],[],PARAMETERS)


def test_budget_stops_suite_without_paid_calls(state):
    path=run_experiment(state['clients'],state['root']/'results',budget_rub=.00001)
    assert not state['calls'] and json.loads((path/'results.json').read_text())['config']['status']=='budget_stopped'


def test_resume_different_config_and_corrupt_json_fail_before_calls(state):
    run(state); path=state['directory']; before=len(state['calls'])
    with pytest.raises(ApiError):run_experiment(state['clients'],state['root']/'results',repeats=2,budget_rub=50,resume_dir=path)
    (path/'results.json').write_text('broken')
    with pytest.raises(ApiError):run_experiment(state['clients'],state['root']/'results',budget_rub=50,resume_dir=path)
    assert len(state['calls'])==before


@pytest.mark.parametrize('fault',['network','length','mismatch'])
def test_failed_chat_keeps_successful_history(state,fault):
    api=state['clients']['small']; path=state['root']/'history.json'
    chat=Conversation.create(path,api.model,provider=PROVIDER); before=path.read_bytes()
    state['fault']=fault
    with pytest.raises((ApiError,HistoryError)):chat.ask('hi',api)
    assert path.read_bytes()==before


@pytest.mark.parametrize('text',['broken','{}','[]'])
def test_corrupt_history_not_overwritten(tmp_path,text):
    path=tmp_path/'history.json'; path.write_text(text)
    with pytest.raises(HistoryError):Conversation.load(path)
    assert path.read_text()==text


def test_disk_failure_happens_before_generation(state,monkeypatch):
    def fail(*a,**kw):raise OSError('disk full')
    monkeypatch.setattr('day5.experiment.atomic_json',fail)
    with pytest.raises(OSError):run(state)
    assert not state['calls']


def test_help_logged_without_key(state):
    out=io.StringIO()
    assert main(['--help'],root=state['root'],env={},stdout=out)==0
    assert 'day5' in out.getvalue().lower() or 'День 5' in out.getvalue()
    assert not state['calls']


@pytest.mark.parametrize('args',[[],['experiment','--budget-rub','nan'],['experiment','--repeats','0'],
    ['chat','--model','bad'],['chat','--session','../secret'],['catalog','--read-timeout','-1']])
def test_invalid_settings_logged_without_generation(state,args):
    out=io.StringIO()
    assert main(args,root=state['root'],env={'POLZA_API_KEY':SECRET},client=state['connection'],stderr=out)==2
    assert 'Ошибка' in out.getvalue() and not state['calls']


def test_cli_chat_inherits_key_and_model_option(state):
    inputs=iter(['hi','/exit']); out=io.StringIO()
    code=main(['chat','--model','medium','--session','demo'],root=state['root'],
        env={'POLZA_API_KEY':SECRET},client=state['connection'],read=lambda _:next(inputs),stdout=out)
    assert code==0 and state['calls'][0]['model']==MODELS['medium']
    assert (state['root']/'history'/'demo.json').exists()


def test_log_timestamp_and_key_masking(state):
    journal=state['clients']['small'].journal; journal.record('USER',SECRET)
    assert re.fullmatch(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_[a-f0-9]+\.log',journal.path.name)
    assert '[REDACTED]' in journal.path.read_text() and SECRET not in journal.path.read_text()


def test_cli_closes_journal_even_after_help(state,monkeypatch):
    journals=[]
    class CapturingJournal(Journal):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs); journals.append(self)
    monkeypatch.setattr('day5.cli.Journal',CapturingJournal)
    assert main(['--help'],root=state['root'],env={},stdout=io.StringIO())==0
    assert journals[0].file.closed


def test_resume_rejects_unknown_task_before_any_generation(state):
    run(state); path=state['directory']; before=len(state['calls'])
    data=json.loads((path/'results.json').read_text()); data['records'][0]['task_id']='T999'
    (path/'results.json').write_text(json.dumps(data))
    with pytest.raises(ApiError):run_experiment(state['clients'],state['root']/'results',budget_rub=50,resume_dir=path)
    assert len(state['calls'])==before


def test_all_completed_raw_files_match_checkpoint(state):
    run(state); path=state['directory']; data=json.loads((path/'results.json').read_text())
    for group in ('pilot','records'):
        for r in data[group]:
            raw=path/'raw'/f'{group}_{r["tier"]}_{r["task_id"]}_{r["repeat"]:02d}.json'
            assert json.loads(raw.read_text())==r


def test_resume_rejects_altered_model_in_record(state):
    run(state); path=state['directory']
    data=json.loads((path/'results.json').read_text()); data['records'][0]['model']='other'
    (path/'results.json').write_text(json.dumps(data))
    with pytest.raises(ApiError):run_experiment(state['clients'],state['root']/'results',budget_rub=50,resume_dir=path)


def test_unwritable_log_prevents_any_http_call(state,monkeypatch):
    def fail(*a,**kw):raise OSError('disk')
    monkeypatch.setattr('day5.cli.Journal',fail)
    assert main(['experiment'],root=state['root'],env={'POLZA_API_KEY':SECRET},
        client=state['connection'],stderr=io.StringIO())==1
    assert not state['calls']


def test_stale_history_detected_before_and_after_network(state):
    api=state['clients']['small']; path=state['root']/'history.json'
    chat=Conversation.create(path,api.model)
    path.write_text('{}')
    with pytest.raises(HistoryError):chat.ask('hi',api)
    assert not state['calls']


@pytest.mark.parametrize('body',[{},[],{'choices':[]},{'choices':[{'message':{'content':None},'finish_reason':'stop'}]}])
def test_invalid_api_body_saved_as_error(state,body):
    state['fault']=lambda request,payload:httpx.Response(200,json=body)
    with pytest.raises(ApiError) as exc:state['clients']['small'].send([{'role':'user','content':'hi'}])
    assert exc.value.result['response']==body and len(state['calls'])==1


@pytest.mark.parametrize('fault,expected',[
    ('network','api_error'),('length','incomplete'),('mismatch','model_mismatch'),('no_metrics',None),('fatal','api_error')])
def test_main_attempt_failure_is_preserved_and_not_resent(state,monkeypatch,fault,expected):
    for api in state['clients'].values():
        original=api.send
        def send(*a,_original=original,**kw):
            state['fault']=fault if len(state['calls'])==3 else None
            return _original(*a,**kw)
        monkeypatch.setattr(api,'send',send)
    run(state)
    data=json.loads((state['directory']/'results.json').read_text())
    assert data['records'][0]['failure']==expected
    if fault=='fatal':
        assert data['config']['status']=='request_rejected' and len(state['calls'])==4
    else:
        assert data['config']['status']=='complete' and len(state['calls'])==75
    if fault in ('network','no_metrics','fatal'):
        assert data['budget']['unknown_reserve_rub']>0


def test_history_disk_failure_keeps_previous_successful_turn(state,monkeypatch):
    api=state['clients']['small']; path=state['root']/'chat.json'
    chat=Conversation.create(path,api.model); chat.ask('first',api); snapshot=path.read_bytes()
    def fail(*a,**kw):raise OSError('disk')
    monkeypatch.setattr('day5.history.atomic_json',fail)
    with pytest.raises(HistoryError):chat.ask('second',api)
    assert path.read_bytes()==snapshot and len(chat.messages)==2 and len(state['calls'])==2


def test_failure_after_post_keeps_durable_uncertain_attempt_on_resume(state,monkeypatch):
    journal=state['clients']['small'].journal; record=journal.record
    def fail_once(event,data):
        if event=='RESPONSE' and len(state['calls'])==4:raise OSError('disk failed after POST')
        return record(event,data)
    monkeypatch.setattr(journal,'record',fail_once)
    with pytest.raises(OSError):run(state)
    path=next((state['root']/'results').iterdir()); data=json.loads((path/'results.json').read_text())
    assert data['records'][0]['failure']=='uncertain' and data['budget']['unknown_reserve_rub']>0
    monkeypatch.setattr(journal,'record',record)
    run_experiment(state['clients'],state['root']/'results',budget_rub=50,resume_dir=path)
    assert len(state['calls'])==75
    data=json.loads((path/'results.json').read_text()); assert data['records'][0]['failure']=='uncertain'
