import io
import json
from pathlib import Path

import httpx
import pytest
from pytest_bdd import given,when,then,scenarios

from day4.core import ApiClient,ApiError,Journal
from day4.history import Conversation,HistoryError
from day4.experiment import (TASKS,TEMPERATURES,REASONING,assess,normalize_name,
    summarize,run_experiment,preflight,zero_findings)
from day4.cli import main

scenarios('features/day4.feature')
MODEL='google/gemini-2.5-flash-lite'; PROVIDER='google-ai-studio/flex'; SECRET='private-test-key'
NAMES=['Пайтон','Код Старт','Python Quest','Пайтончик','Code Path']


def catalog():
    provider={'name':PROVIDER,'max_completion_tokens':65535,
        'supported_parameters':['temperature','reasoning','max_tokens'],
        'capabilities':{'reasoning_toggle':{'status':'supported'}},
        'pricing':{'currency':'RUB','prompt_per_million':'6','completion_per_million':'24'}}
    return {'id':MODEL,'max_completion_tokens':65535,'providers':[provider]}


def response(payload):
    user=payload['messages'][-1]['content']
    body={'answer':56} if user==TASKS['logic']['prompt'] else {'names':NAMES}
    return {'model':MODEL,'choices':[{'finish_reason':'stop','message':{'content':json.dumps(body,ensure_ascii=False)}}],
            'usage':{'prompt_tokens':20,'completion_tokens':40,'total_tokens':60,'cost_rub':.01,
                     'completion_tokens_details':{'reasoning_tokens':15}}}


@pytest.fixture
def state(tmp_path):
    journal=Journal(tmp_path/'logs',SECRET)
    value=dict(root=tmp_path,journal=journal,calls=[],fault=None)
    def respond(request):
        if request.method=='GET':return httpx.Response(200,json={'data':[catalog()]})
        payload=json.loads(request.content); value['calls'].append(payload)
        if value['fault']=='interrupt_third' and len(value['calls'])==3:raise KeyboardInterrupt
        if value['fault']=='network':raise httpx.ReadError(SECRET,request=request)
        if value['fault']=='http':return httpx.Response(503,text=SECRET)
        if value['fault']=='fatal':return httpx.Response(401,text=SECRET)
        body=response(payload)
        if value['fault']=='length':body['choices'][0]['finish_reason']='length'
        if value['fault']=='markdown':body['choices'][0]['message']['content']='```json\n{"answer":56}\n```'
        return httpx.Response(200,json=body)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        value['api']=ApiClient(SECRET,MODEL,journal,client,provider=PROVIDER)
        yield value
    journal.close()


@given('модель поддерживает температуру и фиксированный reasoning')
def ready(state):pass


@when('каждый из двух запросов выполнен пять раз при трёх температурах')
def run(state):
    state['result']=run_experiment(state['api'],state['root']/'results',repeats=5,budget_rub=5)


@then('сохранены тридцать результатов и шесть строк сводки')
def saved(state):
    import csv
    path=state['result']; data=json.loads((path/'results.json').read_text())
    assert data['config']['status']=='complete' and len(data['records'])==30
    assert len(list((path/'raw').glob('*.json')))==30
    with (path/'summary.csv').open(encoding='utf-8-sig') as f:assert len(list(csv.DictReader(f)))==6
    with (path/'details.csv').open(encoding='utf-8-sig') as f:assert len(list(csv.DictReader(f)))==30
    assert 'Выводы' in (path/'report.md').read_text() and 'temperature = 0' in (path/'report.md').read_text()
    assert sum(r['cost_rub'] for r in data['records'])==pytest.approx(.3)


@then('между прогонами меняется только температура')
def isolated(state):
    assert len(state['calls'])==30
    for task in TASKS:
        calls=[p for p in state['calls'] if p['messages'][-1]['content']==TASKS[task]['prompt']]
        assert len(calls)==15
        base={k:v for k,v in calls[0].items() if k!='temperature'}
        for p in calls:
            assert {k:v for k,v in p.items() if k!='temperature'}==base
            assert len(p['messages'])==2 and not any(m['role']=='assistant' for m in p['messages'])
            assert p['reasoning']==REASONING and p['max_tokens']==4096
            assert not {'response_format','stop','seed','top_p'} & p.keys()
        assert {p['temperature'] for p in calls}=={0,.7,1.2}
        assert all(sum(p['temperature']==t for p in calls)==5 for t in TEMPERATURES)


def row(task,text,temperature=0,**kwargs):
    return dict(task_id=task,temperature=temperature,content=text,received=True,complete=True,
        api_error=None,finish_reason='stop',http_status=200,total_tokens=60,prompt_tokens=20,
        completion_tokens=40,reasoning_tokens=15,cost_rub=.01,duration_seconds=1,
        **assess(task,text),**kwargs)


@given('два ответа содержат те же названия с другим порядком и регистром')
def same_names(state):
    state['rows']=[row('names',json.dumps({'names':NAMES},ensure_ascii=False)),
        row('names',json.dumps({'names':[s.upper() for s in reversed(NAMES)]},ensure_ascii=False))]


@when('вычислено разнообразие названий')
def diversity(state):state['summary']=summarize(state['rows'])[0]


@then('исходные тексты различны, а нормализованный набор один')
def not_new(state):
    s=state['summary']
    assert s['unique_raw']==2 and s['unique_semantic']==1
    assert s['unique_names']==5 and s['total_name_slots']==10
    assert s['mean_jaccard']==1 and s['mean_jaccard_distance']==0


@when('разговор с температурой ноль сохранён и восстановлен')
def conversation(state):
    path=state['root']/'chat.json'
    chat=Conversation.create(path,MODEL,{'temperature':0},provider=PROVIDER,mode='min')
    chat.ask('hi',state['api']); Conversation.load(path).ask('remember?',state['api'])


@then('история передаётся модели с температурой ноль')
def zero_history(state):
    assert state['calls'][-1]['temperature']==0 and len(state['calls'][-1]['messages'])==3


@pytest.mark.parametrize('text,json_valid,schema_valid,correct',[
    ('{"answer":56}',True,True,True),('{"answer":55}',True,True,False),
    ('{"answer":56.0}',True,True,True),('{"answer":"56"}',True,False,None),
    ('{"answer":true}',True,False,None),('```json\n{"answer":56}\n```',False,False,None),
    ('{"answer":NaN}',False,False,None),('{"answer":56,"answer":56}',False,False,None),
    ('текст',False,False,None),('{"answer":56,"extra":1}',True,False,None)])
def test_math_correctness_is_not_confused_with_format(text,json_valid,schema_valid,correct):
    result=assess('logic',text)
    assert result['json_valid'] is json_valid and result['schema_valid'] is schema_valid
    assert result['answer_correct'] is correct


@pytest.mark.parametrize('names,valid',[(NAMES,True),(NAMES[:4],False),
    (['Py','py','A','B','C'],False),(['A',' B ','C','D',''],False),
    (['Ａ','A','B','C','D'],False)])
def test_generation_checks_five_nonempty_normalized_distinct_names(names,valid):
    result=assess('names',json.dumps({'names':names},ensure_ascii=False))
    assert result['schema_valid'] and result['requirements_met'] is valid
    assert result['answer_correct'] is None


def test_normalization_unicode_case_and_spaces():
    assert normalize_name('  Ｐｙｔｈｏｎ\t QUEST ')=='python quest'
    assert normalize_name('PyQuest')!=normalize_name('Py Quest')


def test_maximum_name_diversity():
    rows=[row('names',json.dumps({'names':[f'Name-{i}-{j}' for j in range(5)]})) for i in range(5)]
    s=summarize(rows)[0]
    assert s['unique_names']==s['total_name_slots']==25
    assert s['name_uniqueness_ratio']==1 and s['mean_jaccard']==0


def test_one_valid_name_answer_is_not_a_five_run_diversity_estimate():
    s=summarize([row('names',json.dumps({'names':NAMES}))])[0]
    assert s['valid_name_responses']==1 and s['mean_jaccard'] is None


def test_formatting_changes_do_not_count_as_changed_math_answer():
    s=summarize([row('logic','{"answer":56}'),row('logic','{ "answer": 56.0 }')])[0]
    assert s['unique_raw']==2 and s['unique_semantic']==1


def test_zero_findings_describe_observation_without_universal_guarantee():
    same=[row('logic','{"answer":56}') for _ in range(5)]
    text=zero_findings(summarize(same))
    assert 'одинаков' in text and 'гарант' in text
    text=zero_findings(summarize(same[:4]+[row('logic','{"answer":55}')]))
    assert 'различ' in text


@pytest.mark.parametrize('error',[httpx.ConnectError,httpx.ReadError,httpx.WriteError,httpx.ConnectTimeout,
    httpx.ReadTimeout,httpx.WriteTimeout,httpx.PoolTimeout,httpx.RemoteProtocolError])
def test_network_failures_safe_and_not_retried(state,error):
    calls=[]
    def fail(request):calls.append(request); raise error(SECRET,request=request)
    with httpx.Client(transport=httpx.MockTransport(fail)) as client:
        with pytest.raises(ApiError) as exc:
            ApiClient(SECRET,MODEL,state['journal'],client,provider=PROVIDER).send([{'role':'user','content':'hi'}])
    assert len(calls)==1 and SECRET not in str(exc.value) and SECRET not in state['journal'].path.read_text()


@pytest.mark.parametrize('status',[301,400,401,402,403,404,408,413,422,429,500,502,503,504])
def test_http_error_not_success(state,status):
    with httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(status,text=SECRET))) as client:
        with pytest.raises(ApiError):
            ApiClient(SECRET,MODEL,state['journal'],client,provider=PROVIDER).send([{'role':'user','content':'hi'}])
    assert SECRET not in state['journal'].path.read_text()


@pytest.mark.parametrize('missing',['temperature','reasoning','max_tokens'])
def test_unsupported_parameters_stop_before_generation(state,monkeypatch,missing):
    data=catalog(); data['providers'][0]['supported_parameters'].remove(missing)
    monkeypatch.setattr(state['api'],'catalog',lambda:data)
    with pytest.raises(ApiError):preflight(state['api'])
    assert not state['calls']


def test_fatal_api_error_stops_and_keeps_partial_report(state):
    state['fault']='fatal'
    directory=run_experiment(state['api'],state['root']/'results',repeats=5,budget_rub=5)
    data=json.loads((directory/'results.json').read_text())
    assert data['config']['status']=='request_rejected' and len(state['calls'])==1
    assert len(data['records'])==1 and data['records'][0]['answer_correct'] is None


def test_unknown_cost_is_reserved_and_not_zero(state):
    state['fault']='network'
    directory=run_experiment(state['api'],state['root']/'results',repeats=1,budget_rub=5)
    data=json.loads((directory/'results.json').read_text())
    assert len(state['calls'])==6 and data['config']['status']=='complete'
    assert data['config']['budget_accounting']['unknown_reserve_rub']>0
    assert all(r['cost_rub'] is None for r in data['records'])
    assert all(r['answer_correct'] is None for r in data['records'])


def test_budget_prevents_call_before_overspending(state):
    directory=run_experiment(state['api'],state['root']/'results',repeats=5,budget_rub=.001)
    assert not state['calls']
    assert json.loads((directory/'results.json').read_text())['config']['status']=='budget_stopped'


def test_resume_does_not_repeat_completed_attempts(state,monkeypatch):
    original=state['api'].send
    def send(*args,**kwargs):
        if len(state['calls'])==2:raise KeyboardInterrupt
        return original(*args,**kwargs)
    monkeypatch.setattr(state['api'],'send',send)
    with pytest.raises(KeyboardInterrupt):run_experiment(state['api'],state['root']/'results',repeats=5,budget_rub=5)
    path=next((state['root']/'results').iterdir())
    before=json.loads((path/'results.json').read_text())['records']
    monkeypatch.setattr(state['api'],'send',original)
    run_experiment(state['api'],state['root']/'results',repeats=5,budget_rub=5,resume_dir=path)
    after=json.loads((path/'results.json').read_text())
    assert len(state['calls'])==30 and len(after['records'])==30 and after['records'][:2]==before


@pytest.mark.parametrize('fault',['network','http','length'])
def test_failed_chat_keeps_history(state,fault):
    path=state['root']/'chat.json'; chat=Conversation.create(path,MODEL,provider=PROVIDER)
    original=path.read_bytes(); state['fault']=fault
    with pytest.raises((ApiError,HistoryError)):chat.ask('hi',state['api'])
    assert path.read_bytes()==original and not chat.messages


def test_cli_restores_zero_temperature(state):
    def cli(args,inputs):
        values=iter(inputs); out=io.StringIO(); err=io.StringIO()
        code=main(args,root=state['root'],env={'POLZA_API_KEY':SECRET,'LLM_MODEL':MODEL},
            client=state['api'].client,read=lambda p:next(values),stdout=out,stderr=err)
        assert code==0 and not err.getvalue()
    cli(['chat','--session','demo','--temperature','0'],['first','/exit'])
    cli(['chat','--session','demo'],['second','/history','/new','/exit'])
    assert state['calls'][-1]['temperature']==0 and len(state['calls'][-1]['messages'])==3


@pytest.mark.parametrize('args',[[],['chat','--temperature','nan'],['chat','--temperature','-1'],
    ['chat','--temperature','2.1'],['experiment','--repeats','0'],['experiment','--budget-rub','-1'],
    ['chat','--session','../bad']])
def test_invalid_arguments_logged_without_calls(state,args):
    err=io.StringIO()
    code=main(args,root=state['root'],env={'POLZA_API_KEY':SECRET,'LLM_MODEL':MODEL},
        client=state['api'].client,stdout=io.StringIO(),stderr=err)
    assert code==2 and 'Ошибка' in err.getvalue() and not state['calls']


def test_help_logged_without_key(state):
    out=io.StringIO()
    assert main(['--help'],root=state['root'],env={},stdout=out,stderr=io.StringIO())==0
    assert 'usage:' in out.getvalue()


@pytest.mark.parametrize('data',['broken','{}','[]'])
def test_corrupt_history_not_overwritten(tmp_path,data):
    path=tmp_path/'bad.json'; path.write_text(data)
    with pytest.raises(HistoryError):Conversation.load(path)
    assert path.read_text()==data


def test_interrupt_during_network_is_saved_and_never_resent_on_resume(state):
    state['fault']='interrupt_third'
    with pytest.raises(KeyboardInterrupt):
        run_experiment(state['api'],state['root']/'results',repeats=5,budget_rub=5)
    path=next((state['root']/'results').iterdir())
    data=json.loads((path/'results.json').read_text())
    assert len(data['records'])==3 and data['records'][-1]['failure']=='interrupted'
    assert data['config']['budget_accounting']['unknown_reserve_rub']>0
    state['fault']=None
    run_experiment(state['api'],state['root']/'results',repeats=5,budget_rub=5,resume_dir=path)
    assert len(state['calls'])==30


def test_model_mismatch_is_visible_and_not_in_diversity(state,monkeypatch):
    original=state['api'].send
    def send(*args,**kwargs):
        row=original(*args,**kwargs); row['actual_model']='different/model'; return row
    monkeypatch.setattr(state['api'],'send',send)
    path=run_experiment(state['api'],state['root']/'results',repeats=1,budget_rub=5)
    rows=json.loads((path/'results.json').read_text())['records']
    assert all(r['failure']=='model_mismatch' for r in rows)
    assert all(r['unique_raw']==0 and r['model_mismatches']==1 for r in summarize(rows))


def test_zero_text_distinguishes_raw_changes_from_normalized_data():
    summary=summarize([row('logic','{"answer":56}'),row('logic','{ "answer":56.0 }')])
    text=zero_findings(summary)
    assert 'различ' in text and 'нормализац' in text and '1' in text


def test_failed_network_is_not_presented_as_json_breakage(state):
    state['fault']='network'; updates=[]
    run_experiment(state['api'],state['root']/'results',repeats=1,budget_rub=5,notify=updates.append)
    assert len(updates)==6 and all('JSON=не оценено' in text and 'api_error' in text for text in updates)


def test_report_does_not_assume_higher_temperature_means_more_diversity():
    from day4.experiment import temperature_conclusions
    rows=[]
    for i in range(5):
        rows.append(row('names',json.dumps({'names':[f'Name-{i}-{j}' for j in range(5)]}),temperature=.7))
        rows.append(row('names',json.dumps({'names':NAMES}),temperature=1.2))
    text=temperature_conclusions(summarize(rows))
    assert 'не увеличило' in text and '0.7' in text and '1.2' in text
