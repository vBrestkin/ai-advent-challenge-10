from copy import deepcopy
import json,sqlite3
import httpx,pytest
from fastapi.testclient import TestClient
from agent10.metrics import TokenCounter,normalize_usage,aggregate,Budget,BudgetStop
from agent10.transport import Client,MODEL,PROVIDER,preflight
from agent10.log import Journal
from agent10.web import create_app
from agent10.agent import AgentError
from agent10.store import Store,StoreError


def response(content='hello',reason='stop',calls=None,usage=True):
    message={'role':'assistant','content':content}
    if calls:message['tool_calls']=calls
    result={'model':MODEL,'choices':[{'finish_reason':reason,'message':message}]}
    if usage:result['usage']={'prompt_tokens':100,'completion_tokens':20,'total_tokens':120,
        'completion_tokens_details':{'reasoning_tokens':10},'cost_rub':.01}
    return result


def test_counter_separates_text_from_full_request():
    counter=TokenCounter()
    assert counter.count('hello')>0
    assert counter.estimate([{'role':'user','content':'hello'}],[])>counter.count('hello')
    assert counter.estimate([{'role':'user','content':'hello'}],[{'function':{'name':'calc'}}])>counter.estimate([{'role':'user','content':'hello'}],[])
    assert 'estimate' in counter.method

@pytest.mark.parametrize('usage',[None,{}, {'prompt_tokens':True,'total_tokens':-1,'cost_rub':float('nan')}])
def test_unknown_metrics_are_not_zero(usage):
    row=normalize_usage(usage)
    assert row['total_tokens'] is None and row['cost_rub'] is None


def test_reasoning_is_not_added_to_total():
    row=normalize_usage(response()['usage'])
    summary=aggregate([row,row])
    assert summary['total_tokens']==240 and summary['reasoning_tokens']==20
    assert summary['cost_rub']==pytest.approx(.02)


def test_partial_unknown_sums_have_counters():
    rows=[normalize_usage(response()['usage']),normalize_usage(None)]
    summary=aggregate(rows)
    assert summary['total_tokens']==120 and summary['unknown_total_tokens']==1
    assert summary['unknown_costs']==1 and summary['cost_rub']==.01


def test_inconsistent_usage_is_flagged():
    result=normalize_usage({'prompt_tokens':100,'completion_tokens':20,'total_tokens':999,
        'completion_tokens_details':{'reasoning_tokens':50}})
    assert result['usage_warning']


def test_budget_reserves_uncertain_charge_and_stops():
    budget=Budget(.1)
    budget.before(.08)
    budget.after(None,.08)
    assert budget.accounting()['unknown_reserve_rub']==.08
    with pytest.raises(BudgetStop):budget.before(.03)


def test_network_failure_retains_metrics_record_and_no_retry(tmp_path):
    count=[]
    def respond(request):count.append(request);raise httpx.ReadTimeout('secret',request=request)
    with Journal(tmp_path/'logs','secret') as log,httpx.Client(transport=httpx.MockTransport(respond)) as http:
        client=Client('secret',log,http)
        with pytest.raises(AgentError):client.send([{'role':'user','content':'hi'}])
    assert len(count)==1 and len(client.records)==1
    assert client.records[0]['status']=='api_error' and client.records[0]['cost_rub'] is None
    assert client.records[0]['budget_reserve_rub']>0


def test_context_guard_prevents_normal_request_but_lab_can_send(tmp_path):
    sent=[]
    def respond(request):sent.append(request);return httpx.Response(400,json={'error':{'message':'maximum context length exceeded'}})
    with Journal(tmp_path/'logs') as log,httpx.Client(transport=httpx.MockTransport(respond)) as http:
        client=Client('k',log,http);client.context_limit=20
        with pytest.raises(AgentError):client.send([{'role':'user','content':'hello '*200}])
        assert not sent and client.records[0]['status']=='local_context_limit'
        client.allow_overflow=True
        with pytest.raises(AgentError):client.send([{'role':'user','content':'hello '*200}])
        assert len(sent)==1 and client.records[-1]['http_status']==400
        assert 'maximum context' in client.records[-1]['response']


def test_web_persists_all_tool_round_metrics_across_restart(tmp_path):
    calls=[]
    tool={'id':'tool-1','type':'function','function':{'name':'find_order','arguments':'{"order_id":"A-104"}'}}
    def respond(request):
        calls.append(json.loads(request.content))
        body=response(None,'tool_calls',[tool]) if len(calls)==1 else response('Сумма 2390 RUB')
        return httpx.Response(200,json=body)
    with Journal(tmp_path/'transport') as log,httpx.Client(transport=httpx.MockTransport(respond)) as http:
        model=Client('k',log,http)
        with TestClient(create_app(root=tmp_path,model=model)) as app:
            result=app.post('/api/chat',json={'message':'Найди заказ A-104'}).json()
            sid=result['session_id']
            assert result['metrics']['calls']==2 and result['metrics']['total_tokens']==240
            assert result['metrics']['cost_rub']==pytest.approx(.02)
            assert result['llm_calls']==2
        with TestClient(create_app(root=tmp_path,model=model)) as app:
            history=app.get('/api/sessions/'+sid).json()
            assert history['metrics']['calls']==2 and history['metrics']['total_tokens']==240
            assert len(history['calls'])==2
            estimate=app.post('/api/estimate',json={'session_id':sid,'message':'А сколько было?'}).json()
            assert estimate['history_tokens_estimate']>0 and estimate['new_message_tokens_local']>0
            assert len(calls)==2


def test_http_error_is_saved_without_poisoning_successful_history(tmp_path):
    def respond(request):return httpx.Response(400,text='context length exceeded')
    with Journal(tmp_path/'transport') as log,httpx.Client(transport=httpx.MockTransport(respond)) as http:
        model=Client('k',log,http)
        with TestClient(create_app(root=tmp_path,model=model)) as app:
            assert app.post('/api/chat',json={'message':'hi'}).status_code==502
            sid=app.get('/api/sessions').json()[0]['id']
            row=app.get('/api/sessions/'+sid).json()
            assert row['messages']==[] and row['calls'][0]['http_status']==400
            assert row['metrics']['unknown_costs']==1


def test_metrics_write_failure_blocks_post_before_spending(tmp_path,monkeypatch):
    def fail(*a,**kw):raise StoreError('disk')
    monkeypatch.setattr(Store,'begin_call',fail)
    with Journal(tmp_path/'transport') as log,httpx.Client(transport=httpx.MockTransport(lambda request:pytest.fail('POST sent'))) as http:
        with TestClient(create_app(root=tmp_path,model=Client('k',log,http))) as app:
            assert app.post('/api/chat',json={'message':'hi'}).status_code==503


def test_pending_metric_survives_restart_as_unknown(tmp_path):
    from agent10.web import PROFILE
    path=tmp_path/'db.sqlite3'
    with Store(path) as store:
        sid=store.create('hi',PROFILE);attempt=store.begin(sid,'hi')
        store.begin_call(attempt,{'status':'pending','cost_rub':None,'budget_reserve_rub':.2})
    with Store(path) as store:
        row=store.load(sid)
        assert row['calls'][0]['status']=='uncertain'
        assert row['attempts'][0]['status']=='interrupted'


def test_v1_schema_migration_preserves_full_context(tmp_path):
    path=tmp_path/'db.sqlite3'
    with sqlite3.connect(path) as db:
        db.executescript("CREATE TABLE sessions(id TEXT PRIMARY KEY,title TEXT,profile TEXT,revision INTEGER,messages TEXT,turns TEXT); CREATE TABLE attempts(id TEXT PRIMARY KEY,session_id TEXT,revision INTEGER,prompt TEXT,status TEXT,messages TEXT,error TEXT,created_at TEXT); PRAGMA user_version=1;")
        db.execute('INSERT INTO sessions VALUES(?,?,?,?,?,?)',('s','title','{}',1,'[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]','[{"user":"hi","answer":"hello","steps":[]}]'))
    with Store(path) as store:
        assert store.load('s')['messages'][-1]['content']=='hello' and store.load('s')['calls']==[]
    with sqlite3.connect(path) as db:assert db.execute('PRAGMA user_version').fetchone()[0]==4


def test_preflight_uses_selected_route_not_default(tmp_path):
    provider={'name':PROVIDER,'context_length':40960,'max_completion_tokens':4096,
        'supported_parameters':['tools','tool_choice','max_tokens','temperature','reasoning','response_format','structured_outputs'],
        'capabilities':{'tools':{'status':'supported'},'json_schema_strict':{'status':'supported'}},
        'pricing':{'currency':'RUB','prompt_per_million':'14','completion_per_million':'28'}}
    body={'data':[{'id':MODEL,'context_length':999999,'providers':[provider]}]}
    with Journal(tmp_path/'log') as log,httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,json=body))) as http:
        client=Client('key',log,http);checked=preflight(client)
        assert client.context_limit==40960 and checked['provider']['name']==PROVIDER
        provider['supported_parameters'].remove('tools')
        with pytest.raises(AgentError):preflight(client)

def test_empty_history_estimate_is_zero(tmp_path):
    from test_agent import Fake
    with TestClient(create_app(root=tmp_path,model=Fake(0))) as app:
        result=app.post('/api/estimate',json={'message':'hello'}).json()
    assert result['history_tokens_estimate']==0
    assert result['request_tokens_estimate']>result['new_message_tokens_local']

def test_local_counter_uses_tokenizer_and_validates_asset_hash(tmp_path,monkeypatch):
    import hashlib
    from tokenizers import Tokenizer,models,pre_tokenizers
    tokenizer=Tokenizer(models.WordLevel({'[UNK]':0,'hello':1,'world':2},unk_token='[UNK]'))
    tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
    path=tmp_path/'tokenizer.json';tokenizer.save(str(path))
    monkeypatch.setattr('agent10.metrics.TOKENIZER_SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
    counter=TokenCounter(path)
    assert counter.exact_text and counter.count('hello world')==2
    path.write_text('{}')
    with pytest.raises(ValueError):TokenCounter(path)


def test_catalog_completion_limit_is_checked(tmp_path):
    route={'name':PROVIDER,'context_length':40960,'max_completion_tokens':1000,
        'supported_parameters':['tools','tool_choice','max_tokens','temperature','reasoning','response_format','structured_outputs'],
        'capabilities':{'tools':{'status':'supported'},'json_schema_strict':{'status':'supported'}},
        'pricing':{'currency':'RUB','prompt_per_million':'14','completion_per_million':'28'}}
    with Journal(tmp_path/'log') as log,httpx.Client(transport=httpx.MockTransport(lambda request:
        httpx.Response(200,json={'data':[{'id':MODEL,'providers':[route]}]}))) as http:
        with pytest.raises(AgentError):preflight(Client('key',log,http))

def test_estimate_accepts_empty_editor_while_chat_rejects_it(tmp_path):
    from test_agent import Fake
    with TestClient(create_app(root=tmp_path,model=Fake(0))) as app:
        result=app.post('/api/estimate',json={'message':''})
        assert result.status_code==200 and result.json()['new_message_tokens_local']==0
        assert app.post('/api/chat',json={'message':''}).status_code==422

@pytest.mark.parametrize('literal',['NaN','1e999'])
def test_nonfinite_raw_usage_is_saved_as_error_without_breaking_sqlite(tmp_path,literal):
    payload='{"model":"'+MODEL+'","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"hello"}}],"usage":{"cost_rub":'+literal+'}}'
    with Journal(tmp_path/'transport') as log,httpx.Client(transport=httpx.MockTransport(lambda request:
        httpx.Response(200,text=payload))) as http:
        with TestClient(create_app(root=tmp_path,model=Client('key',log,http))) as app:
            assert app.post('/api/chat',json={'message':'hi'}).status_code==502
            sid=app.get('/api/sessions').json()[0]['id']
            state=app.get('/api/sessions/'+sid).json()
            assert state['calls'][0]['status']=='api_error' and state['calls'][0]['cost_rub'] is None
            assert state['messages']==[]
