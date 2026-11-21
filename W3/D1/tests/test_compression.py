from copy import deepcopy
import json
import httpx
import pytest
from pytest_bdd import scenarios, given, when, then
from fastapi.testclient import TestClient
from agent11.memory import Memory, compact, context, validate_memory
from agent11.agent import Agent, AgentError
from agent11.store import Store, StoreError
from agent11.transport import Client
from agent11.log import Journal
from agent11.web import create_app, PROFILE

scenarios('features/compression.feature')

def pairs(n):
    return [m for i in range(n) for m in ({'role':'user','content':f'fact {i}'},{'role':'assistant','content':f'ack {i}'})]

class Model:
    def __init__(self): self.calls=[]; self.fail=False
    def send(self,messages,**kwargs):
        self.calls.append((deepcopy(messages),kwargs))
        if kwargs.get('purpose')=='summary':
            if self.fail: raise AgentError('summary timeout')
            return {'role':'assistant','content':'Project Маяк, Ирина; order 2390 RUB.'}
        return {'role':'assistant','content':'Принято.'}

@given('twelve complete conversation messages',target_fixture='case')
def case(): return {'archive':pairs(6),'model':Model()}

@when('summary memory is updated keeping six recent messages')
def updated(case): case['before']=deepcopy(case['archive']); case['memory']=compact(case['archive'],Memory(),case['model'])

@then('the request contains the summary and six original messages')
def request_context(case):
    memory=case['memory']; assert memory.covered==6
    sent=context(case['archive'],memory)
    assert sent[-6:]==case['archive'][-6:] and 'Маяк' in json.dumps(sent,ensure_ascii=False)
    assert case['model'].calls[0][1]=={'purpose':'summary','max_tokens':512}

@then('the complete twelve-message archive remains unchanged')
def unchanged(case): assert case['archive']==case['before']

@pytest.mark.parametrize('n',[0,1,2,3,4])
def test_below_threshold_never_spends(n):
    model=Model(); memory=Memory()
    assert compact(pairs(n),memory,model)==memory and not model.calls

def test_updates_only_new_prefix_and_previous_summary():
    model=Model(); archive=pairs(5); first=compact(archive,Memory(),model)
    assert first.covered==4
    second=compact(pairs(7),first,model)
    assert second.covered==8 and 'Project' in str(model.calls[-1][0])
    assert 'fact 0' not in str(model.calls[-1][0])
    assert 'fact 2' in str(model.calls[-1][0])


def tool_turn():
    return [{'role':'user','content':'order'}, {'role':'assistant','content':None,'tool_calls':[
        {'id':'a','type':'function','function':{'name':'find_order','arguments':'{}'}}]},
        {'role':'tool','tool_call_id':'a','content':'2390'},
        {'role':'assistant','content':None,'tool_calls':[{'id':'b','type':'function','function':{'name':'calculate','arguments':'{}'}}]},
        {'role':'tool','tool_call_id':'b','content':'2031.5'}, {'role':'assistant','content':'done'}]

def test_does_not_split_turn_with_sequential_tools():
    archive=tool_turn()+pairs(2); model=Model()
    assert compact(archive,Memory(),model).covered==0 and not model.calls
    memory=compact(archive+pairs(1),Memory(),model)
    assert memory.covered==6
    assert context(archive+pairs(1),memory)[-6:]==pairs(2)+pairs(1)

@pytest.mark.parametrize('reply',[{'content':''},{'content':'   '},{'content':None},{'content':'x'*12001},{'content':'hi','tool_calls':[{}]}])
def test_invalid_summary_never_advances_memory(reply):
    class Bad:
        def send(self,*a,**k):return reply
    memory=Memory()
    with pytest.raises(AgentError):compact(pairs(5),memory,Bad())
    assert memory.covered==0

def test_summary_failure_is_not_retried():
    model=Model();model.fail=True
    with pytest.raises(AgentError):compact(pairs(5),Memory(),model)
    assert len(model.calls)==1

def test_hash_or_boundary_corruption_rejected():
    archive=pairs(5); memory=compact(archive,Memory(),Model())
    archive[0]['content']='tampered'
    with pytest.raises(ValueError):validate_memory(archive,memory)
    with pytest.raises(ValueError):validate_memory(pairs(5),Memory(summary='bad',covered=1,digest='x'))


def test_agent_keeps_full_archive_but_sends_selected_context():
    archive=pairs(5); model=Model(); memory=compact(archive,Memory(),model)
    agent=Agent(model,context_builder=lambda messages:context(messages,memory)); agent.messages=archive
    result=agent.run('continue')
    assert result['messages'][:-2]==archive
    assert model.calls[-1][0][-7:-1]==archive[-6:]
    assert 'fact 0' not in str(model.calls[-1][0])


def test_memory_atomic_save_restore_and_no_archive_loss(tmp_path):
    path=tmp_path/'history.sqlite3'; archive=pairs(5); memory=compact(archive,Memory(),Model())
    with Store(path) as store:
        sid=store.create('title',PROFILE)
        for i in range(1,6):
            attempt=store.begin(sid,'hi')
            store.complete(attempt,archive[:i*2],{},memory=memory.to_dict() if i==5 else None)
    with Store(path) as store:
        row=store.load(sid)
        assert row['messages']==archive and row['memory']==memory.to_dict()


def test_stale_or_invalid_memory_cannot_commit(tmp_path):
    with Store(tmp_path/'history.sqlite3') as store:
        sid=store.create('title',PROFILE); attempt=store.begin(sid,'hi')
        with pytest.raises(StoreError):store.complete(attempt,pairs(1),{},memory={'summary':'bad','covered':99,'digest':'bad'})
        assert store.load(sid)['messages']==[]


def test_web_restart_summary_and_compression_failure_preserves_success(tmp_path):
    model=Model(); sid=''
    with TestClient(create_app(root=tmp_path,model=model)) as client:
        for i in range(5):
            response=client.post('/api/chat',json={'session_id':sid,'message':f'fact {i}','mode':'summary'})
            assert response.status_code==200; sid=response.json()['session_id']
        state=client.get('/api/sessions/'+sid).json()
        assert state['memory']['covered']==4 and len(state['messages'])==10
    model=Model();model.fail=True
    with TestClient(create_app(root=tmp_path,model=model)) as client:
        assert client.get('/api/sessions/'+sid).json()['memory']==state['memory']
        for i in range(2):
            response=client.post('/api/chat',json={'session_id':sid,'message':'continue','mode':'summary'})
            assert response.status_code==200
        after=client.get('/api/sessions/'+sid).json()
        assert after['memory']==state['memory'] and len(after['messages'])==14
        assert 'summary timeout' in response.json()['compression_warning']
        assert 'fact 0' not in str(model.calls[0][0])
        assert client.post('/api/chat',json={'session_id':sid,'message':'hi','mode':'bad'}).status_code==422

@pytest.mark.parametrize('fault',['timeout','disconnect','503','length','tools'])
def test_summary_http_failures_accounted_no_retries(tmp_path,fault):
    requests=[]
    def respond(request):
        payload=json.loads(request.content);requests.append(payload)
        assert 'tools' not in payload and 'tool_choice' not in payload and payload['max_tokens']==512
        if fault=='timeout':raise httpx.ReadTimeout('x',request=request)
        if fault=='disconnect':raise httpx.RemoteProtocolError('x',request=request)
        if fault=='503':return httpx.Response(503,text='unavailable')
        return httpx.Response(200,json={'model':payload['model'],'choices':[{'finish_reason':'length' if fault=='length' else 'tool_calls',
            'message':{'role':'assistant','content':'bad','tool_calls':[{}]}}]})
    with Journal(tmp_path) as log,httpx.Client(transport=httpx.MockTransport(respond)) as http:
        client=Client('key',log,http)
        with pytest.raises(AgentError):compact(pairs(5),Memory(),client)
        assert len(client.records)==1 and client.records[0]['purpose']=='summary'
        assert client.records[0]['status']=='api_error'
    assert len(requests)==1


def test_corrupted_persistent_summary_stops_before_spending(tmp_path):
    import sqlite3
    model=Model()
    with TestClient(create_app(root=tmp_path,model=model)) as client:
        sid=client.post('/api/chat',json={'message':'hello'}).json()['session_id']
    with sqlite3.connect(tmp_path/'data/history.sqlite3') as db:
        db.execute('INSERT OR REPLACE INTO memories VALUES(?,?)',(sid,json.dumps({'summary':'bad','covered':2,'digest':'wrong'})))
    model=Model()
    with TestClient(create_app(root=tmp_path,model=model)) as client:
        assert client.post('/api/chat',json={'message':'continue','session_id':sid,'mode':'summary'}).status_code==503
    assert not model.calls


def test_v2_migration_creates_empty_memory_without_changing_archive(tmp_path):
    import sqlite3
    path=tmp_path/'history.sqlite3'
    with Store(path) as store:
        sid=store.create('t',PROFILE);a=store.begin(sid,'hi');store.complete(a,pairs(1),{})
    with sqlite3.connect(path) as db:db.executescript('DROP TABLE layer_dialogues; DROP TABLE layer_tasks; DROP TABLE layer_projects; DROP TABLE layer_workspaces; DROP TABLE branches; DROP TABLE checkpoints; DROP TABLE facts_states; DROP TABLE memories; PRAGMA user_version=2;')
    with Store(path) as store:
        assert store.load(sid)['messages']==pairs(1)
        assert store.load(sid)['memory']==Memory().to_dict()


def test_http_summary_failure_keeps_answer_and_persists_unknown_cost(tmp_path):
    def provider(request):
        payload=json.loads(request.content)
        if 'tools' not in payload:return httpx.Response(503,text='provider unavailable')
        return httpx.Response(200,json={'model':payload['model'],'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':'Принято.'}}],
            'usage':{'prompt_tokens':100,'completion_tokens':5,'total_tokens':105,'cost_rub':.01}})
    with Journal(tmp_path/'transport') as journal,httpx.Client(transport=httpx.MockTransport(provider)) as http:
        model=Client('key',journal,http)
        with TestClient(create_app(root=tmp_path,model=model)) as web:
            sid=''
            for i in range(5):
                response=web.post('/api/chat',json={'message':f'fact {i}','session_id':sid,'mode':'summary'})
                assert response.status_code==200;sid=response.json()['session_id']
            state=web.get('/api/sessions/'+sid).json()
            assert len(state['messages'])==10 and state['memory']['covered']==0
            assert state['calls'][-1]['purpose']=='summary' and state['calls'][-1]['http_status']==503
            assert state['summary_metrics']['unknown_costs']==1
            assert state['summary_metrics']['unknown_reserve_rub']>0
            assert response.json()['compression_warning']


def test_ui_and_inspector_show_actual_context_with_system(tmp_path):
    from agent11.agent import SYSTEM
    model=Model()
    with TestClient(create_app(root=tmp_path,model=model)) as client:
        html=client.get('/strategies').text
        assert '<title>Стратегии контекста · День 10</title>' in html
        sid=''
        for i in range(5):sid=client.post('/api/chat',json={'message':f'fact {i}','session_id':sid,'mode':'summary'}).json()['session_id']
        calls=len(model.calls)
        compressed=client.get('/api/sessions/'+sid+'?mode=summary').json()
        full=client.get('/api/sessions/'+sid+'?mode=full').json()
        assert compressed['sent_context'][0]==full['sent_context'][0]=={'role':'system','content':SYSTEM}
        assert full['sent_context'][1:]==full['messages']
        assert len(compressed['sent_context'])<len(full['sent_context'])
        assert len(model.calls)==calls
