from copy import deepcopy
import json,sqlite3
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from pytest_bdd import scenarios,given,when,then
from agent12.web import create_app,PROFILE
from agent12.store import Store,StoreError
from agent12.agent import AgentError

scenarios('features/persistence.feature')

class Lookup:
    def __init__(self):self.calls=[]
    def send(self,messages):
        self.calls.append(deepcopy(messages))
        if messages[-1]['role']=='user':
            return {'role':'assistant','content':None,'tool_calls':[{'id':'lookup-1','type':'function',
                'function':{'name':'find_order','arguments':'{"order_id":"A-104"}'}}],
                'extra_content':{'google':{'thought_signature':'preserve-this-metadata'}}}
        return {'role':'assistant','content':'Сумма 2390 RUB.'}

class Remember:
    def __init__(self):self.calls=[]
    def send(self,messages):
        self.calls.append(deepcopy(messages))
        return {'role':'assistant','content':'Ранее найденная сумма 2390 RUB.'}

@pytest.fixture
@given('a saved conversation with an order lookup',target_fixture='saved')
def saved(tmp_path):
    model=Lookup()
    with TestClient(create_app(root=tmp_path,model=model)) as client:
        result=client.post('/api/chat',json={'message':'Найди заказ A-104'}).json()
        sid=result['session_id']
        record=client.get('/api/sessions/'+sid).json()
    return {'root':tmp_path,'sid':sid,'record':record}

@when('a new application instance continues that conversation')
def restart(saved):
    model=Remember()
    with TestClient(create_app(root=saved['root'],model=model)) as client:
        assert client.get('/api/sessions').json()[0]['id']==saved['sid']
        response=client.post('/api/chat',json={'session_id':saved['sid'],'message':'Какова ранее найденная сумма?'})
        assert response.status_code==200
        saved['after']=client.get('/api/sessions/'+saved['sid']).json()
    saved['calls']=model.calls

@then('the model receives the exact saved protocol and the order amount')
def protocol(saved):
    old=saved['record']['messages']
    assert saved['calls'][0][1:-1]==old
    assert old[1]['extra_content']['google']['thought_signature']=='preserve-this-metadata'
    assert old[2]['tool_call_id']=='lookup-1' and '2390' in old[2]['content']

@then('the browser history includes both successful turns')
def turns(saved):assert len(saved['after']['turns'])==2

def test_failed_generation_leaves_old_context_and_attempt_evidence(saved):
    class Broken:
        def send(self,messages):raise AgentError('network')
    with TestClient(create_app(root=saved['root'],model=Broken())) as client:
        assert client.post('/api/chat',json={'session_id':saved['sid'],'message':'продолжим'}).status_code==502
        state=client.get('/api/sessions/'+saved['sid']).json()
        assert state['messages']==saved['record']['messages'] and len(state['turns'])==1
        assert state['attempts'][-1]['status']=='failed'


def test_crashed_partial_tool_turn_is_archived_without_reexecution(tmp_path):
    path=tmp_path/'history.sqlite3'
    with Store(path) as store:
        sid=store.create('title',PROFILE)
        attempt=store.begin(sid,'Найди заказ')
        partial=[{'role':'user','content':'Найди заказ'},
            {'role':'assistant','content':None,'tool_calls':[{'id':'x','type':'function','function':{
                'name':'find_order','arguments':'{"order_id":"A-104"}'}}]},
            {'role':'tool','tool_call_id':'x','content':'{"amount":"2390"}'}]
        store.checkpoint(attempt,partial)
    with Store(path) as store:
        row=store.load(sid)
        assert row['messages']==[] and row['attempts'][0]['status']=='interrupted'
        assert row['attempts'][0]['messages']==partial


def test_database_lock_blocks_second_process_instance(tmp_path):
    path=tmp_path/'history.sqlite3'
    with Store(path):
        with pytest.raises(StoreError):Store(path)
    with Store(path):pass

@pytest.mark.parametrize('data',[b'not a database',b''])
def test_corrupt_database_is_not_replaced(tmp_path,data):
    path=tmp_path/'history.sqlite3';path.write_bytes(data)
    with pytest.raises(StoreError):Store(path)
    assert path.read_bytes()==data


def test_unknown_database_version_is_not_modified(tmp_path):
    path=tmp_path/'history.sqlite3'
    with sqlite3.connect(path) as c:c.execute('PRAGMA user_version=99')
    before=path.read_bytes()
    with pytest.raises(StoreError):Store(path)
    assert path.read_bytes()==before


def test_changed_profile_blocks_api_before_spending(saved):
    path=saved['root']/'data/history.sqlite3'
    with sqlite3.connect(path) as c:
        c.execute('UPDATE sessions SET profile=?',('{}',))
    model=Remember()
    with TestClient(create_app(root=saved['root'],model=model)) as client:
        assert client.post('/api/chat',json={'session_id':saved['sid'],'message':'hi'}).status_code==409
    assert not model.calls


def test_disk_commit_failure_rolls_back_complete_history(saved,monkeypatch):
    def fail(*a,**kw):raise StoreError('disk full')
    monkeypatch.setattr(Store,'complete',fail)
    with TestClient(create_app(root=saved['root'],model=Remember())) as client:
        assert client.post('/api/chat',json={'session_id':saved['sid'],'message':'hi'}).status_code==503
        assert client.get('/api/sessions/'+saved['sid']).json()['messages']==saved['record']['messages']


def test_invalid_stored_tool_protocol_blocks_generation(saved):
    path=saved['root']/'data/history.sqlite3'
    with sqlite3.connect(path) as c:c.execute('UPDATE sessions SET messages=?',('[{"role":"tool","tool_call_id":"lost","content":"bad"}]',))
    model=Remember()
    with TestClient(create_app(root=saved['root'],model=model)) as client:
        assert client.post('/api/chat',json={'session_id':saved['sid'],'message':'hi'}).status_code==503
    assert not model.calls


def test_sessions_are_isolated(tmp_path):
    model=Remember()
    with TestClient(create_app(root=tmp_path,model=model)) as client:
        a=client.post('/api/chat',json={'message':'Меня зовут Алиса'}).json()
        b=client.post('/api/chat',json={'message':'Меня зовут Борис'}).json()
        assert a['session_id']!=b['session_id']
        assert len(model.calls[1])==2
    with TestClient(create_app(root=tmp_path,model=model)) as client:
        assert len(client.get('/api/sessions').json())==2

@pytest.mark.parametrize('bad_calls',[{},0,''])
def test_malformed_tool_calls_in_saved_context_rejected(tmp_path,bad_calls):
    path=tmp_path/'history.sqlite3'
    with Store(path) as store:
        sid=store.create('test',PROFILE)
    invalid=[{'role':'user','content':'hi'},{'role':'assistant','content':'hi','tool_calls':bad_calls}]
    with sqlite3.connect(path) as c:c.execute('UPDATE sessions SET messages=?,turns=? WHERE id=?',
        (json.dumps(invalid),'[{}]',sid))
    with Store(path) as store:
        with pytest.raises(StoreError):store.load(sid)


def test_checkpoint_failure_preserves_old_successful_context(saved,monkeypatch):
    def fail(*a,**kw):raise StoreError('disk full')
    monkeypatch.setattr(Store,'checkpoint',fail)
    with TestClient(create_app(root=saved['root'],model=Lookup())) as client:
        response=client.post('/api/chat',json={'session_id':saved['sid'],'message':'Найди заказ'})
        assert response.status_code==503
        assert client.get('/api/sessions/'+saved['sid']).json()['messages']==saved['record']['messages']


def test_schema_permissions_and_complete_commit(tmp_path):
    path=tmp_path/'data/history.sqlite3'
    with Store(path) as store:
        assert path.stat().st_mode & 0o777==0o600
        sid=store.create('test',PROFILE)
        attempt=store.begin(sid,'hi')
        messages=[{'role':'user','content':'hi'},{'role':'assistant','content':'hello'}]
        store.complete(attempt,messages,{'user':'hi','answer':'hello','steps':[]})
        assert store.load(sid)['revision']==1
        assert store.load(sid)['attempts'][0]['status']=='complete'
    with Store(path) as store:assert store.load(sid)['messages']==messages

def test_api_key_is_redacted_in_database_and_history(tmp_path,monkeypatch):
    secret='unique-api-key-secret-value'
    monkeypatch.setenv('POLZA_API_KEY',secret)
    with TestClient(create_app(root=tmp_path,model=Remember())) as client:
        sid=client.post('/api/chat',json={'message':'Привет '+secret}).json()['session_id']
        row=client.get('/api/sessions/'+sid).json()
        assert secret not in json.dumps(row)
    assert secret.encode() not in (tmp_path/'data/history.sqlite3').read_bytes()


def test_new_session_uses_persisted_context_after_several_restarts(tmp_path):
    sid=''
    for count in range(3):
        model=Remember()
        with TestClient(create_app(root=tmp_path,model=model)) as client:
            response=client.post('/api/chat',json={'session_id':sid,'message':f'message {count}'})
            assert response.status_code==200
            sid=response.json()['session_id']
            assert len(model.calls[0])==2+2*count
            assert len(client.get('/api/sessions/'+sid).json()['turns'])==count+1

def test_invalid_completed_protocol_has_storage_error_and_no_commit(tmp_path):
    with Store(tmp_path/'history.sqlite3') as store:
        sid=store.create('test',PROFILE)
        attempt=store.begin(sid,'hi')
        with pytest.raises(StoreError):store.complete(attempt,[{'role':'tool','content':'lost'}],{})
        assert store.load(sid)['messages']==[]
