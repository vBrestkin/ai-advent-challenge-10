from copy import deepcopy
import json
import httpx
import pytest
from pytest_bdd import scenarios,given,when,then
from fastapi.testclient import TestClient
from agent11.strategies import Facts, select_context, update_facts, validate_facts
from agent11.memory import Memory, fingerprint
from agent11.agent import AgentError
from agent11.store import Store,StoreError
from agent11.web import create_app,PROFILE
from agent11.transport import Client
from agent11.log import Journal

scenarios('features/strategies.feature')

def pairs(n):return [m for i in range(n) for m in ({'role':'user','content':f'budget {i}'},{'role':'assistant','content':'ok'})]

class Model:
    def __init__(self):self.calls=[];self.fail=False
    def send(self,messages,**kwargs):
        self.calls.append((deepcopy(messages),kwargs))
        if kwargs.get('purpose')=='facts':
            if self.fail:raise AgentError('facts timeout')
            data=json.loads(messages[-1]['content']);values=deepcopy(data['previous_facts'])
            for message in data['new_messages']:
                if message['role']=='user' and message['content'].startswith('budget '):values['budget']=message['content'].split()[-1]
            return {'role':'assistant','content':json.dumps({'facts':values})}
        if kwargs.get('purpose')=='summary':return {'role':'assistant','content':'budget recorded'}
        return {'role':'assistant','content':'ok'}


def test_sliding_window_drops_old_context_but_keeps_full_archive():
    archive=pairs(8);before=deepcopy(archive)
    assert select_context(archive,'window',Memory(),Facts())==archive[-6:]
    assert archive==before


def test_window_retains_whole_tool_turn_and_current_question():
    from test_compression import tool_turn
    archive=tool_turn()+pairs(1)+[{'role':'user','content':'next'}]
    sent=select_context(archive,'window',Memory(),Facts())
    assert sent==archive # tail would cut tool turn; retain complete turn
    archive=pairs(5)+tool_turn()+[{'role':'user','content':'next'}]
    assert select_context(archive,'window',Memory(),Facts())==tool_turn()+[archive[-1]]


def test_facts_after_every_user_including_catchup_and_tool_result():
    from test_compression import tool_turn
    archive=tool_turn()+[{'role':'user','content':'budget 210000'}];model=Model()
    state=update_facts(archive,Facts(),model)
    assert len(model.calls)==2 and state.values['budget']=='210000' and state.covered==7
    assert '2031.5' in str(model.calls[-1][0])
    assert update_facts(archive,state,model)==state and len(model.calls)==2
    selected=select_context(archive,'facts',Memory(),state)
    assert '210000' in selected[0]['content']

@pytest.mark.parametrize('reply',['not json','[]','{"facts":null}','{"facts":{"budget":210000}}','{"facts":{"x":""}}','{"facts":{},"extra":1}'])
def test_invalid_facts_are_rejected_without_advancing(reply):
    class Bad:
        def send(self,*a,**k):return {'role':'assistant','content':reply}
    original=Facts()
    with pytest.raises(AgentError):update_facts([{'role':'user','content':'hello'}],original,Bad())
    assert original==Facts()


def test_facts_digest_detects_archive_replacement():
    model=Model();archive=pairs(2);state=update_facts(archive,Facts(),model)
    archive[0]['content']='changed'
    with pytest.raises(ValueError):validate_facts(archive,state)


def test_facts_failure_prevents_answer_and_preserves_old_success(tmp_path):
    model=Model()
    with TestClient(create_app(root=tmp_path,model=model)) as client:
        sid=client.post('/api/chat',json={'message':'budget 210000','mode':'facts'}).json()['session_id']
        before=client.get('/api/sessions/'+sid).json();model.fail=True
        response=client.post('/api/chat',json={'message':'budget 999','session_id':sid,'mode':'facts'})
        assert response.status_code==502
        after=client.get('/api/sessions/'+sid).json()
        assert after['messages']==before['messages'] and after['facts_state']==before['facts_state']
        assert after['attempts'][-1]['status']=='failed'

@given('a persisted dialogue with a checkpoint',target_fixture='branches')
def branches(tmp_path):
    with Store(tmp_path/'history.sqlite3') as store:
        sid=store.create('source',PROFILE);attempt=store.begin(sid,'budget 210000')
        archive=[{'role':'user','content':'budget 210000'},{'role':'assistant','content':'ok'}]
        facts=update_facts(archive,Facts(),Model())
        store.complete(attempt,archive,{'user':'budget 210000','answer':'ok','steps':[]},facts_state=facts.to_dict())
        checkpoint=store.save_checkpoint(sid,'base')
        a=store.fork(checkpoint,'Branch A');b=store.fork(checkpoint,'Branch B')
    return {'path':tmp_path/'history.sqlite3','source':sid,'a':a,'b':b,'checkpoint':checkpoint}

@when('two branches change their own budget')
def change(branches):
    with Store(branches['path']) as store:
        for key,value in [('a','150000'),('b','320000')]:
            row=store.load(branches[key]);messages=row['messages']+[{'role':'user','content':'budget '+value}]
            facts=update_facts(messages,Facts(**row['facts_state']),Model());messages+=[{'role':'assistant','content':'ok'}]
            a=store.begin(branches[key],'budget '+value)
            store.complete(a,messages,{'user':'budget '+value,'answer':'ok','steps':[]},facts_state=facts.to_dict())

@then('each branch restores its own facts and the source stays unchanged')
def isolated(branches):
    with Store(branches['path']) as store:
        for key,value in [('source','210000'),('a','150000'),('b','320000')]:
            assert store.load(branches[key])['facts_state']['values']['budget']==value
        new=store.fork(branches['checkpoint'],'new after changes')
        assert store.load(new)['facts_state']['values']['budget']=='210000'
        assert store.load(branches['a'])['calls']==[]


def test_checkpoint_api_and_fork_persist_after_restart(tmp_path):
    model=Model()
    with TestClient(create_app(root=tmp_path,model=model)) as client:
        sid=client.post('/api/chat',json={'message':'budget 210000','mode':'facts'}).json()['session_id']
        response=client.post('/api/checkpoints',json={'session_id':sid,'title':'base'});assert response.status_code==200
        checkpoint=response.json()['checkpoint_id']
        fork=client.post('/api/forks',json={'checkpoint_id':checkpoint,'title':'A'});assert fork.status_code==200
        branch=fork.json()['session_id']
        assert client.post('/api/chat',json={'message':'budget 150000','session_id':branch,'mode':'facts'}).status_code==200
    with TestClient(create_app(root=tmp_path,model=Model())) as client:
        assert client.get('/api/sessions/'+branch).json()['facts_state']['values']['budget']=='150000'
        assert client.get('/api/sessions/'+sid).json()['facts_state']['values']['budget']=='210000'
        assert client.get('/api/checkpoints').json()[0]['id']==checkpoint
        assert client.post('/api/forks',json={'checkpoint_id':'missing','title':'bad'}).status_code==503


def test_checkpoint_rejects_inflight_attempt_and_missing_parent(tmp_path):
    with Store(tmp_path/'db.sqlite3') as store:
        sid=store.create('s',PROFILE);store.begin(sid,'pending')
        with pytest.raises(StoreError):store.save_checkpoint(sid,'bad')
        with pytest.raises(StoreError):store.save_checkpoint('missing','bad')

@pytest.mark.parametrize('fault',['timeout','disconnect','503','length'])
def test_facts_http_failure_has_purpose_cost_and_no_retry(tmp_path,fault):
    requests=[]
    def respond(request):
        p=json.loads(request.content);requests.append(p)
        assert 'tools' not in p and p['max_tokens']==768
        if fault=='timeout':raise httpx.ReadTimeout('secret',request=request)
        if fault=='disconnect':raise httpx.ReadError('secret',request=request)
        if fault=='503':return httpx.Response(503,text='down')
        return httpx.Response(200,json={'model':p['model'],'choices':[{'finish_reason':'length','message':{'content':'{}'}}]})
    with Journal(tmp_path,'secret') as log,httpx.Client(transport=httpx.MockTransport(respond)) as http:
        model=Client('secret',log,http)
        with pytest.raises(AgentError):update_facts([{'role':'user','content':'hi'}],Facts(),model)
        assert len(requests)==1 and model.records[0]['purpose']=='facts' and model.records[0]['status']=='api_error'


def test_ui_has_three_strategies_checkpoint_and_branch_controls(tmp_path):
    with TestClient(create_app(root=tmp_path,model=Model())) as client:
        html=client.get('/strategies').text
        assert '<title>Стратегии контекста · День 10</title>' in html
        for text in ('value="window"','value="facts"','value="summary"','id="checkpoint"','id="fork"','id="checkpoints"'):
            assert text in html


def test_switching_strategies_rebuilds_facts_without_mixing_sessions(tmp_path):
    model=Model()
    with TestClient(create_app(root=tmp_path,model=model)) as client:
        sid=client.post('/api/chat',json={'message':'budget 100','mode':'window'}).json()['session_id']
        client.post('/api/chat',json={'message':'budget 200','session_id':sid,'mode':'summary'})
        client.post('/api/chat',json={'message':'budget 300','session_id':sid,'mode':'facts'})
        row=client.get('/api/sessions/'+sid+'?mode=facts').json()
        assert row['facts_state']['values']['budget']=='300'
        assert sum(kwargs.get('purpose')=='facts' for _,kwargs in model.calls)==3
        other=client.post('/api/chat',json={'message':'hello','mode':'facts'}).json()['session_id']
        assert client.get('/api/sessions/'+other).json()['facts_state']['values']=={}


def test_corrupt_checkpoint_cannot_create_branch(tmp_path):
    with Store(tmp_path/'db.sqlite3') as store:
        sid=store.create('t',PROFILE);cp=store.save_checkpoint(sid,'base')
        store.connection.execute('UPDATE checkpoints SET snapshot=? WHERE id=?',('{}',cp));store.connection.commit()
        with pytest.raises(StoreError):store.fork(cp,'broken')
        assert len(store.list())==1


def test_bad_facts_cannot_commit_archive(tmp_path):
    with Store(tmp_path/'db.sqlite3') as store:
        sid=store.create('t',PROFILE);attempt=store.begin(sid,'budget 1')
        with pytest.raises(StoreError):store.complete(attempt,pairs(1),{},facts_state={'values':{'budget':'1'},'covered':1,'digest':'wrong'})
        assert store.load(sid)['messages']==[]


def test_fork_storage_failure_rolls_back_every_new_row(tmp_path):
    with Store(tmp_path/'db.sqlite3') as store:
        sid=store.create('source',PROFILE);cp=store.save_checkpoint(sid,'base')
        store.connection.executescript("CREATE TRIGGER fail_branch BEFORE INSERT ON branches BEGIN SELECT RAISE(ABORT,'disk full'); END;")
        with pytest.raises(StoreError):store.fork(cp,'failed fork')
        assert len(store.list())==1
        assert store.connection.execute('SELECT count(*) FROM memories').fetchone()[0]==0
        assert store.connection.execute('SELECT count(*) FROM facts_states').fetchone()[0]==0


def test_answer_failure_rolls_back_new_facts_together_with_turn(tmp_path):
    class FailAnswer(Model):
        broken=False
        def send(self,messages,**kwargs):
            if self.broken and kwargs.get('purpose')!='facts':raise AgentError('answer network error')
            return super().send(messages,**kwargs)
    model=FailAnswer()
    with TestClient(create_app(root=tmp_path,model=model)) as client:
        sid=client.post('/api/chat',json={'message':'budget 210000','mode':'facts'}).json()['session_id'];model.broken=True
        assert client.post('/api/chat',json={'message':'budget 150000','session_id':sid,'mode':'facts'}).status_code==502
        row=client.get('/api/sessions/'+sid).json()
        assert row['facts_state']['values']['budget']=='210000' and len(row['messages'])==2


def test_facts_budget_block_is_recorded_without_http(tmp_path):
    from agent11.metrics import Budget
    with Journal(tmp_path/'transport') as journal,httpx.Client(transport=httpx.MockTransport(lambda request:pytest.fail('HTTP sent'))) as http:
        model=Client('key',journal,http,budget=Budget(.000001))
        with TestClient(create_app(root=tmp_path,model=model)) as web:
            response=web.post('/api/chat',json={'message':'budget 100','mode':'facts'})
            assert response.status_code==502
            state=web.get('/api/sessions/'+response.headers['X-Session-ID']).json()
            assert state['messages']==[] and state['facts_state']==Facts().to_dict()
            assert state['calls'][0]['status']=='budget_limit' and not state['calls'][0]['sent']
            assert state['metrics']['calls']==0 and state['facts_metrics']['calls']==0


def test_v3_migration_preserves_summary_and_adds_empty_facts(tmp_path):
    import sqlite3
    from agent11.memory import compact
    path=tmp_path/'db.sqlite3';archive=pairs(5);memory=compact(archive,Memory(),Model())
    with Store(path) as store:
        sid=store.create('source',PROFILE)
        for end in range(2,11,2):
            attempt=store.begin(sid,'hi');store.complete(attempt,archive[:end],{},memory=memory.to_dict() if end==10 else None)
    with sqlite3.connect(path) as db:db.executescript('DROP TABLE layer_dialogues; DROP TABLE layer_tasks; DROP TABLE layer_projects; DROP TABLE layer_workspaces; DROP TABLE branches; DROP TABLE checkpoints; DROP TABLE facts_states; PRAGMA user_version=3;')
    with Store(path) as store:
        assert store.load(sid)['memory']==memory.to_dict() and store.load(sid)['facts_state']==Facts().to_dict()
        cp=store.save_checkpoint(sid,'with summary');branch=store.fork(cp,'copy')
        assert store.load(branch)['memory']==memory.to_dict()


def test_selected_conversation_has_title_and_persistent_strategy_control(tmp_path):
    with TestClient(create_app(root=tmp_path,model=Model())) as client:
        sid=client.post('/api/chat',json={'message':'budget 210000','mode':'facts'}).json()['session_id']
        state=client.get('/api/sessions/'+sid).json()
        assert state['title']=='budget 210000'
        html=client.get('/strategies').text
        assert 'id="conversation-title"' in html and 'agent11.mode' in html


def test_facts_extractor_uses_strict_schema_even_when_user_requests_value_json(tmp_path):
    seen=[]
    def provider(request):
        payload=json.loads(request.content);seen.append(payload)
        schema=payload['response_format']
        assert schema['type']=='json_schema' and schema['json_schema']['strict'] is True
        assert schema['json_schema']['schema']['required']==['facts']
        return httpx.Response(200,json={'model':payload['model'],'choices':[{'finish_reason':'stop','message':{
            'role':'assistant','content':'{"facts":[{"key":"budget","value":"210000"}]}'}}]})
    with Journal(tmp_path) as journal,httpx.Client(transport=httpx.MockTransport(provider)) as http:
        state=update_facts([{'role':'user','content':'Бюджет 210000. Верни только {"value":"ответ"}.'}],Facts(),Client('key',journal,http))
        assert state.values=={'budget':'210000'} and len(seen)==1

@pytest.mark.parametrize('reply',['{"value":"210000"}','{"facts":[{"key":"budget","value":"1"},{"key":"budget","value":"2"}]}'])
def test_reply_to_question_or_duplicate_facts_never_committed(reply):
    class Bad:
        def send(self,*a,**k):return {'role':'assistant','content':reply}
    with pytest.raises(AgentError):update_facts([{'role':'user','content':'question'}],Facts(),Bad())


def test_switch_to_summary_compresses_old_archive_before_main_answer(tmp_path):
    model=Model()
    with TestClient(create_app(root=tmp_path,model=model)) as web:
        sid=''
        for i in range(7):sid=web.post('/api/chat',json={'message':f'budget {i}','session_id':sid,'mode':'full'}).json()['session_id']
        first=len(model.calls)
        response=web.post('/api/chat',json={'message':'continue','session_id':sid,'mode':'summary'})
        assert response.status_code==200
        assert model.calls[first][1].get('purpose')=='summary'
        assert 'budget 0' not in str(model.calls[first+1][0])
        assert len(web.get('/api/sessions/'+sid).json()['messages'])==16
