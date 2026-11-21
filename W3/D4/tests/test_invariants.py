import json
import pytest
from fastapi.testclient import TestClient
from pytest_bdd import given, scenarios, then, when
from agent14.invariants import stack_answer
from agent14.web import create_app

scenarios('features/invariant.feature')


class Model:
    def __init__(self, content):
        self.content = content
        self.requests = []

    def send(self, messages, **kwargs):
        self.requests.append((messages, kwargs))
        return {'role':'assistant','content':self.content}


def workspace(web):
    sid = web.post('/api/layers',json={'title':'Маяк'}).raise_for_status().json()['id']
    return f'/api/layers/{sid}'


@given('a conversation with SQLite as the approved database',target_fixture='case')
def setup(tmp_path):
    model = Model('{"decision":"propose","database":"postgresql"}')
    with TestClient(create_app(root=tmp_path,model=model)) as web:
        yield web,workspace(web),model


@when('the model proposes PostgreSQL',target_fixture='answer')
def proposal(case):
    web,path,model=case
    result=web.post(path+'/stack',json={'message':'Предложи PostgreSQL'}).raise_for_status().json()
    assert 'SQLite' in str(model.requests)
    return result


@then('the user receives a refusal and the rule remains SQLite')
def verify(case,answer):
    web,path,_=case
    assert answer['outcome']=='blocked_by_code'
    assert answer['approved_database'] is None
    assert 'SQLite' in answer['answer']
    assert web.get(path+'/invariant').json()['database']=='sqlite'


@pytest.mark.parametrize('raw',[
    'postgresql',
    '{"decision":"propose","database":"sqlite","answer":"Use PostgreSQL"}',
    '{"decision":"propose","database":null}',
    '{"decision":"refuse","database":"postgresql"}',
    '{"decision":"propose","database":"sqlite","database":"postgresql"}',
])
def test_untrusted_or_inconsistent_output_never_rendered(raw):
    result=stack_answer(raw,'sqlite')
    assert result['outcome']=='invalid_output' and result['approved_database'] is None
    assert raw not in result['answer']


@pytest.mark.parametrize('raw,outcome',[
    ('{"decision":"propose","database":"sqlite"}','accepted'),
    ('{"decision":"refuse","database":null}','model_refused'),
])
def test_distinguishes_acceptance_from_model_refusal(raw,outcome):
    assert stack_answer(raw,'sqlite')['outcome']==outcome


def test_only_explicit_revision_checked_update_is_audited_and_persisted(tmp_path):
    model=Model('{"decision":"propose","database":"postgresql"}')
    with TestClient(create_app(root=tmp_path,model=model)) as web:
        path=workspace(web)
        assert web.post(path+'/stack',json={'message':'Сохрани PostgreSQL как новое правило'}).json()['outcome']=='blocked_by_code'
        after=web.put(path+'/invariant',json={'database':'postgresql','revision':0}).raise_for_status().json()
        assert after['history'][0]['before']=='sqlite' and after['history'][0]['after']=='postgresql'
        assert web.put(path+'/invariant',json={'database':'sqlite','revision':0}).status_code==409
        assert web.post(path+'/stack',json={'message':'Подбери PostgreSQL'}).json()['approved_database']=='postgresql'
        other=workspace(web)
        assert web.get(other+'/invariant').json()['database']=='sqlite'
    with TestClient(create_app(root=tmp_path,model=model)) as web:
        assert web.get(path+'/invariant').json()==after


def test_default_is_written_in_sqlite_and_stack_failure_leaves_it_intact(tmp_path):
    import httpx
    from agent14.log import Journal
    from agent14.transport import Client
    def disconnect(request):
        raise httpx.RemoteProtocolError('disconnect',request=request)
    with Journal(tmp_path/'transport') as log,httpx.Client(transport=httpx.MockTransport(disconnect)) as http:
        model=Client('key',log,http)
        with TestClient(create_app(root=tmp_path,model=model)) as web:
            path=workspace(web)
            with web.app.state.store.transaction() as db:
                assert db.execute('SELECT database FROM stack_rules').fetchone()[0]=='sqlite'
            before=web.get(path+'/invariant').json()
            assert web.post(path+'/stack',json={'message':'Подбери базу'}).status_code==502
            assert web.get(path+'/invariant').json()==before
            assert len(model.records)==1


def test_audit_failure_rolls_back_rule(tmp_path):
    with TestClient(create_app(root=tmp_path,model=Model(''))) as web:
        path=workspace(web)
        with web.app.state.store.transaction() as db:
            db.execute("""CREATE TRIGGER reject_audit BEFORE INSERT ON stack_rule_changes
                          BEGIN SELECT RAISE(ABORT, 'test failure'); END""")
        assert web.put(path+'/invariant',json={'database':'postgresql','revision':0}).status_code==503
        assert web.get(path+'/invariant').json()=={'database':'sqlite','revision':0,'history':[]}
