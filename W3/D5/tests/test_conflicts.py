from datetime import datetime, timezone
import json
import httpx
from fastapi.testclient import TestClient
from pytest_bdd import given, scenarios, then, when

from agent15.project_facts import Change, ProjectFacts
from agent15.web import create_app

scenarios('features/conflicts.feature')


class Model:
    def __init__(self):
        self.requests=[]
    def send(self,messages,**kwargs):
        self.requests.append(messages)
        return {'role':'assistant','content':'2026-12-22'}


def workspace(web):
    sid=web.post('/api/layers',json={'title':'Маяк'}).raise_for_status().json()['id']
    return sid,f'/api/layers/{sid}'


def say(web,path,message):
    return web.post(path+'/chat',json={'message':message}).raise_for_status().json()


@given('a project with a recorded deadline',target_fixture='case')
def project(tmp_path):
    model=Model()
    with TestClient(create_app(root=tmp_path, model=model)) as web:
        _,path=workspace(web)
        say(web,path,'Срок: 2026-12-18')
        web.post(path+'/write/working',json={'key':'legacy','value':'2026-12-18'}).raise_for_status()
        yield web,path,model


@when('the deadline is changed in chat')
def change(case):
    web,path,_=case
    say(web,path,'Срок: 2026-12-22')


@then('the model sees only the new deadline and the audit records both values')
def verify(case):
    web,path,model=case
    say(web,path,'Какой актуальный срок?')
    sent=json.dumps(model.requests[-1],ensure_ascii=False)
    assert '2026-12-22' in sent and '2026-12-18' not in sent
    view=web.get(path+'/facts').raise_for_status().json()
    assert view['history'][-1]['before']=='2026-12-18' and view['history'][-1]['after']=='2026-12-22'
    assert '2026-12-18' in str(web.get(path).json()['memory']['short'])


def test_older_write_and_late_arrival_cannot_revive_cancelled_requirement(tmp_path):
    with TestClient(create_app(root=tmp_path,model=Model())) as web:
        sid,path=workspace(web)
        facts=ProjectFacts(web.app.state.store)
        old=datetime(2026,11,20,tzinfo=timezone.utc)
        new=datetime(2026,11,21,tzinfo=timezone.utc)
        facts.apply(sid,Change('requirement','1080p'),written_at=old)
        assert facts.apply(sid,Change('requirement',None),written_at=new)['applied']
        assert not facts.apply(sid,Change('requirement','4K'),written_at=old)['applied']
        state=web.get(path+'/facts').json()
        assert state['current']['requirement']['value'] is None and len(state['history'])==2
        assert state['current']['requirement']['written_at'].startswith('2026-11-21')


def task_action(web,path,action,note=''):
    before=web.get(path+'/task').raise_for_status().json()
    return web.post(path+'/task/actions',json={'action':action,'note':note,'revision':before['revision']})


def test_decisions_accumulate_and_finish_clears_only_working_memory(tmp_path):
    with TestClient(create_app(root=tmp_path,model=Model())) as web:
        _,path=workspace(web)
        say(web,path,'Решение: Проверить звук')
        say(web,path,'Решение: Записать экран')
        say(web,path,'Версия: v1.2')
        web.post(path+'/write/working',json={'key':'step','value':'Черновик'}).raise_for_status()
        web.post(path+'/write/long',json={'key':'leader','value':'Ирина'}).raise_for_status()
        web.post(path+'/task',json={'title':'Демо'}).raise_for_status()
        task_action(web,path,'approve_plan','План').raise_for_status()
        task_action(web,path,'submit_result','Результат').raise_for_status()
        assert web.get(path).json()['memory']['working']
        task_action(web,path,'accept_validation','Проверено').raise_for_status()
    with TestClient(create_app(root=tmp_path,model=Model())) as web:
        state=web.get(path).json()
        assert state['task']['stage']=='done' and state['memory']['working']=={}
        assert state['memory']['long']['leader']=='Ирина'
        facts=web.get(path+'/facts').json()
        assert [d['value'] for d in facts['decisions']]==['Проверить звук','Записать экран']
        assert facts['current']['version']['value']=='v1.2'


def test_network_failure_preserves_facts_and_dialogue(tmp_path):
    from agent15.log import Journal
    from agent15.transport import Client
    def fail(request):
        raise httpx.ReadTimeout('timeout',request=request)
    with Journal(tmp_path/'transport') as log,httpx.Client(transport=httpx.MockTransport(fail)) as http:
        model=Client('test-key',log,http)
        with TestClient(create_app(root=tmp_path,model=model)) as web:
            _,path=workspace(web)
            say(web,path,'Подход: Репетиция')
            old=web.get(path).json()
            answer=web.post(path+'/chat',json={'message':'Какой подход?'})
            assert answer.status_code==502 and len(model.records)==1
            assert web.get(path).json()==old


def test_copy_keeps_independent_fact_projection_and_never_replays_stale_transcript(tmp_path):
    model=Model()
    with TestClient(create_app(root=tmp_path,model=model)) as web:
        _,path=workspace(web)
        say(web,path,'Срок: 2026-12-18')
        say(web,path,'Срок: 2026-12-22')
        new=web.post(path+'/copy').raise_for_status().json()['id']
        copy=f'/api/layers/{new}'
        assert web.get(copy+'/facts').json()==web.get(path+'/facts').json()
        say(web,copy,'Какой срок?')
        assert '2026-12-18' not in str(model.requests[-1])
        say(web,copy,'Срок: 2026-12-25')
        assert web.get(path+'/facts').json()['current']['deadline']['value']=='2026-12-22'


def test_completed_task_and_cleanup_rollback_together_on_sqlite_failure(tmp_path):
    with TestClient(create_app(root=tmp_path,model=Model())) as web:
        _,path=workspace(web)
        web.post(path+'/write/working',json={'key':'step','value':'Черновик'}).raise_for_status()
        web.post(path+'/task',json={'title':'Демо'}).raise_for_status()
        task_action(web,path,'approve_plan','План').raise_for_status()
        task_action(web,path,'submit_result','Результат').raise_for_status()
        before=web.get(path).json()
        with web.app.state.store.transaction() as db:
            db.execute("""CREATE TRIGGER stop_cleanup BEFORE UPDATE ON layer_tasks
                BEGIN SELECT RAISE(ABORT, 'test failure'); END""")
        assert task_action(web,path,'accept_validation','Проверено').status_code==503
        assert web.get(path).json()==before
