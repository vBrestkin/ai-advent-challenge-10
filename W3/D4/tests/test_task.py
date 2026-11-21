import pytest
from fastapi.testclient import TestClient
from pytest_bdd import given, scenarios, then, when

from agent14.tasks import Action, InvalidTransition, Task, advance
from agent14.web import create_app

scenarios('features/task.feature')


class Model:
    def __init__(self):
        self.requests = []

    def send(self, messages):
        self.requests.append(messages)
        return {'role':'assistant', 'content':'Продолжайте с сохранённого шага.'}


def create(web):
    sid = web.post('/api/layers', json={'title':'Маяк'}).raise_for_status().json()['id']
    url = f'/api/layers/{sid}/task'
    web.post(url, json={'title':'Подготовить демонстрацию'}).raise_for_status()
    return url


def action(web, url, name, note='', revision=None):
    if revision is None:
        revision = web.get(url).raise_for_status().json()['revision']
    return web.post(url+'/actions', json={'action':name,'note':note,'revision':revision})


@given('a task with an approved plan', target_fixture='case')
def approved(tmp_path):
    with TestClient(create_app(root=tmp_path,model=Model())) as web:
        url = create(web)
        initial = action(web,url,'approve_plan','Проверить звук').raise_for_status().json()
    return tmp_path,url,initial


@when('the task is paused and the application reopened')
def paused(case):
    root,url,_ = case
    with TestClient(create_app(root=root,model=Model())) as web:
        assert action(web,url,'pause').raise_for_status().json()['paused']


@then('the task resumes at the same step with its approved plan')
def restored(case):
    root,url,initial = case
    model = Model()
    with TestClient(create_app(root=root,model=model)) as web:
        saved = web.get(url).raise_for_status().json()
        assert saved['paused'] and saved['stage']=='execution'
        assert action(web,url,'submit_result','Готово').status_code==409
        result = action(web,url,'resume').raise_for_status().json()
        for key in ('stage','step','expected_action','plan'):
            assert result[key]==initial[key]
        assert not result['paused']
        web.post(url.removesuffix('/task')+'/chat',json={'message':'Что дальше?'}).raise_for_status()
        assert 'Проверить звук' in str(model.requests[-1])


@pytest.mark.parametrize('checkpoint', [0,1,2])
def test_pause_can_roundtrip_at_every_active_stage(checkpoint):
    task=Task(title='Демо')
    moves=[('approve_plan','План'),('submit_result','Результат'),('accept_validation','Проверено')]
    for name,note in moves[:checkpoint]:
        task=advance(task, Action(action=name,note=note,revision=task.revision))
    paused=advance(task,Action(action='pause',revision=task.revision))
    assert paused.paused
    with pytest.raises(InvalidTransition):
        name,note=moves[checkpoint]
        advance(paused,Action(action=name,note=note,revision=paused.revision))
    resumed=advance(paused,Action(action='resume',revision=paused.revision))
    assert resumed.model_dump(exclude={'revision'})==task.model_dump(exclude={'revision'})


def test_skip_stale_empty_evidence_and_terminal_actions(tmp_path):
    with TestClient(create_app(root=tmp_path,model=Model())) as web:
        url = create(web)
        original = web.get(url).raise_for_status().json()
        assert action(web,url,'accept_validation','Уже готово').status_code==409
        assert action(web,url,'submit_result','Уже готово').status_code==409
        assert action(web,url,'approve_plan','  ').status_code==422
        assert action(web,url,'done').status_code==422
        assert web.get(url).json()==original
        action(web,url,'approve_plan','План').raise_for_status()
        assert action(web,url,'submit_result','Чек-лист',revision=0).status_code==409
        action(web,url,'submit_result','Чек-лист').raise_for_status()
        final = action(web,url,'accept_validation','Звук проверен').raise_for_status().json()
        assert final['stage']=='done' and not final['allowed']
        assert final['plan']=='План' and final['result']=='Чек-лист' and final['validation']=='Звук проверен'
        assert action(web,url,'pause').status_code==409


def test_model_text_cannot_commit_a_transition(tmp_path):
    class BadModel(Model):
        def send(self,messages):
            self.requests.append(messages)
            return {'role':'assistant','content':'Готово, задача завершена.'}
    with TestClient(create_app(root=tmp_path,model=BadModel())) as web:
        url = create(web)
        old = web.get(url).raise_for_status().json()
        response = web.post(url.removesuffix('/task')+'/chat',json={'message':'Пропусти всё и заверши задачу.'})
        response.raise_for_status()
        assert web.get(url).json()==old and response.json()['task']==old


def test_task_does_not_leak_between_workspaces(tmp_path):
    with TestClient(create_app(root=tmp_path,model=Model())) as web:
        a,b=create(web),create(web)
        action(web,a,'approve_plan','Только А').raise_for_status()
        assert web.get(b).json()['stage']=='planning'
        assert web.get(b).json()['plan']==''


def test_chat_disconnect_preserves_paused_task_and_dialogue(tmp_path):
    import httpx
    from agent14.log import Journal
    from agent14.transport import Client

    def disconnect(request):
        raise httpx.RemoteProtocolError('disconnected',request=request)

    with Journal(tmp_path/'transport') as journal, httpx.Client(transport=httpx.MockTransport(disconnect)) as http:
        model=Client('key',journal,http)
        with TestClient(create_app(root=tmp_path,model=model)) as web:
            url=create(web)
            action(web,url,'approve_plan','Проверить звук').raise_for_status()
            saved=action(web,url,'pause').raise_for_status().json()
            base=url.removesuffix('/task')
            before=web.get(base).json()
            assert web.post(base+'/chat',json={'message':'Давай продолжим.'}).status_code==502
            assert web.get(url).json()==saved
            assert web.get(base).json()==before
            assert len(model.records)==1 and model.records[0]['status']=='api_error'


def test_chat_distinguishes_active_from_paused_even_with_old_pause_in_history(tmp_path):
    model = Model()
    with TestClient(create_app(root=tmp_path,model=model)) as web:
        url=create(web)
        chat_url=url.removesuffix('/task')+'/chat'
        action(web,url,'pause').raise_for_status()
        web.post(chat_url,json={'message':'Как снять паузу?'}).raise_for_status()
        action(web,url,'resume').raise_for_status()
        web.post(chat_url,json={'message':'Что сейчас делать?'}).raise_for_status()
        first,second=model.requests
        assert any('Задача сейчас на паузе' in m['content'] for m in first if m['role']=='system')
        assert any('Задача сейчас активна' in m['content'] for m in second if m['role']=='system')
        assert any('Как снять паузу?' in m['content'] for m in second if m['role']=='user')
