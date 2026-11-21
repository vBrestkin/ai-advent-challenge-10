import json
import httpx
import pytest
from fastapi.testclient import TestClient
from pytest_bdd import given, scenarios, then, when
from agent15.profiles import preference_patch
from agent15.web import create_app
from agent15.log import Journal
from agent15.transport import Client

scenarios('features/profile.feature')


class Model:
    def __init__(self):
        self.requests = []

    def send(self, messages):
        self.requests.append(messages)
        return {'role': 'assistant', 'content': 'Понял.'}


def new_chat(web, title):
    return web.post('/api/layers', json={'title': title}).raise_for_status().json()['id']


def chat(web, sid, message, suffix=''):
    return web.post(f'/api/layers/{sid}/chat{suffix}', json={'message': message})


def profile(web, sid):
    return web.get(f'/api/layers/{sid}').raise_for_status().json()['profile']


@given('two separate conversations', target_fixture='case')
def case(tmp_path):
    with TestClient(create_app(root=tmp_path, model=Model())) as web:
        a, b = (new_chat(web, title) for title in ('A', 'B'))
        web.put(f'/api/layers/{a}/profile', json={'length':'detailed','code':'with'}).raise_for_status()
    return tmp_path, a, b


@when('the user updates one profile in chat')
def update(case):
    root, a, _ = case
    model = Model()
    with TestClient(create_app(root=root, model=model)) as web:
        result = chat(web, a, 'Теперь отвечай кратко и без кода.').raise_for_status().json()
        assert result['profile_changed']
        assert 'Не включай блоки кода' in str(model.requests[0])


@then('after reopening only that conversation uses the new preference')
def restored(case):
    root, a, b = case
    model = Model()
    with TestClient(create_app(root=root, model=model)) as web:
        assert profile(web, a) == {'length':'brief','code':'without'}
        assert profile(web, b) == {'length':'neutral','code':'neutral'}
        chat(web, a, 'Как считать слова?').raise_for_status()
        chat(web, b, 'Как считать слова?').raise_for_status()
        assert 'Не включай блоки кода' in str(model.requests[0])
        assert 'Не включай блоки кода' not in str(model.requests[1])


@pytest.mark.parametrize('message', [
    'Объясни, что значит «отвечай кратко и без кода».',
    'На этот вопрос ответь кратко.',
    'Ответь кратко и объясни, что такое список.',
    'Отвечай кратко и подробно.',
])
def test_nonpersistent_or_ambiguous_requests_do_not_change_profile(message):
    assert preference_patch(message) == {}


def test_partial_preference_patch_keeps_other_setting():
    assert preference_patch('Отвечай подробно.') == {'length':'detailed'}


@pytest.mark.parametrize('fault', [httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError])
def test_network_error_does_not_commit_turn_or_preference(tmp_path, fault):
    def fail(request):
        raise fault('network', request=request)
    with Journal(tmp_path/'transport') as log, httpx.Client(transport=httpx.MockTransport(fail)) as http:
        model = Client('key', log, http)
        with TestClient(create_app(root=tmp_path, model=model)) as web:
            sid = new_chat(web, 'A')
            url = f'/api/layers/{sid}'
            old = web.get(url).json()
            assert chat(web, sid, 'Теперь отвечай кратко и без кода.').status_code == 502
            assert web.get(url).json() == old
            assert len(model.records) == 1


def test_read_only_chat_applies_preference_without_persisting_it(tmp_path):
    model = Model()
    with TestClient(create_app(root=tmp_path, model=model)) as web:
        sid = new_chat(web, 'A')
        result = chat(web, sid, 'Теперь отвечай кратко и без кода.', '?save=false').raise_for_status().json()
        assert not result['profile_changed'] and profile(web,sid)['length']=='neutral'
        assert 'Не включай блоки кода' in str(model.requests[0])


def test_profiles_apply_to_every_tool_round_and_copy_is_independent(tmp_path):
    class ToolModel(Model):
        def send(self, messages):
            self.requests.append(messages)
            if len(self.requests)==1:
                return {'role':'assistant','content':None,'tool_calls':[{'id':'x','type':'function',
                    'function':{'name':'find_order','arguments':json.dumps({'order_id':'A-104'})}}]}
            return {'role':'assistant','content':'Понял.'}
    model = ToolModel()
    with TestClient(create_app(root=tmp_path, model=model)) as web:
        sid = new_chat(web, 'A')
        web.put(f'/api/layers/{sid}/profile', json={'length':'brief','code':'without'}).raise_for_status()
        chat(web,sid,'Найди заказ').raise_for_status()
        assert len(model.requests)==2
        assert all(any('Не включай блоки кода' in str(m) for m in call) for call in model.requests)
        copy = web.post(f'/api/layers/{sid}/copy').raise_for_status().json()['id']
        assert profile(web,copy)==profile(web,sid)
        web.put(f'/api/layers/{copy}/profile',json={'length':'detailed','code':'with'}).raise_for_status()
        assert profile(web,sid)=={'length':'brief','code':'without'}


def test_day11_database_opens_with_neutral_profile_and_retains_dialogue(tmp_path):
    import sqlite3
    from agent15.layers import Layers
    from agent15.store import Store
    path = tmp_path/'history.sqlite3'
    with Store(path) as store:
        layers = Layers(store)
        sid = layers.create('День 11')
        messages = [{'role':'user','content':'Старый диалог'},{'role':'assistant','content':'Принято'}]
        layers.save_dialogue(sid, messages)
    with sqlite3.connect(path) as db:
        db.executescript('DROP TABLE project_fact_events; DROP TABLE stack_rule_changes; DROP TABLE stack_rules; DROP TABLE task_states; DROP TABLE layer_profiles; PRAGMA user_version=5;')
    with TestClient(create_app(root=tmp_path, model=Model(), data_dir=tmp_path)) as web:
        restored = web.get(f'/api/layers/{sid}').raise_for_status().json()
        assert restored['memory']['short']==messages
        assert restored['profile']=={'length':'neutral','code':'neutral'}
        assert chat(web,sid,'Впредь отвечай подробно.').raise_for_status().json()['profile_changed']


def test_profile_and_dialogue_rollback_together_on_storage_failure(tmp_path):
    model = Model()
    with TestClient(create_app(root=tmp_path, model=model)) as web:
        sid = new_chat(web, 'A')
        url = f'/api/layers/{sid}'
        old = web.get(url).json()
        with web.app.state.store.transaction() as db:
            db.execute("""CREATE TRIGGER reject_profile BEFORE UPDATE ON layer_profiles
                BEGIN SELECT RAISE(ABORT, 'test write failure'); END""")
        assert chat(web,sid,'Теперь отвечай кратко и без кода.').status_code==503
        assert web.get(url).json()==old


def test_personalized_chat_has_general_role_and_retains_order_tools(tmp_path):
    model = Model()
    with TestClient(create_app(root=tmp_path, model=model)) as web:
        sid = new_chat(web, 'A')
        chat(web,sid,'Как посчитать частоту слов?').raise_for_status()
        system = model.requests[0][0]['content']
        assert 'обычные вопросы объясняй напрямую' in system
        assert 'find_order' in system
        assert 'Ты помощник с учебными заказами' not in system
