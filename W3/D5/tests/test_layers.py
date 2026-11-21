from copy import deepcopy
import pytest
from pytest_bdd import scenarios, given, when, then
from fastapi.testclient import TestClient
from agent15.layers import Layers, LAYER_NAMES
from agent15.store import Store
from agent15.web import create_app

scenarios('features/layers.feature')

@pytest.fixture
@given('three isolated pieces of knowledge', target_fixture='sample')
def sample(tmp_path):
    return tmp_path, {'short': [{'role':'user','content':'Фраза: Зажигаем маяк'},
                               {'role':'assistant','content':'Принято'}],
                      'working': {'current_step':'проверить микрофон'},
                      'long': {'project_leader':'Ирина'}}

@when('a copy loses its working memory', target_fixture='snapshots')
def cleared(sample):
    path, seed = sample
    with Store(path/'memory.sqlite3') as store:
        layers = Layers(store)
        source = layers.create('Source', seed)
        copy = layers.create('Copy', layers.load(source)['memory'])
        layers.clear(copy,'working')
    with Store(path/'memory.sqlite3') as store:
        layers = Layers(store)
        return layers.load(source), layers.load(copy)

@then('only the working fact is removed from the model context')
def verify(snapshots):
    source, copy = snapshots
    assert source['memory']['working'] == {'current_step':'проверить микрофон'}
    assert copy['memory']['working'] == {}
    for name in ('short','long'):
        assert copy['memory'][name] == source['memory'][name]
    assert 'Ирина' in str(Layers.context(copy['memory']))
    assert 'микрофон' not in str(Layers.context(copy['memory']))

@pytest.mark.parametrize('removed', LAYER_NAMES)
def test_each_clear_is_isolated(sample, removed):
    path, seed = sample
    with Store(path/'memory.sqlite3') as store:
        layers=Layers(store); sid=layers.create('test',seed)
        layers.clear(sid,removed)
        result=layers.load(sid)['memory']
        assert not result[removed]
        assert all(result[name]==seed[name] for name in LAYER_NAMES if name!=removed)

def test_explicit_writes_merge_facts_and_do_not_copy_into_transcript(sample):
    path,seed=sample
    with Store(path/'memory.sqlite3') as store:
        layers=Layers(store); sid=layers.create('test',seed)
        layers.write(sid,'long',{'preferred_interface':'веб'})
        state=layers.load(sid)
        assert state['memory']['long']=={'project_leader':'Ирина','preferred_interface':'веб'}
        assert state['memory']['short']==seed['short']
        assert state['memory']['working']==seed['working']
        assert 'веб' not in str(state['memory']['short'])

def test_window_keeps_recent_turns_and_does_not_mutate_storage(sample):
    path,seed=sample
    seed['short']=[m for i in range(5) for m in
        ({'role':'user','content':f'fact {i}'},{'role':'assistant','content':'ok'})]
    original=deepcopy(seed)
    context=Layers.context(seed)
    assert context[-6:]==seed['short'][-6:]
    assert 'fact 0' not in str(context)
    assert seed==original

def test_failed_provider_does_not_write_short_dialogue(sample):
    import httpx
    from agent15.log import Journal
    from agent15.transport import Client
    path, seed=sample
    def fail(request):raise httpx.ReadTimeout('down',request=request)
    with Journal(path/'transport') as journal,httpx.Client(transport=httpx.MockTransport(fail)) as http:
        model=Client('example',journal,http)
        with TestClient(create_app(root=path,model=model)) as web:
            sid=web.post('/api/layers',json={'title':'demo','memory':seed}).json()['id']
            assert web.post(f'/api/layers/{sid}/chat',json={'message':'Продолжим'}).status_code==502
            assert web.get(f'/api/layers/{sid}').json()['memory']==seed
            assert len(model.records)==1 and model.records[0]['status']=='api_error'

def test_chat_sends_selected_layers_and_writes_only_short(sample):
    class Model:
        def send(self,messages):
            assert 'проверить микрофон' in str(messages) and 'Ирина' in str(messages)
            return {'role':'assistant','content':'Готово'}
    path,seed=sample
    with TestClient(create_app(root=path,model=Model())) as web:
        sid=web.post('/api/layers',json={'title':'demo','memory':seed}).json()['id']
        assert web.post(f'/api/layers/{sid}/chat',json={'message':'Продолжим'}).status_code==200
        state=web.get(f'/api/layers/{sid}').json()['memory']
        assert state['short'][-2:]==[{'role':'user','content':'Продолжим'},{'role':'assistant','content':'Готово'}]
        assert state['working']==seed['working'] and state['long']==seed['long']
