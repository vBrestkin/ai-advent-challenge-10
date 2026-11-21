import io,json
import httpx,pytest
from agent13.transport import Client
from agent13.agent import AgentError
from agent13.log import Journal
from agent13.web import create_app
from fastapi.testclient import TestClient

@pytest.mark.parametrize('fault',['timeout','network','http','bad','length'])
def test_provider_errors_logged(tmp_path,fault):
    calls=[]
    def respond(request):
        calls.append(request)
        if fault=='timeout': raise httpx.ReadTimeout('secret',request=request)
        if fault=='network': raise httpx.ConnectError('secret',request=request)
        if fault=='http': return httpx.Response(503,text='secret')
        if fault=='bad': return httpx.Response(200,json={})
        return httpx.Response(200,json={'model':'qwen/qwen3-14b','choices':[{'finish_reason':'length','message':{'content':''}}]})
    with Journal(tmp_path,'secret') as log, httpx.Client(transport=httpx.MockTransport(respond)) as http:
        with pytest.raises(AgentError): Client('secret',log,http).send([{'role':'user','content':'hi'}])
    assert len(calls)==1
    assert 'secret' not in next(tmp_path.glob('*.log')).read_text()

def test_tool_response_with_null_content(tmp_path):
    tool={'id':'c','type':'function','function':{'name':'find_order','arguments':'{"order_id":"A-104"}'}}
    def respond(req):
        p=json.loads(req.content); assert p['tool_choice']=='auto' and len(p['tools'])==2
        return httpx.Response(200,json={'model':p['model'],'choices':[{'finish_reason':'tool_calls','message':{'role':'assistant','content':None,'tool_calls':[tool]}}]})
    with Journal(tmp_path) as log,httpx.Client(transport=httpx.MockTransport(respond)) as h:
        assert Client('key',log,h).send([])['tool_calls']==[tool]

def test_web_chat_and_invalid_session(tmp_path):
    from test_agent import Fake
    with TestClient(create_app(root=tmp_path,model=Fake(2))) as c:
        assert c.get('/strategies').status_code==200
        result=c.post('/api/chat',json={'message':'hi'}).json()
        assert len(result['steps'])==2
        assert c.get('/api/sessions').json()[0]['id']==result['session_id']
        assert c.post('/api/chat',json={'message':'','session_id':result['session_id']}).status_code==422
        assert c.post('/api/chat',json={'message':'hi','session_id':'missing'}).status_code==404

def test_pilot_detects_model_skipping_tools(tmp_path):
    from test_agent import Fake
    from agent13.pilot import run_checks
    with TestClient(create_app(root=tmp_path,model=Fake(0))) as client:
        checks=run_checks(client)
    assert checks[0]['passed'] and not checks[1]['passed'] and not checks[2]['passed']

@pytest.mark.parametrize('error',[httpx.ConnectTimeout,httpx.WriteTimeout,httpx.PoolTimeout,httpx.ReadError,httpx.WriteError,httpx.RemoteProtocolError])
def test_network_variants(tmp_path,error):
    calls=[]
    def respond(request):
        calls.append(request);raise error('secret',request=request)
    with Journal(tmp_path,'secret') as log,httpx.Client(transport=httpx.MockTransport(respond)) as h:
        with pytest.raises(AgentError):Client('secret',log,h).send([])
    assert len(calls)==1 and 'secret' not in next(tmp_path.glob('*.log')).read_text()

@pytest.mark.parametrize('status',[301,400,401,402,403,404,408,413,422,429,500,502,504])
def test_http_status_variants(tmp_path,status):
    with Journal(tmp_path) as log,httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(status,text='failure'))) as h:
        with pytest.raises(AgentError):Client('key',log,h).send([])

def test_log_failure_prevents_generation(tmp_path,monkeypatch):
    with Journal(tmp_path) as log,httpx.Client(transport=httpx.MockTransport(lambda r:pytest.fail('HTTP was sent'))) as h:
        def fail(*a):raise OSError('disk full')
        monkeypatch.setattr(log,'record',fail)
        with pytest.raises(OSError):Client('key',log,h).send([])

def test_pilot_exports_structured_trace(tmp_path,monkeypatch):
    import agent13.pilot as pilot
    from test_agent import Fake
    monkeypatch.setattr(pilot,'ROOT',tmp_path)
    monkeypatch.setattr(pilot,'create_app',lambda:create_app(root=tmp_path,model=Fake(0)))
    assert pilot.main()==1
    trace=json.loads((tmp_path/'submission'/'trace.json').read_text())
    assert any(row['event']=='INPUT' for row in trace)

def test_provider_message_without_role_is_normalized_before_storage(tmp_path):
    def respond(request):
        return httpx.Response(200,json={'model':'qwen/qwen3-14b','choices':[
            {'finish_reason':'stop','message':{'content':'ok'}}]})
    with Journal(tmp_path) as log,httpx.Client(transport=httpx.MockTransport(respond)) as http:
        assert Client('key',log,http).send([])['role']=='assistant'

def test_cli_help_is_logged(tmp_path,monkeypatch):
    import agent13.web as web
    import sys
    monkeypatch.setattr(web,'ROOT',tmp_path)
    monkeypatch.setattr(sys,'argv',['agent13','--help'])
    with pytest.raises(SystemExit) as exc:web.main()
    assert exc.value.code==0
    logs=list((tmp_path/'logs').glob('*.log'))
    assert logs and 'CLI_OUTPUT' in logs[0].read_text()
