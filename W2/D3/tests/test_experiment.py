import hashlib,json
import httpx,pytest
from agent8.setup_tokenizer import install
from agent8.experiment import evaluate,write_report
from agent8.log import Journal


def test_tokenizer_install_is_atomic_and_checks_hash(tmp_path,monkeypatch):
    payload=b'{"tokenizer":"test"}'
    monkeypatch.setattr('agent8.setup_tokenizer.TOKENIZER_SHA256',hashlib.sha256(payload).hexdigest())
    path=tmp_path/'tokenizer.json'
    with Journal(tmp_path/'log') as log,httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,content=payload))) as http:
        install(path,http,log)
        assert path.read_bytes()==payload
        monkeypatch.setattr('agent8.setup_tokenizer.TOKENIZER_SHA256','wrong')
        with pytest.raises(ValueError):install(path,http,log)
        assert path.read_bytes()==payload


def test_evaluation_requires_api_rejection_for_overflow():
    rows=[{'scenario':'short','http_status':200,'calls':[{'sent':True,'total_tokens':100,'cost_rub':.01}], 'history_before':[], 'history_after':[{'role':'user'}]},
          {'scenario':'long','http_status':200,'calls':[{'sent':True,'total_tokens':1000,'cost_rub':.1}], 'history_before':[], 'history_after':[{'role':'user'}]},
          {'scenario':'overflow','http_status':502,'calls':[{'sent':True,'http_status':400,'response':'context length exceeded','cost_rub':None}], 'history_before':[], 'history_after':[]}]
    result=evaluate(rows)
    assert result['overflow_rejected_by_api'] and result['overflow_history_preserved']
    rows[-1]['calls'][0].update(sent=False,status='local_context_limit')
    assert not evaluate(rows)['overflow_rejected_by_api']


def test_report_contains_three_scenarios_and_unknown_cost(tmp_path):
    rows=[{'scenario':name,'http_status':200,'calls':[{'sent':True,'cost_rub':None}],
        'history_before':[],'history_after':[],'prompt_tokens_local':10,'request_tokens_estimate':20}
        for name in ('short','long','overflow')]
    data={'records':rows,'budget':{'limit_rub':20,'known_cost_rub':0,'unknown_reserve_rub':.2,'accounted_rub':.2},
          'status':'complete','checks':evaluate(rows),'context_limit':40960,'catalog':{},'tokenizer':{'method':'test'}}
    write_report(tmp_path,data)
    text=(tmp_path/'report.md').read_text()
    assert all(name in text for name in ('short','long','overflow'))
    assert 'неизвест' in text.lower() and (tmp_path/'turns.csv').exists()

def test_complete_experiment_through_http_with_mock_provider(tmp_path,monkeypatch):
    import math
    import agent8.experiment as experiment
    import agent8.web as web
    from agent8.transport import Client,MODEL
    from agent8.metrics import TokenCounter
    class Counter(TokenCounter):
        def __init__(self,*args):
            self.tokenizer=None;self.method='mock-json-estimate';self.exact_text=True
    monkeypatch.setattr(web,'TokenCounter',Counter)
    sent=[]
    def provider(request):
        data=json.loads(request.content);sent.append(data)
        messages=data['messages'];last=messages[-1]
        if last['role']=='user' and last['content'].startswith('Тест большого'):
            return httpx.Response(400,text='maximum context length exceeded')
        name,args=None,None
        if last['role']=='user' and last['content'].startswith('Найди сумму'):
            name,args='find_order',{'order_id':'A-104'}
        elif last['role']=='tool' and 'amount' in last['content']:
            name,args='calculate',{'operation':'multiply','a':'2390','b':'0.85'}
        message={'role':'assistant','content':'ПРИНЯТО'}
        if name:message.update(content=None,tool_calls=[{'id':name,'type':'function','function':{'name':name,'arguments':json.dumps(args)}}])
        return httpx.Response(200,json={'model':MODEL,'choices':[{'message':message,'finish_reason':'tool_calls' if name else 'stop'}],
            'usage':{'prompt_tokens':100,'completion_tokens':10,'total_tokens':110,'cost_rub':.01}})
    original=web.create_app
    with Journal(tmp_path/'transport') as journal,httpx.Client(transport=httpx.MockTransport(provider)) as http:
        model=Client('key',journal,http,counter=Counter())
        def factory(**kwargs):
            model.budget=kwargs['budget'];model.allow_overflow=kwargs['allow_overflow']
            return original(root=kwargs['root'],data_dir=kwargs['data_dir'],model=model)
        monkeypatch.setattr(experiment,'create_app',factory)
        directory=experiment.run(tmp_path)
        result=json.loads((directory/'experiment.json').read_text())
    assert result['status']=='complete' and all(result['checks'].values())
    assert len(result['records'])==12 and len(sent)==13
    assert result['records'][-1]['calls'][0]['sent'] is False
    assert result['budget']['accounted_rub']<=20

def test_experiment_help_is_logged_before_exit(tmp_path,monkeypatch):
    import sys
    import agent8.experiment as experiment
    monkeypatch.setattr(experiment,'ROOT',tmp_path)
    monkeypatch.setattr(sys,'argv',['experiment','--help'])
    with pytest.raises(SystemExit):experiment.main()
    assert list((tmp_path/'logs').glob('*.log'))


def test_tokenizer_help_does_not_download_and_is_logged(tmp_path,monkeypatch):
    import sys
    import agent8.setup_tokenizer as setup
    monkeypatch.setattr(setup,'ROOT',tmp_path)
    monkeypatch.setattr(sys,'argv',['tokenizer','--help'])
    downloads=[]
    monkeypatch.setattr(setup,'install',lambda *a:downloads.append(1))
    with pytest.raises(SystemExit):setup.main()
    assert not downloads and list((tmp_path/'logs').glob('*.log'))
