import hashlib,json
import httpx,pytest
from agent9.setup_tokenizer import install
from agent9.experiment import write_report
from agent9.log import Journal


def test_tokenizer_install_is_atomic_and_checks_hash(tmp_path,monkeypatch):
    payload=b'{"tokenizer":"test"}'
    monkeypatch.setattr('agent9.setup_tokenizer.TOKENIZER_SHA256',hashlib.sha256(payload).hexdigest())
    path=tmp_path/'tokenizer.json'
    with Journal(tmp_path/'log') as log,httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,content=payload))) as http:
        install(path,http,log)
        assert path.read_bytes()==payload
        monkeypatch.setattr('agent9.setup_tokenizer.TOKENIZER_SHA256','wrong')
        with pytest.raises(ValueError):install(path,http,log)
        assert path.read_bytes()==payload


def test_grader_requires_json_and_current_value():
    from agent9.experiment import grade
    assert grade('{"value":"210000"}', ['210000'])
    assert not grade('{"value":"180000"}', ['210000'])
    assert not grade('210000', ['210000'])
    assert not grade('Возможно 180000 или 210000', ['210000'])


def test_scenario_and_facts_are_fixed_before_model_calls():
    from agent9.experiment import PROMPTS, FACTS
    assert len(PROMPTS)==12 and len(FACTS)==10
    assert all(set(fact)=={'id','question','answers'} for fact in FACTS)
    assert '210000' in str(FACTS)


def test_summary_report_includes_maintenance_and_missing_facts(tmp_path):
    from agent9.experiment import write_report
    data={'status':'complete','budget':{'known_cost_rub':1.,'unknown_reserve_rub':0.,'accounted_rub':1.,'limit_rub':20},
          'calls':[],'quizzes':[], 'archive':[], 'memory':{'summary':'','covered':0,'digest':''},'context_sizes':{}}
    write_report(tmp_path,data)
    text=(tmp_path/'report.md').read_text()
    assert 'сжатия' in text and 'summary' in text and (tmp_path/'summary.csv').exists()


def test_experiment_help_is_logged_before_exit(tmp_path,monkeypatch):
    import sys
    import agent9.experiment as experiment
    monkeypatch.setattr(experiment,'ROOT',tmp_path)
    monkeypatch.setattr(sys,'argv',['experiment','--help'])
    with pytest.raises(SystemExit):experiment.main()
    assert list((tmp_path/'logs').glob('*.log'))


def test_tokenizer_help_does_not_download_and_is_logged(tmp_path,monkeypatch):
    import sys
    import agent9.setup_tokenizer as setup
    monkeypatch.setattr(setup,'ROOT',tmp_path)
    monkeypatch.setattr(sys,'argv',['tokenizer','--help'])
    downloads=[]
    monkeypatch.setattr(setup,'install',lambda *a:downloads.append(1))
    with pytest.raises(SystemExit):setup.main()
    assert not downloads and list((tmp_path/'logs').glob('*.log'))


def test_end_to_end_experiment_uses_same_archive_independent_quizzes_and_persists_calls(tmp_path,monkeypatch):
    import agent9.experiment as experiment
    from agent9.transport import Client,MODEL
    from agent9.metrics import TokenCounter
    from agent9.memory import Memory,context
    class Counter(TokenCounter):
        def __init__(self,*a):super().__init__();self.exact_text=True
    monkeypatch.setattr(experiment,'TokenCounter',Counter)
    monkeypatch.setenv('POLZA_API_KEY','fake-key-for-test')
    monkeypatch.setattr(experiment,'preflight',lambda client:{'context_limit':40960})
    sent=[]
    def respond(request):
        payload=json.loads(request.content);sent.append(payload);last=payload['messages'][-1];name=None;args={}
        if last['role']=='user' and last['content']==experiment.PROMPTS[0]:name='find_order';args={'order_id':'A-104'}
        elif last['role']=='tool' and 'amount' in last['content']:name='calculate';args={'operation':'multiply','a':'2390','b':'0.85'}
        message={'role':'assistant','content':'Принято.'}
        if 'tools' not in payload:message['content']='Маяк Ирина SQLite Россия 2026-12-18 210000 персональные данные локальный веб-чат 2390 2031.50'
        for fact in experiment.FACTS:
            if last['role']=='user' and last['content'].startswith(fact['question']):
                message['content']=json.dumps({'value':fact['answers'][0]},ensure_ascii=False)
        if name:message.update(content=None,tool_calls=[{'id':name,'type':'function','function':{'name':name,'arguments':json.dumps(args)}}])
        return httpx.Response(200,json={'model':MODEL,'choices':[{'finish_reason':'tool_calls' if name else 'stop','message':message}],
            'usage':{'prompt_tokens':100,'completion_tokens':10,'total_tokens':110,'cost_rub':.01}})
    original=Client
    with Journal(tmp_path/'http-log') as log,httpx.Client(transport=httpx.MockTransport(respond)) as http:
        monkeypatch.setattr(experiment,'Client',lambda key,journal,unused,**kwargs:original(key,journal,http,**kwargs))
        directory=experiment.run(tmp_path)
    result=json.loads((directory/'experiment.json').read_text())
    assert result['status']=='complete',result.get('error')
    assert len(result['archive'])==28 and result['restart_verified']
    assert len(result['summaries'])==5 and result['memory']['covered']==22
    assert len(result['calls'])==39 and len(result['quizzes'])==20
    assert all(row['passed'] for row in result['quizzes'])
    quiz_requests=[p for p in sent if p['messages'][-1].get('content','').endswith('value="неизвестно".')]
    memory=Memory(**result['memory'])
    for index,payload in enumerate(quiz_requests):
        expected=result['archive'] if index%2==0 else context(result['archive'],memory)
        assert payload['messages'][1:-1]==expected
    assert result['budget']['accounted_rub']<20
