import hashlib,json
import httpx,pytest
from agent15.setup_tokenizer import install
from agent15.experiment import write_report
from agent15.log import Journal


def test_tokenizer_install_is_atomic_and_checks_hash(tmp_path,monkeypatch):
    payload=b'{"tokenizer":"test"}'
    monkeypatch.setattr('agent15.setup_tokenizer.TOKENIZER_SHA256',hashlib.sha256(payload).hexdigest())
    path=tmp_path/'tokenizer.json'
    with Journal(tmp_path/'log') as log,httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,content=payload))) as http:
        install(path,http,log)
        assert path.read_bytes()==payload
        monkeypatch.setattr('agent15.setup_tokenizer.TOKENIZER_SHA256','wrong')
        with pytest.raises(ValueError):install(path,http,log)
        assert path.read_bytes()==payload


def test_grader_requires_json_and_current_value():
    from agent15.experiment import grade
    assert grade('{"value":"210000"}', ['210000'])
    assert not grade('{"value":"180000"}', ['210000'])
    assert not grade('210000', ['210000'])
    assert not grade('Возможно 180000 или 210000', ['210000'])


def test_scenario_and_facts_are_fixed_before_model_calls():
    from agent15.experiment import PROMPTS, FACTS
    assert len(PROMPTS)==12 and len(FACTS)==10
    assert all(set(fact)=={'id','question','answers'} for fact in FACTS)
    assert '210000' in str(FACTS)


def test_experiment_help_is_logged_before_exit(tmp_path,monkeypatch):
    import sys
    import agent15.experiment as experiment
    monkeypatch.setattr(experiment,'ROOT',tmp_path)
    monkeypatch.setattr(sys,'argv',['experiment','--help'])
    with pytest.raises(SystemExit):experiment.main()
    assert list((tmp_path/'logs').glob('*.log'))


def test_tokenizer_help_does_not_download_and_is_logged(tmp_path,monkeypatch):
    import sys
    import agent15.setup_tokenizer as setup
    monkeypatch.setattr(setup,'ROOT',tmp_path)
    monkeypatch.setattr(sys,'argv',['tokenizer','--help'])
    downloads=[]
    monkeypatch.setattr(setup,'install',lambda *a:downloads.append(1))
    with pytest.raises(SystemExit):setup.main()
    assert not downloads and list((tmp_path/'logs').glob('*.log'))


def test_summary_report_accounts_for_each_strategy_and_unknown_cost(tmp_path):
    from agent15.experiment import write_report
    data={'status':'complete','calls':[],'quizzes':[],'archive':[],'states':{},'context_sizes':{},'branches':{},
          'budget':{'known_cost_rub':0,'unknown_reserve_rub':0,'accounted_rub':0,'limit_rub':20}}
    write_report(tmp_path,data)
    text=(tmp_path/'report.md').read_text()
    assert all(mode in text for mode in ('window','facts','summary'))
    assert (tmp_path/'summary.csv').exists() and 'обслуживания' in text


def test_strategy_code_measurement_is_explicit():
    from agent15.experiment import complexity
    rows=complexity()
    assert set(rows)=={'window','facts','summary'}
    assert rows['window']['nonblank_lines']<rows['facts']['nonblank_lines']




def test_complete_strategy_experiment_with_mock_http_provider(tmp_path,monkeypatch):
    from copy import deepcopy
    import re
    import agent15.experiment as experiment
    from agent15.transport import Client,MODEL
    from agent15.metrics import TokenCounter
    from test_compression import tool_turn
    first=tool_turn();first[0]['content']=experiment.PROMPTS[0]
    archive=first+[m for p in experiment.PROMPTS[1:] for m in ({'role':'user','content':p},{'role':'assistant','content':'ok'})]
    source=tmp_path/'source.json';source.write_text(json.dumps({'archive':archive,'facts':experiment.FACTS}))
    class Counter(TokenCounter):
        def __init__(self,*a):super().__init__();self.exact_text=True
    monkeypatch.setattr(experiment,'TokenCounter',Counter);monkeypatch.setenv('POLZA_API_KEY','fake-test-key')
    monkeypatch.setattr(experiment,'preflight',lambda client:{'context_limit':40960})
    sent=[]
    def provider(request):
        payload=json.loads(request.content);sent.append(payload);messages=payload['messages'];last=messages[-1]['content'];answer='ok'
        if 'tools' not in payload:
            data=json.loads(last)
            if 'previous_facts' in data:
                values=deepcopy(data['previous_facts']) or {'project':'Маяк','budget_rub':'210000','deadline':'2026-12-18'}
                for message in data['new_messages']:
                    if message['role']=='user' and message['content'].startswith('Только для этой ветки:'):
                        values['budget_rub']=re.search(r'бюджет теперь (\d+)',message['content'])[1]
                        values['deadline']=re.search(r'\d{4}-\d{2}-\d{2}',message['content'])[0]
                answer=json.dumps({'facts':values},ensure_ascii=False)
            else:answer='Маяк, Ирина, 210000, 2026-12-18.'
        elif last.startswith('Каковы актуальные бюджет'):
            block=next(m['content'].split('Facts JSON: ',1)[1] for m in messages if 'Facts JSON: ' in str(m.get('content')))
            values=json.loads(block);answer=json.dumps({k:values[k] for k in ('budget_rub','deadline')})
        else:
            for fact in experiment.FACTS:
                if last.startswith(fact['question']):answer=json.dumps({'value':fact['answers'][0]},ensure_ascii=False)
        return httpx.Response(200,json={'model':MODEL,'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':answer}}],
            'usage':{'prompt_tokens':100,'completion_tokens':10,'total_tokens':110,'cost_rub':.01}})
    real=Client
    with httpx.Client(transport=httpx.MockTransport(provider)) as http:
        monkeypatch.setattr(experiment,'Client',lambda key,journal,unused,**kwargs:real(key,journal,http,**kwargs))
        directory=experiment.run(tmp_path,source_path=source)
    result=json.loads((directory/'experiment.json').read_text())
    assert result['status']=='complete',result.get('error')
    assert len(result['calls'])==len(sent)==65 and len(result['quizzes'])==30
    assert result['restart_verified'] and result['branches']['restart_verified'] and result['branches']['parent_unchanged']
    assert all(v['passed'] for v in result['branches']['variants'].values())
    assert sum(c['purpose']=='facts' and c['phase']=='quiz' for c in result['calls'])==10
    # Every quiz starts from the same source, no previous quiz answer in its dialogue.
    quiz_calls=[p for p in sent if p['messages'][-1]['content'].endswith('value="неизвестно".') and 'tools' in p]
    assert len(quiz_calls)==30
    for payload in quiz_calls:
        assert sum(m.get('content','').endswith('value="неизвестно".') for m in payload['messages'] if isinstance(m.get('content'),str))==1
    trace=json.loads((directory/'trace.json').read_text())
    assert sum(row['event']=='FORK' for row in trace)==2
    assert sum(row['event']=='CHECKPOINT' for row in trace)==1
    assert any(row['event']=='SOURCE_IMPORT' for row in trace)
    assert result['budget']['unknown_reserve_rub']==0 and result['budget']['accounted_rub']<20


def test_absent_maintenance_is_zero_but_unknown_usage_is_not():
    from agent15.experiment import summary_rows
    rows=summary_rows({'quizzes':[],'calls':[{'mode':'facts','purpose':'facts','phase':'build','sent':True,'cost_rub':None,'total_tokens':None}]})
    by_mode={row['mode']:row for row in rows}
    assert by_mode['window']['maintenance_cost_rub_known']==0
    assert by_mode['window']['maintenance_input_tokens']==0
    assert by_mode['facts']['maintenance_cost_rub_known'] is None and by_mode['facts']['unknown_costs']==1
