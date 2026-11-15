import json
import pytest
from pytest_bdd import scenarios, given, when, then, parsers
from agent6.agent import Agent, AgentError
from agent6.tools import execute
scenarios('features/agent.feature')

class Fake:
    def __init__(self, rounds): self.rounds,self.calls=rounds,[]
    def send(self,messages):
        self.calls.append(json.loads(json.dumps(messages)))
        n=len(self.calls)
        if n<=self.rounds:
            name,args=('find_order',{'order_id':'A-104'}) if n==1 else ('calculate',{'operation':'multiply','a':'2390','b':'0.85'})
            return {'role':'assistant','content':None,'tool_calls':[{'id':f'c{n}','type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}
        return {'role':'assistant','content':'Готово: 2031.50'}

@given(parsers.parse('a model that requests {rounds:d} tool rounds'), target_fixture='model')
def model(rounds): return Fake(rounds)
@when('the agent receives a question',target_fixture='result')
def run(model): return Agent(model).run('Рассчитай скидку')
@then(parsers.parse('it returns a final answer after {rounds:d} tool rounds'))
def check(model,result,rounds):
    assert result['answer']=='Готово: 2031.50'
    assert len(result['steps'])==rounds and len(model.calls)==rounds+1
    if rounds==2:
        assert model.calls[1][-1]['role']=='tool' and '2390' in model.calls[1][-1]['content']
        assert model.calls[2][-1]['tool_call_id']=='c2'

@pytest.mark.parametrize('name,args',[
    ('unknown',{}),('find_order',{'order_id':'missing'}),
    ('calculate',{'operation':'divide','a':'1','b':'0'}),
    ('calculate',{'operation':'add','a':'NaN','b':'1'}),
    ('calculate',{'operation':'add','a':True,'b':'1'}),
    ('calculate',{'operation':'evil','a':'1','b':'2'}),
    ('find_order',{'order_id':'A-104','extra':1})])
def test_invalid_tool_returns_error(name,args): assert 'error' in execute(name,args)

def test_tools():
    assert execute('find_order',{'order_id':'A-104'})['amount']=='2390'
    assert execute('calculate',{'operation':'multiply','a':'347','b':'29'})['result']=='10063'

def test_limit_and_history_rollback():
    a=Agent(Fake(100),max_rounds=2)
    with pytest.raises(AgentError): a.run('hello')
    assert a.messages==[]

def test_history_keeps_tool_protocol():
    model=Fake(1); a=Agent(model); a.run('one'); a.run('two')
    assert any(m['role']=='tool' for m in model.calls[-1])
    assert model.calls[-1][-1]['content']=='two'

def test_network_failure_rolls_back():
    class Broken:
        def send(self,messages): raise AgentError('network')
    a=Agent(Broken())
    with pytest.raises(AgentError): a.run('hi')
    assert a.messages==[]

def test_repeated_call_id_rejected():
    class Repeated(Fake):
        def send(self,m):
            r=super().send(m)
            if r.get('tool_calls'): r['tool_calls'][0]['id']='same'
            return r
    with pytest.raises(AgentError): Agent(Repeated(2)).run('hi')

def test_extreme_decimal_exponent_rejected():
    assert 'error' in execute('calculate',{'operation':'add','a':'1e-999999','b':'0'})

def test_failure_after_tool_keeps_previous_history():
    class Broken(Fake):
        def send(self,messages):
            if self.calls: raise AgentError('network')
            return super().send(messages)
    a=Agent(Broken(1))
    with pytest.raises(AgentError):a.run('hi')
    assert a.messages==[]
