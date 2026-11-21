from copy import deepcopy
from agent11.restart_pilot import verify


def sample():
    before={'messages':[{'role':'user','content':'Проект Маяк, менеджер Ирина, заказ A-104'},
        {'role':'assistant','content':None,'tool_calls':[{'id':'x'}]},
        {'role':'tool','tool_call_id':'x','content':'{"amount":"2390","currency":"RUB"}'},
        {'role':'assistant','content':'2390 RUB'}], 'turns':[{}]}
    after=deepcopy(before)
    after['messages'] += [{'role':'user','content':'Напомни'},
        {'role':'assistant','content':'Проект Маяк, менеджер Ирина, сумма 2390 RUB'}]
    after['turns'].append({})
    return before,deepcopy(before),after


def test_probe_confirms_exact_history_and_recalled_details():
    assert all(verify(*sample()).values())


def test_probe_detects_lost_metadata_and_missing_detail():
    before,restored,after=sample()
    restored['messages'][1]['tool_calls'][0]['id']='changed'
    after['messages'][-1]['content']='Проект Маяк'
    result=verify(before,restored,after)
    assert not result['history_restored_exactly'] and not result['details_recalled']
