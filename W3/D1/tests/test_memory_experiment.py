import httpx
from agent11.memory_experiment import QUESTIONS, run_matrix
from agent11.log import Journal
from agent11.transport import Client
from agent11.web import create_app
from fastapi.testclient import TestClient


def test_matrix_controls_do_not_modify_state_and_measure_one_layer(tmp_path):
    import json
    seen = []
    def provider(request):
        payload = json.loads(request.content)
        seen.append(payload)
        question = payload['messages'][-1]['content']
        key, (text, expected) = next((key, item) for key, item in QUESTIONS.items() if item[0] in question)
        present = expected in str(payload['messages'][:-1])
        answer = json.dumps({'value': expected if present else 'неизвестно'})
        tokens = sum(len(m['content']) for m in payload['messages'])
        return httpx.Response(200, json={'model':payload['model'],'choices':[{'finish_reason':'stop','message':{'content':answer}}],
                                       'usage':{'prompt_tokens':tokens,'completion_tokens':10,'total_tokens':tokens+10,'cost_rub':.01}})
    with Journal(tmp_path/'transport') as journal, httpx.Client(transport=httpx.MockTransport(provider)) as http:
        model = Client('key', journal, http)
        with TestClient(create_app(root=tmp_path,model=model)) as web:
            rows = []
            states = run_matrix(web, model, {'project_name':'Маяк','project_leader':'Ирина'}, rows.append)
            assert states['all']['memory']['long']=={'project_name':'Маяк','project_leader':'Ирина'}
            assert len(states)==4 and len(rows)==12 and len(seen)==12
            assert all(row['passed'] and row['unchanged'] for row in rows)
            for row in rows:
                same = next(other for other in rows if other['mode']=='all' and other['question_id']==row['question_id'])
                if row['mode']!='all':
                    assert same['prompt_tokens']>row['prompt_tokens']
