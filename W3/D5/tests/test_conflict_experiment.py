import json
from fastapi.testclient import TestClient
from agent15.conflict_experiment import run_scenario
from agent15.web import create_app


def test_four_changes_check_actual_model_requests(tmp_path):
    class Model:
        def __init__(self):
            self.requests=[]
            self.records=[]
        def send(self, messages, **kwargs):
            self.requests.append({'messages':messages})
            self.records.append({'status':'complete'})
            card=json.loads(messages[0]['content'].split('Актуальные факты: ',1)[1])
            key=next(k for k in card if k.casefold() in messages[-1]['content'].casefold())
            return {'content':card[key]}
    model=Model()
    events=[]
    def app():
        return TestClient(create_app(root=tmp_path,model=model))
    run_scenario(app,model,events.append,lambda:model.requests)
    checks=[row for row in events if row['kind']=='question']
    assert len(checks)==4 and all(row['correct'] and not row['stale_in_request'] for row in checks)
    assert events[-1]['working_after']=={} and events[-1]['facts_preserved']
