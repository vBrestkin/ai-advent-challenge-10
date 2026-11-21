from fastapi.testclient import TestClient
from agent14.task_experiment import run_scenario
from agent14.web import create_app


def test_experiment_reopens_database_and_reaches_done(tmp_path):
    class Model:
        records=[]
        def send(self, messages):
            return {'role':'assistant','content':'Текущий этап сохранён.'}
    model=Model()
    events=[]
    def app():
        return TestClient(create_app(root=tmp_path,model=model))
    sid=run_scenario(app,model,events.append)
    assert len([e for e in events if e['kind']=='chat'])==5
    restored=next(e for e in events if e['kind']=='restart')
    assert restored['before']==restored['after'] and restored['after']['paused']
    assert all(e['before']==e['after'] for e in events if e['status']==409)
    with app() as web:
        assert web.get(f'/api/layers/{sid}/task').json()['stage']=='done'
