from fastapi.testclient import TestClient
from agent14.invariant_experiment import run_scenario
from agent14.web import create_app


def test_full_matrix_uses_same_rule_for_all_attempts(tmp_path):
    class Model:
        records=[]
        def send(self, messages, **kwargs):
            return {'content':'{"decision":"propose","database":"postgresql"}' if
                    'разрешена только PostgreSQL' in messages[0]['content'] else
                    '{"decision":"propose","database":"sqlite"}'}
    model=Model()
    events=[]
    def app():
        return TestClient(create_app(root=tmp_path,model=model))
    sid=run_scenario(app,model,events.append)
    attacks=[row for row in events if row['kind']=='challenge']
    assert len(attacks)==8 and all(row['before']==row['after'] for row in attacks)
    assert not any(row['passed'] for row in attacks)
    with app() as web:
        rule=web.get(f'/api/layers/{sid}/invariant').json()
        assert rule['database']=='postgresql' and len(rule['history'])==1
