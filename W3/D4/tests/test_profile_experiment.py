import pytest
from fastapi.testclient import TestClient
from agent14.profile_experiment import assess, first_violation, run_matrix
from agent14.web import create_app


@pytest.mark.parametrize('length,words,expected', [('brief',45,True),('brief',46,False),
                                                    ('detailed',99,False),('detailed',100,True)])
def test_prose_thresholds_exclude_fenced_code(length,words,expected):
    answer='слово '*words+'\n```python\nprint("hi")\n```\n'
    result=assess(answer, {'length':length,'code':'with'})
    assert result['words']==words and result['code_ok'] and result['length_ok']==expected


def test_handling_of_first_violation_or_no_violation():
    assert first_violation([{'number':1,'passed':True},{'number':2,'passed':False}])==2
    assert first_violation([{'number':1,'passed':True}]) is None


def test_matrix_uses_identical_question_updates_profile_and_preserves_one_sequence(tmp_path):
    class Model:
        records=[]
        def send(self,messages):
            detailed=any('минимум 100 слов' in m.get('content','') for m in messages if m.get('role')=='system')
            answer='пример '*100+'\n```python\nprint(1)\n```\n' if detailed else 'Короткое объяснение.'
            return {'role':'assistant','content':answer}
    model=Model()
    rows=[]
    with TestClient(create_app(root=tmp_path,model=model)) as web:
        sessions=run_matrix(web,model,rows.append)
        assert len(rows)==14
        assert rows[0]['question']==rows[1]['question']
        assert rows[2]['profile_changed'] and rows[3]['passed']
        series=[row for row in rows if row['phase']=='series']
        assert [row['number'] for row in series]==list(range(1,11))
        assert {row['id'] for row in series}=={sessions['series']}
        state=web.get(f"/api/layers/{sessions['series']}").json()
        assert len(state['memory']['short'])==20
