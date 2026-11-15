"""Три реальные проверки через тот же HTTP-интерфейс, что использует браузер."""
import json
from pathlib import Path
from fastapi.testclient import TestClient
from .web import create_app,ROOT

CASES=[('Без инструментов','Объясни, что такое JSON, в одном предложении.',[],None),
       ('Один инструмент','Посчитай 347 умножить на 29. Используй калькулятор.',['calculate'],'10063'),
       ('Два инструмента подряд','Найди сумму заказа A-104 и рассчитай цену со скидкой 15%.',['find_order','calculate'],'2031.5')]


def run_checks(client):
    results=[]
    for title,prompt,names,number in CASES:
        response=client.post('/api/chat',json={'message':prompt})
        data=response.json()
        observed=[s['tool'] for s in data.get('steps',[])]
        answer=data.get('answer','').replace(',','.').replace(' ','').replace('\u00a0','')
        passed=response.status_code==200 and observed==names and (number is None or number in answer)
        results.append({'case':title,'prompt':prompt,'http_status':response.status_code,'passed':passed,'response':data})
    return results


def main():
    app=create_app()
    with TestClient(app) as client:
        results=run_checks(client)
        usage=getattr(app.state.model,'usage',[])
        report={'verified':all(r['passed'] for r in results),'cases':results,'usage':usage,
                'cost_rub_known':sum(u['cost_rub'] for u in usage if type(u.get('cost_rub')) in (int,float)),
                'unknown_costs':sum(u.get('cost_rub') is None for u in usage)}
        directory=ROOT/'submission';directory.mkdir(exist_ok=True)
        report=app.state.journal.scrub(report)
        (directory/'pilot.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        # Учебные запросы пилота фиксированы. Эта копия не содержит личных разговоров.
        trace=[]
        for line in app.state.journal.path.read_text(encoding='utf-8').splitlines():
            timestamp,event,payload=line.split(' ',2)
            trace.append({'timestamp':timestamp,'event':event,'data':json.loads(payload)})
        (directory/'trace.json').write_text(json.dumps(trace,ensure_ascii=False,indent=2),encoding='utf-8')
        app.state.journal.record('PILOT',report)
        print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['verified'] else 1

if __name__=='__main__':raise SystemExit(main())
