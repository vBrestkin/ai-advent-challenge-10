"""Локальный веб-чат. Сессии живут в памяти процесса до задания дня 7."""
from contextlib import asynccontextmanager
from pathlib import Path
import os,threading
from uuid import uuid4
import httpx
from dotenv import dotenv_values
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel,Field
from .agent import Agent,AgentError
from .transport import Client
from .log import Journal

ROOT=Path(__file__).resolve().parents[2]

class ChatRequest(BaseModel):
    message:str=Field(min_length=1,max_length=20000)
    session_id:str=Field(default='',max_length=64)


def create_app(*,root=ROOT,model=None):
    root=Path(root)
    sessions={}
    lock=threading.Lock()
    @asynccontextmanager
    async def lifespan(app):
        settings={}
        repo=root.parents[1]
        for day in range(1,6): settings.update(dotenv_values(repo/'W1'/f'D{day}'/'.env',interpolate=False))
        settings.update(dotenv_values(root/'.env',interpolate=False)); settings.update(os.environ)
        key=(settings.get('POLZA_API_KEY') or '').strip()
        with Journal(root/'logs',key) as journal,httpx.Client(transport=httpx.HTTPTransport(retries=0)) as http:
            app.state.journal=journal
            app.state.model=model if model is not None else Client(key,journal,http)
            app.state.ready=model is not None or bool(key)
            yield
            journal.record('END','Остановка сервера')
    app=FastAPI(lifespan=lifespan)

    @app.middleware('http')
    async def journal_requests(request:Request,call_next):
        app.state.journal.record('HTTP',{'method':request.method,'path':request.url.path})
        response=await call_next(request)
        app.state.journal.record('HTTP_STATUS',response.status_code)
        return response

    @app.get('/',response_class=HTMLResponse)
    def index():return Path(__file__).with_name('index.html').read_text(encoding='utf-8')

    @app.get('/api/sessions')
    def list_sessions():
        with lock: return [{'id':sid,'title':item['title']} for sid,item in sessions.items()]

    @app.get('/api/sessions/{session_id}')
    def history(session_id:str):
        with lock:
            if session_id not in sessions: raise HTTPException(404,'Разговор не найден.')
            return {'turns':sessions[session_id]['turns']}

    @app.post('/api/chat')
    def chat(data:ChatRequest):
        if not data.message.strip(): raise HTTPException(422,'Введите сообщение.')
        if not app.state.ready: raise HTTPException(503,'Задайте POLZA_API_KEY в .env.')
        # Один выполняемый ход на процесс исключает гонки истории и перемешивание журналов.
        if not lock.acquire(blocking=False): raise HTTPException(409,'Агент занят. Дождитесь ответа.')
        try:
            sid=data.session_id or uuid4().hex
            if data.session_id and sid not in sessions: raise HTTPException(404,'Разговор не найден.')
            if sid not in sessions:
                sessions[sid]={'agent':Agent(app.state.model),'title':data.message[:50],'turns':[]}
            app.state.journal.record('INPUT',{'session_id':sid,'message':data.message})
            try: result=sessions[sid]['agent'].run(data.message)
            except AgentError as exc:
                app.state.journal.record('ERROR',str(exc))
                raise HTTPException(502,str(exc)) from None
            except OSError:
                raise HTTPException(503,'Не удалось записать журнал. Работа остановлена.') from None
            result.pop('messages')
            result=app.state.journal.scrub(result)
            turn={'user':app.state.journal.scrub(data.message),**result}
            sessions[sid]['turns'].append(turn)
            app.state.journal.record('OUTPUT',turn)
            return {'session_id':sid,**result}
        finally: lock.release()
    return app


def main():
    import uvicorn
    uvicorn.run(create_app(),host='127.0.0.1',port=8060,access_log=False)
