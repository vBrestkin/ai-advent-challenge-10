"""Веб-чат с полным контекстом в SQLite между запусками процесса."""
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
import os
import threading
from typing import Literal
from .memory import Memory, compact, context, KEEP, TRIGGER
from .strategies import Facts,select_context,update_facts

import httpx
from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .agent import Agent, AgentError, SYSTEM
from .transport import Client, MODEL, PROVIDER, PARAMETERS, preflight
from .metrics import TokenCounter, aggregate
from .tools import TOOLS
from .log import Journal, LoggingParser
from .store import Store, StoreError

ROOT = Path(__file__).resolve().parents[2]
PROFILE = {'version': 1, 'model': MODEL, 'provider': PROVIDER,
           'system': SYSTEM, 'tools': TOOLS,
           'parameters': PARAMETERS}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500000)
    session_id: str = Field(default='', max_length=64)
    mode: Literal['full','summary','window','facts'] = 'full'


class CheckpointRequest(BaseModel):
    session_id: str = Field(min_length=1,max_length=64)
    title: str = Field(default='Checkpoint',min_length=1,max_length=120)

class ForkRequest(BaseModel):
    checkpoint_id: str = Field(min_length=1,max_length=64)
    title: str = Field(min_length=1,max_length=120)

class EstimateRequest(ChatRequest):
    message: str = Field(default='', max_length=500000)


def load_settings(root):
    settings = {}
    repo = root.parents[1]
    for day in range(1, 6):
        settings.update(dotenv_values(repo / 'W1' / f'D{day}' / '.env', interpolate=False))
    settings.update(dotenv_values(root.parent / 'D1' / '.env', interpolate=False))
    settings.update(dotenv_values(root.parent / 'D2' / '.env', interpolate=False))
    settings.update(dotenv_values(root.parent / 'D3' / '.env', interpolate=False))
    settings.update(dotenv_values(root.parent / 'D4' / '.env', interpolate=False))
    settings.update(dotenv_values(root / '.env', interpolate=False))
    settings.update(os.environ)
    return settings


def create_app(*, root=ROOT, model=None, data_dir=None, budget=None, allow_overflow=False):
    root = Path(root)
    directory = Path(data_dir or os.environ.get('AGENT10_DATA_DIR') or root / 'data')
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        settings = load_settings(root)
        key = (settings.get('POLZA_API_KEY') or '').strip()
        with Journal(root / 'logs', key) as journal:
            app.state.journal = journal
            try:
                with httpx.Client(transport=httpx.HTTPTransport(retries=0)) as http, Store(directory / 'history.sqlite3') as store:
                    counter = TokenCounter(root / 'data' / 'tokenizer.json')
                    app.state.counter = counter
                    app.state.model = model if model is not None else Client(key, journal, http, counter=counter, budget=budget, allow_overflow=allow_overflow)
                    app.state.catalog = None
                    if model is None and key:
                        app.state.catalog = preflight(app.state.model)
                    app.state.ready = model is not None or bool(key)
                    app.state.store = store
                    journal.record('RESTORE', {'database': str(store.path), 'sessions': len(store.list())})
                    yield
            except StoreError as exc:
                journal.record('STORE_ERROR', str(exc))
                raise
            finally:
                journal.record('END', 'Остановка сервера')

    app = FastAPI(lifespan=lifespan)

    @app.middleware('http')
    async def journal_requests(request: Request, call_next):
        app.state.journal.record('HTTP', {'method': request.method, 'path': request.url.path})
        response = await call_next(request)
        app.state.journal.record('HTTP_STATUS', response.status_code)
        return response

    @app.get('/', response_class=HTMLResponse)
    def index():
        return Path(__file__).with_name('index.html').read_text(encoding='utf-8')

    @app.get('/health')
    def health():
        return {'ready': app.state.ready, 'model': MODEL, 'provider': PROVIDER, 'context_limit': getattr(app.state.model, 'context_limit', 40960), 'counter_method': app.state.counter.method}

    @app.get('/api/sessions')
    def list_sessions():
        try:
            return app.state.store.list()
        except StoreError as exc:
            raise HTTPException(503, str(exc)) from None

    @app.get('/api/sessions/{session_id}')
    def history(session_id: str, mode: Literal['full','summary','window','facts']='summary'):
        try:
            row = app.state.store.load(session_id)
            if row is None:
                raise HTTPException(404, 'Разговор не найден.')
            return {'title':row['title'], 'turns': row['turns'], 'messages': row['messages'],
                    'attempts': row['attempts'], 'revision': row['revision'],
                    'calls': row['calls'], 'metrics': aggregate(row['calls']),
                    'memory': row['memory'], 'facts_state':row['facts_state'], 'sent_context': [{'role':'system','content':SYSTEM}] + select_context(row['messages'],mode,Memory(**row['memory']),Facts(**row['facts_state'])),
                    'summary_metrics': aggregate([call for call in row['calls'] if call.get('purpose')=='summary']),
                    'facts_metrics':aggregate([call for call in row['calls'] if call.get('purpose')=='facts'])}
        except StoreError as exc:
            raise HTTPException(503, str(exc)) from None

    @app.get('/api/checkpoints')
    def checkpoints():
        try:return app.state.store.list_checkpoints()
        except StoreError as exc:raise HTTPException(503,str(exc)) from None

    @app.post('/api/checkpoints')
    def save_checkpoint(data: CheckpointRequest):
        if not lock.acquire(blocking=False):raise HTTPException(409,'Агент занят. Дождитесь ответа.')
        try:
            result=app.state.store.save_checkpoint(data.session_id,app.state.journal.scrub(data.title))
            app.state.journal.record('CHECKPOINT',{'session_id':data.session_id,'checkpoint_id':result})
            return {'checkpoint_id':result}
        except StoreError as exc:raise HTTPException(503,str(exc)) from None
        finally:lock.release()

    @app.post('/api/forks')
    def fork(data: ForkRequest):
        if not lock.acquire(blocking=False):raise HTTPException(409,'Агент занят. Дождитесь ответа.')
        try:
            result=app.state.store.fork(data.checkpoint_id,app.state.journal.scrub(data.title))
            app.state.journal.record('FORK',{'checkpoint_id':data.checkpoint_id,'session_id':result})
            return {'session_id':result}
        except StoreError as exc:raise HTTPException(503,str(exc)) from None
        finally:lock.release()

    @app.post('/api/estimate')
    def estimate(data: EstimateRequest):
        try:
            row = app.state.store.load(data.session_id) if data.session_id else None
            if data.session_id and row is None:
                raise HTTPException(404, 'Разговор не найден.')
            messages = row['messages'] if row else []
            counter = app.state.counter
            selected = select_context(messages,data.mode,Memory(**row['memory']) if row else Memory(),Facts(**row['facts_state']) if row else Facts())
            full = [{'role': 'system', 'content': SYSTEM}] + selected + [{'role': 'user', 'content': data.message}]
            predicted = counter.estimate(full, TOOLS)
            limit = getattr(app.state.model, 'context_limit', 40960)
            return {'new_message_tokens_local': counter.count(data.message),
                    'history_tokens_estimate': counter.estimate(messages, []) if messages else 0,
                    'request_tokens_estimate': predicted,
                    'output_reserve': PARAMETERS['max_tokens'], 'context_limit': limit,
                    'fits_estimate': predicted + PARAMETERS['max_tokens'] <= limit,
                    'counter_method': counter.method, 'local_tokenizer': counter.exact_text}
        except StoreError as exc:
            raise HTTPException(503, str(exc)) from None

    @app.post('/api/chat')
    def chat(data: ChatRequest):
        if not data.message.strip():
            raise HTTPException(422, 'Введите сообщение.')
        if not app.state.ready:
            raise HTTPException(503, 'Задайте POLZA_API_KEY в .env.')
        if not lock.acquire(blocking=False):
            raise HTTPException(409, 'Агент занят. Дождитесь ответа.')
        attempt_id = None
        store = app.state.store
        journal = app.state.journal
        try:
            prompt = journal.scrub(data.message)
            session_id = data.session_id
            if not session_id:
                session_id = store.create(prompt[:50], PROFILE)
            row = store.load(session_id)
            if row is None:
                raise HTTPException(404, 'Разговор не найден.')
            if row['profile'] != PROFILE:
                raise HTTPException(409, 'Профиль разговора изменился. Используйте прежнюю версию приложения или новый разговор.')
            memory = Memory(**row['memory'])
            facts = Facts(**row['facts_state'])
            journal.record('INPUT', {'session_id': session_id, 'message': prompt})
            attempt_id = store.begin(session_id, prompt)
            def observe(event, record, call_id):
                if event == 'start':
                    return store.begin_call(attempt_id, journal.scrub(record))
                store.finish_call(call_id, journal.scrub(record))

            if isinstance(app.state.model, Client):
                app.state.model.observer = observe
            try:
                if data.mode=='summary':
                    memory=compact(row['messages'],memory,app.state.model)
                if data.mode=='facts':
                    facts=update_facts(row['messages']+[{'role':'user','content':prompt}],facts,app.state.model)
                agent=Agent(app.state.model,context_builder=lambda messages:select_context(messages,data.mode,memory,facts))
                agent.messages=deepcopy(row['messages'])
                result = agent.run(prompt, checkpoint=lambda messages: store.checkpoint(attempt_id, journal.scrub(messages)))
                result['mode'] = data.mode
                result['compression_warning'] = ''
                if data.mode == 'summary':
                    try:
                        memory = compact(result['messages'],memory,app.state.model)
                    except AgentError as exc:
                        result['compression_warning'] = str(exc)
                        journal.record('COMPRESSION_ERROR', str(exc))
                result['memory'] = memory.to_dict()
                result['facts_state'] = facts.to_dict()
            finally:
                if isinstance(app.state.model, Client):
                    app.state.model.observer = None
            calls = [call for call in store.load(session_id)['calls'] if call['attempt_id'] == attempt_id]
            result['metrics'] = aggregate(calls)
            result['history_tokens_estimate'] = app.state.counter.estimate(result['messages'], [])
            result['counter_method'] = app.state.counter.method
            messages = journal.scrub(result.pop('messages'))
            result = journal.scrub(result)
            turn = {'user': prompt, **result}
            # Единственная транзакция фиксирует и протокол для LLM, и представление для интерфейса.
            store.complete(attempt_id, messages, turn, memory=journal.scrub(memory.to_dict()),facts_state=journal.scrub(facts.to_dict()))
            journal.record('HISTORY_SAVED', {'session_id': session_id, 'attempt_id': attempt_id})
            journal.record('OUTPUT', turn)
            return {'session_id': session_id, **result}
        except (AgentError, StoreError, OSError) as exc:
            message = str(exc) if isinstance(exc, (AgentError, StoreError)) else 'Журнал или история недоступны. Работа остановлена.'
            if attempt_id is not None:
                try:
                    store.fail(attempt_id, message)
                except StoreError:
                    pass
            try:
                journal.record('ERROR', message)
            except OSError:
                pass
            raise HTTPException(502 if isinstance(exc, AgentError) else 503, message, headers={'X-Session-ID': session_id}) from None
        finally:
            lock.release()
    return app


def main():
    import uvicorn
    key = (load_settings(ROOT).get('POLZA_API_KEY') or '').strip()
    with Journal(ROOT / 'logs', key) as journal:
        parser = LoggingParser(journal=journal, description='День 10. Стратегии контекста и ветвление диалога')
        parser.add_argument('--port', type=int, default=8064)
        parser.add_argument('--data-dir', type=Path)
        try:
            args = parser.parse_args()
            if not 1 <= args.port <= 65535:
                parser.error('Порт должен быть от 1 до 65535.')
            journal.record('CLI_CONFIG', {'port': args.port, 'data_dir': str(args.data_dir or ROOT / 'data')})
            uvicorn.run(create_app(root=ROOT, data_dir=args.data_dir), host='127.0.0.1', port=args.port, access_log=False)
        finally:
            journal.record('CLI_END', 'Команда завершена')


if __name__ == '__main__':
    main()
