"""Веб-чат с полным контекстом в SQLite между запусками процесса."""
import argparse
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
import os
import threading

import httpx
from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .agent import Agent, AgentError, SYSTEM
from .transport import Client, MODEL
from .tools import TOOLS
from .log import Journal
from .store import Store, StoreError

ROOT = Path(__file__).resolve().parents[2]
PROFILE = {'version': 1, 'model': MODEL, 'provider': 'google-ai-studio/flex',
           'system': SYSTEM, 'tools': TOOLS,
           'parameters': {'max_tokens': 4096, 'temperature': .2,
                          'reasoning': {'enabled': True, 'max_tokens': 1024}}}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20000)
    session_id: str = Field(default='', max_length=64)


def load_settings(root):
    settings = {}
    repo = root.parents[1]
    for day in range(1, 6):
        settings.update(dotenv_values(repo / 'W1' / f'D{day}' / '.env', interpolate=False))
    settings.update(dotenv_values(root.parent / 'D1' / '.env', interpolate=False))
    settings.update(dotenv_values(root / '.env', interpolate=False))
    settings.update(os.environ)
    return settings


def create_app(*, root=ROOT, model=None, data_dir=None):
    root = Path(root)
    directory = Path(data_dir or os.environ.get('AGENT7_DATA_DIR') or root / 'data')
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        settings = load_settings(root)
        key = (settings.get('POLZA_API_KEY') or '').strip()
        with Journal(root / 'logs', key) as journal:
            app.state.journal = journal
            try:
                with httpx.Client(transport=httpx.HTTPTransport(retries=0)) as http, Store(directory / 'history.sqlite3') as store:
                    app.state.model = model if model is not None else Client(key, journal, http)
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
        return {'ready': app.state.ready}

    @app.get('/api/sessions')
    def list_sessions():
        try:
            return app.state.store.list()
        except StoreError as exc:
            raise HTTPException(503, str(exc)) from None

    @app.get('/api/sessions/{session_id}')
    def history(session_id: str):
        try:
            row = app.state.store.load(session_id)
            if row is None:
                raise HTTPException(404, 'Разговор не найден.')
            return {'turns': row['turns'], 'messages': row['messages'],
                    'attempts': row['attempts'], 'revision': row['revision']}
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
            agent = Agent(app.state.model)
            agent.messages = deepcopy(row['messages'])
            journal.record('INPUT', {'session_id': session_id, 'message': prompt})
            attempt_id = store.begin(session_id, prompt)
            result = agent.run(prompt, checkpoint=lambda messages: store.checkpoint(attempt_id, journal.scrub(messages)))
            messages = journal.scrub(result.pop('messages'))
            result = journal.scrub(result)
            turn = {'user': prompt, **result}
            # Единственная транзакция фиксирует и протокол для LLM, и представление для интерфейса.
            store.complete(attempt_id, messages, turn)
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
            raise HTTPException(502 if isinstance(exc, AgentError) else 503, message) from None
        finally:
            lock.release()
    return app


def main():
    import uvicorn
    key = (load_settings(ROOT).get('POLZA_API_KEY') or '').strip()
    with Journal(ROOT / 'logs', key) as journal:
        class Parser(argparse.ArgumentParser):
            def _print_message(self, message, file=None):
                safe = journal.scrub(message or '')
                journal.record('CLI_OUTPUT', safe)
                super()._print_message(safe, file)

        parser = Parser(description='День 7. Диалог с восстановлением контекста SQLite')
        parser.add_argument('--port', type=int, default=8061)
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
