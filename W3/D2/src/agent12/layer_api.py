"""Normal web entry points for memory workspaces."""
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Optional
from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from .agent import AgentError
from .store import StoreError
from .layers import Layers
from .profiles import Profile, preference_patch

LayerName = Literal['short', 'working', 'long']
CardName = Literal['working', 'long']


class CreateWorkspace(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    memory: Optional[dict] = None


class Message(BaseModel):
    message: str = Field(min_length=1, max_length=500000)


class Fact(BaseModel):
    key: str
    value: str


def register(app, lock):
    @contextmanager
    def write_access(request):
        if not lock.acquire(blocking=False):
            raise HTTPException(409, 'Дождитесь завершения текущего запроса.')
        try:
            yield Layers(request.app.state.store), request.app.state.journal
        except KeyError:
            raise HTTPException(404, 'Разговор со слоями не найден.') from None
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from None
        except (StoreError, OSError):
            raise HTTPException(503, 'Не удалось прочитать или сохранить память.') from None
        finally:
            lock.release()

    def state(layers, sid):
        row = layers.load(sid)
        if row is None:
            raise HTTPException(404, 'Разговор со слоями не найден.')
        return row

    @app.get('/', response_class=HTMLResponse)
    def interface():
        return Path(__file__).with_name('layers.html').read_text(encoding='utf-8')

    @app.get('/api/layers')
    def listing(request: Request):
        return Layers(request.app.state.store).list()

    @app.get('/api/layers/{sid}')
    def get(sid: str, request: Request):
        return state(Layers(request.app.state.store), sid)

    @app.post('/api/layers')
    def create(data: CreateWorkspace, request: Request):
        with write_access(request) as (layers, journal):
            sid = layers.create(journal.scrub(data.title), journal.scrub(data.memory))
            journal.record('LAYERS_CREATE', {'id': sid})
            return {'id': sid}

    @app.post('/api/layers/{sid}/copy')
    def copy(sid: str, request: Request):
        with write_access(request) as (layers, journal):
            source = state(layers, sid)
            result = layers.create('Копия: ' + source['title'][:100], source['memory'], Profile(**source['profile']))
            journal.record('LAYERS_COPY', {'source': sid, 'id': result})
            return {'id': result}

    @app.post('/api/layers/{sid}/clear/{layer}')
    def clear(sid: str, layer: LayerName, request: Request):
        with write_access(request) as (layers, journal):
            layers.clear(sid, layer)
            journal.record('LAYER_CLEAR', {'id': sid, 'layer': layer})
            return state(layers, sid)

    @app.post('/api/layers/{sid}/write/{layer}')
    def write(sid: str, layer: CardName, data: Fact, request: Request):
        with write_access(request) as (layers, journal):
            values = journal.scrub({data.key: data.value})
            layers.write(sid, layer, values)
            journal.record('LAYER_WRITE', {'id': sid, 'layer': layer, 'values': values})
            return state(layers, sid)

    @app.put('/api/layers/{sid}/profile')
    def update_profile(sid: str, data: Profile, request: Request):
        with write_access(request) as (layers, journal):
            before = state(layers, sid)['profile']
            layers.save_profile(sid, data)
            journal.record('PROFILE_UPDATE', {'id': sid, 'source': 'form', 'before': before, 'after': data.model_dump()})
            return state(layers, sid)

    @app.post('/api/layers/{sid}/chat')
    def chat(sid: str, data: Message, request: Request, save: bool = True):
        if not data.message.strip():
            raise HTTPException(422, 'Введите сообщение.')
        if not request.app.state.ready:
            raise HTTPException(503, 'Укажите POLZA_API_KEY.')
        with write_access(request) as (layers, journal):
            question = journal.scrub(data.message)
            journal.record('LAYER_INPUT', {'id': sid, 'message': question, 'save': save})
            try:
                row = state(layers, sid)
                before = row['profile']
                updated = Profile(**{**before, **preference_patch(question)})
                result = Layers.ask(request.app.state.model, row['memory'], question, updated)
                result = journal.scrub(result)
                messages = result.pop('messages')
                if save:
                    layers.save_dialogue(sid, messages, updated)
                changed = save and updated.model_dump() != before
                if changed:
                    journal.record('PROFILE_UPDATE', {'id': sid, 'source': 'dialogue', 'before': before, 'after': updated.model_dump()})
                result.update(profile_changed=changed, profile=updated.model_dump() if save else before)
            except AgentError as exc:
                journal.record('LAYER_ERROR', str(exc))
                raise HTTPException(502, str(exc)) from None
            journal.record('LAYER_OUTPUT', result)
            return result
