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
from .tasks import Action, InvalidTransition, TaskStore
from .invariants import RuleConflict, RuleUpdate, Rules, stack_answer, stack_messages

LayerName = Literal['short', 'working', 'long']
CardName = Literal['working', 'long']


class CreateWorkspace(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    memory: Optional[dict] = None


class Message(BaseModel):
    message: str = Field(min_length=1, max_length=500000)


class TaskTitle(BaseModel):
    title: str = Field(min_length=1, max_length=120)


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
        except (InvalidTransition, RuleConflict) as exc:
            raise HTTPException(409, str(exc)) from None
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
        store = request.app.state.store
        row = state(Layers(store), sid)
        task = TaskStore(store).read(sid)
        return {**row, 'task': task.view() if task else None, 'invariant': Rules(store).read(sid)}

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
            database = Rules(layers.store).read(sid)['database']
            result = layers.create('Копия: ' + source['title'][:100], source['memory'],
                                   Profile(**source['profile']), approved_database=database)
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

    @app.get('/api/layers/{sid}/task')
    def read_task(sid: str, request: Request):
        task = TaskStore(request.app.state.store).read(sid)
        if task is None:
            raise HTTPException(404, 'Задача не найдена.')
        return task.view()

    @app.post('/api/layers/{sid}/task')
    def create_task(sid: str, data: TaskTitle, request: Request):
        with write_access(request) as (layers, journal):
            task = TaskStore(layers.store).create(sid, journal.scrub(data.title))
            journal.record('TASK_CREATED', {'id':sid,'state':task.view()})
            return task.view()

    @app.post('/api/layers/{sid}/task/actions')
    def change_task(sid: str, data: Action, request: Request):
        with write_access(request) as (layers, journal):
            tasks = TaskStore(layers.store)
            command = data.model_copy(update={'note':journal.scrub(data.note)})
            before = tasks.read(sid)
            try:
                after = tasks.apply(sid, command)
            except (InvalidTransition, ValueError) as exc:
                journal.record('TASK_REJECTED', {'id':sid,'action':command.action,
                    'state':before.view() if before else None,'reason':str(exc)})
                raise
            journal.record('TASK_CHANGED', {'id':sid,'action':command.action,
                'before':before.view() if before else None,'after':after.view()})
            return after.view()

    @app.get('/api/layers/{sid}/invariant')
    def read_invariant(sid: str, request: Request):
        try:
            return Rules(request.app.state.store).read(sid)
        except KeyError:
            raise HTTPException(404, 'Разговор не найден.') from None

    @app.put('/api/layers/{sid}/invariant')
    def replace_invariant(sid: str, data: RuleUpdate, request: Request):
        with write_access(request) as (layers, journal):
            rule = Rules(layers.store).replace(sid, data)
            if rule['revision'] != data.revision:
                journal.record('STACK_RULE_CHANGED', {'id':sid, **rule['history'][-1]})
            return rule

    @app.post('/api/layers/{sid}/stack')
    def suggest_stack(sid: str, data: Message, request: Request):
        if not data.message.strip():
            raise HTTPException(422, 'Введите запрос.')
        if not request.app.state.ready:
            raise HTTPException(503, 'Укажите POLZA_API_KEY.')
        with write_access(request) as (layers, journal):
            rule = Rules(layers.store).read(sid)
            question = journal.scrub(data.message)
            journal.record('STACK_INPUT', {'id':sid,'message':question,'database':rule['database']})
            try:
                reply = request.app.state.model.send(stack_messages(question,rule['database']),
                                                     purpose='stack',max_tokens=160)
            except AgentError:
                journal.record('STACK_ERROR', {'id':sid})
                raise HTTPException(502, 'Не удалось получить проверяемое предложение. Правило действует.') from None
            raw = reply.get('content') if not reply.get('tool_calls') else None
            output = stack_answer(raw, rule['database'])
            journal.record('STACK_DECISION', {'id':sid,'raw':raw,'revision':rule['revision'],**output})
            return output

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
                task = TaskStore(layers.store).read(sid)
                result = Layers.ask(request.app.state.model, row['memory'], question, updated, task)
                if task is not None:
                    result['task'] = task.view()
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
