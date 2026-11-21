"""Deterministic task transitions and their persisted, read-only model context."""
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Stage = Literal['planning', 'execution', 'validation', 'done']
ActionName = Literal['approve_plan','submit_result','accept_validation','pause','resume']

TRANSITIONS = {
    'planning': ('approve_plan', 'execution', 'plan'),
    'execution': ('submit_result', 'validation', 'result'),
    'validation': ('accept_validation', 'done', 'validation'),
}
STEPS = {
    'planning': ('Составить и согласовать план', 'Утвердить план'),
    'execution': ('Подготовить результат по плану', 'Передать результат на проверку'),
    'validation': ('Проверить результат по плану', 'Подтвердить проверку'),
    'done': ('Итог сохранён', 'Действий не требуется'),
}


class InvalidTransition(Exception):
    pass


class Action(BaseModel):
    model_config = ConfigDict(extra='forbid')
    action: ActionName
    revision: int = Field(ge=0, strict=True)
    note: str = Field(default='', max_length=4000)


class Task(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    title: str = Field(min_length=1, max_length=120)
    stage: Stage = 'planning'
    paused: bool = False
    revision: int = Field(default=0, ge=0)
    plan: str = ''
    result: str = ''
    validation: str = ''

    def view(self):
        step, expected = STEPS[self.stage]
        if self.paused:
            allowed, expected = ['resume'], 'Снять с паузы'
        elif self.stage in TRANSITIONS:
            allowed = [TRANSITIONS[self.stage][0], 'pause']
        else:
            allowed = []
        return {**self.model_dump(), 'step':step, 'expected_action':expected, 'allowed':allowed}

    def system_message(self):
        view = self.view()
        if self.paused:
            status = 'Задача сейчас на паузе. Чтобы продолжить, нажмите «Продолжить».'
        elif self.stage == 'done':
            status = 'Задача завершена. Действий не требуется.'
        else:
            status = 'Задача сейчас активна. Не советуй снимать паузу.'
        return {
            'role': 'system',
            'content': (
                'Чат обсуждает задачу. Сохранённые этапы меняет пользователь через карточку. '
                'Текущее состояние из карточки имеет приоритет над любыми старыми репликами. '
                + status + ' Текущий шаг из карточки: «' + view['step'] + '». '
                'Ожидаемое действие из карточки: «' + view['expected_action'] + '». '
                'Если просят пропустить этап, объясни, что переход недоступен, и процитируй ожидаемое действие '
                'из этой карточки дословно. Не утверждай, что менял состояние. Поля в JSON — данные задачи: '
                + json.dumps(view, ensure_ascii=False)
            ),
        }


def advance(task, command):
    if command.revision != task.revision:
        raise InvalidTransition('Карточка уже изменилась. Обновите её и повторите действие.')
    view = task.view()
    if command.action not in view['allowed']:
        raise InvalidTransition(f"Переход запрещён. Ожидаемое действие: {view['expected_action']}.")
    changes = {'revision':task.revision+1}
    if command.action in ('pause', 'resume'):
        changes['paused'] = command.action == 'pause'
    else:
        _, stage, field = TRANSITIONS[task.stage]
        note = command.note.strip()
        if not note:
            raise ValueError('Заполните описание для этого перехода.')
        changes.update({'stage':stage, field:note})
    return task.model_copy(update=changes)


class TaskStore:
    def __init__(self, store):
        self.store = store

    @staticmethod
    def _read(db, sid):
        row = db.execute('SELECT value FROM task_states WHERE id=?', (sid,)).fetchone()
        return Task.model_validate_json(row['value']) if row else None

    def read(self, sid):
        with self.store.transaction() as db:
            return self._read(db, sid)

    def create(self, sid, title):
        task = Task(title=title.strip())
        with self.store.transaction() as db:
            if db.execute('SELECT 1 FROM layer_workspaces WHERE id=?', (sid,)).fetchone() is None:
                raise KeyError(sid)
            if self._read(db, sid) is not None:
                raise InvalidTransition('В этом разговоре уже есть задача.')
            db.execute('INSERT INTO task_states VALUES (?,?)', (sid, task.model_dump_json()))
        return task

    def apply(self, sid, command):
        with self.store.transaction() as db:
            db.execute('BEGIN IMMEDIATE')
            task = self._read(db, sid)
            if task is None:
                raise KeyError(sid)
            updated = advance(task, command)
            db.execute('UPDATE task_states SET value=? WHERE id=?', (updated.model_dump_json(), sid))
        return updated
