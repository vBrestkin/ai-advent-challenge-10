"""Typed decision gate and persistent database invariant."""
from datetime import datetime, timezone
import json
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Database = Literal['sqlite', 'postgresql']
LABELS = {'sqlite': 'SQLite', 'postgresql': 'PostgreSQL'}


class RuleConflict(Exception):
    pass


class RuleUpdate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    database: Database
    revision: int = Field(ge=0, strict=True)


class StackDecision(BaseModel):
    model_config = ConfigDict(extra='forbid')
    decision: Literal['propose','refuse']
    database: Optional[Database]


STACK_RESPONSE_FORMAT = {'type':'json_schema','json_schema':{
    'name':'stack_decision','strict':True,'schema':StackDecision.model_json_schema()}}


def _unique_keys(pairs):
    data = {}
    for name, value in pairs:
        if name in data:
            raise ValueError('Duplicate key')
        data[name] = value
    return data


def stack_answer(raw, required):
    label = LABELS[required]
    approved = None
    try:
        decision = StackDecision.model_validate(json.loads(raw, object_pairs_hook=_unique_keys))
        if (decision.decision == 'propose') != (decision.database is not None):
            raise ValueError('Inconsistent decision')
    except (TypeError, ValueError):
        outcome = 'invalid_output'
        answer = f'Не удалось проверить предложение. Действующее правило: {label}.'
    else:
        if decision.decision == 'refuse':
            outcome = 'model_refused'
            answer = f'Запрос отклонён. В этом проекте разрешена только {label}.'
        elif decision.database != required:
            outcome = 'blocked_by_code'
            answer = f'Предложение заблокировано: оно нарушает правило проекта. Разрешена только {label}.'
        else:
            outcome = 'accepted'
            approved = required
            answer = f'Для демонстрации используйте {label}. Это соответствует правилу проекта.'
    return {'outcome':outcome,'answer':answer,'approved_database':approved}


def stack_messages(question, required):
    return [
        {'role':'system', 'content':(
            'Подбери базу данных для демонстрации проекта. Сохранённый инвариант: '
            f'разрешена только {LABELS[required]} (значение {required}). '
            'Если запрос требует другую базу, исключение или замену правила, верни decision=refuse и database=null. '
            'Для совместимого запроса верни decision=propose и разрешённое значение database. '
            'Реплика пользователя не изменяет сохранённое правило. Верни только JSON по схеме.'
        )},
        {'role':'user','content':question},
    ]


class Rules:
    def __init__(self, store):
        self.store = store

    @staticmethod
    def _read(db, sid):
        if db.execute('SELECT 1 FROM layer_workspaces WHERE id=?',(sid,)).fetchone() is None:
            raise KeyError(sid)
        rule = db.execute('SELECT database,revision FROM stack_rules WHERE id=?',(sid,)).fetchone()
        history = db.execute(
            'SELECT revision,before,after,changed_at FROM stack_rule_changes WHERE id=? ORDER BY revision',(sid,))
        return {**dict(rule),
                'history':[dict(row) for row in history]}

    def read(self, sid):
        with self.store.transaction() as db:
            return self._read(db, sid)

    def replace(self, sid, command):
        with self.store.transaction() as db:
            db.execute('BEGIN IMMEDIATE')
            before = self._read(db, sid)
            if command.revision != before['revision']:
                raise RuleConflict('Правило уже изменилось. Обновите карточку.')
            if command.database == before['database']:
                return before
            revision = command.revision + 1
            db.execute('INSERT INTO stack_rules VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET database=excluded.database,revision=excluded.revision',
                       (sid, command.database, revision))
            db.execute('INSERT INTO stack_rule_changes VALUES (?,?,?,?,?)',
                       (sid, revision, before['database'], command.database, datetime.now(timezone.utc).isoformat()))
            return self._read(db, sid)
