"""Полные снимки диалогов и незавершённых попыток в транзакциях SQLite."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import threading
from uuid import uuid4
from .memory import Memory, validate_memory
from .strategies import Facts,validate_facts


class StoreError(Exception):
    pass


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def validate_messages(messages):
    if not isinstance(messages, list):
        raise ValueError('messages должен быть массивом')
    state = 'user'
    pending = set()
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError('Некорректное сообщение')
        role = message.get('role')
        if role == 'user':
            if state != 'user' or not isinstance(message.get('content'), str) or not message['content'].strip():
                raise ValueError('Нарушена последовательность ходов')
            state = 'assistant'
        elif role == 'assistant':
            if state != 'assistant' or pending:
                raise ValueError('Отсутствуют результаты инструментов')
            calls = message.get('tool_calls')
            if calls is None:
                calls = []
            if not isinstance(calls, list):
                raise ValueError('Некорректные tool_calls')
            if calls:
                for call in calls:
                    call_id = call['id']
                    if not isinstance(call_id, str) or not call_id or call_id in pending:
                        raise ValueError('Некорректный идентификатор инструмента')
                    if call.get('type') != 'function' or not isinstance(call['function']['name'], str) or not isinstance(call['function']['arguments'], str):
                        raise ValueError('Некорректный вызов инструмента')
                    pending.add(call_id)
                state = 'tool'
            else:
                if not isinstance(message.get('content'), str) or not message['content'].strip():
                    raise ValueError('Пустой ответ')
                state = 'user'
        elif role == 'tool':
            if state != 'tool' or message.get('tool_call_id') not in pending or not isinstance(message.get('content'), str):
                raise ValueError('Результат без соответствующего вызова')
            pending.remove(message['tool_call_id'])
            if not pending:
                state = 'assistant'
        else:
            raise ValueError('Неизвестная роль')
    if state != 'user' or pending:
        raise ValueError('Незавершённый протокол')


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.mutex = threading.RLock()
        self.connection = None
        self.lock_file = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor = os.open(str(self.path) + '.lock', os.O_CREAT | os.O_RDWR, 0o600)
            self.lock_file = os.fdopen(descriptor, 'a')
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            existed = self.path.exists()
            if not existed:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                os.close(descriptor)
            self.connection = sqlite3.connect(self.path, timeout=1, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            version = self.connection.execute('PRAGMA user_version').fetchone()[0]
            if existed and version not in (1, 2, 3, 4):
                raise StoreError('Неизвестная версия базы. Исходный файл сохранён.')
            if not existed:
                self.connection.executescript('''
                    BEGIN;
                    CREATE TABLE sessions (
                        id TEXT PRIMARY KEY, title TEXT NOT NULL, profile TEXT NOT NULL,
                        revision INTEGER NOT NULL DEFAULT 0,
                        messages TEXT NOT NULL DEFAULT '[]', turns TEXT NOT NULL DEFAULT '[]');
                    CREATE TABLE attempts (
                        id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
                        revision INTEGER NOT NULL, prompt TEXT NOT NULL,
                        status TEXT NOT NULL, messages TEXT NOT NULL DEFAULT '[]',
                        error TEXT, created_at TEXT NOT NULL);
                    PRAGMA user_version=1;
                    COMMIT;
                ''')
            if not existed or version == 1:
                with self.connection:
                    self.connection.execute('BEGIN IMMEDIATE')
                    self.connection.execute("""CREATE TABLE calls (
                        id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
                        attempt_id TEXT NOT NULL REFERENCES attempts(id), record TEXT NOT NULL)""")
                    self.connection.execute('PRAGMA user_version=2')
            if not existed or version in (1, 2):
                with self.connection:
                    self.connection.execute("CREATE TABLE memories (session_id TEXT PRIMARY KEY REFERENCES sessions(id), value TEXT NOT NULL)")
                    self.connection.execute('PRAGMA user_version=3')
            if not existed or version in (1,2,3):
                with self.connection:
                    self.connection.execute('BEGIN IMMEDIATE')
                    self.connection.execute('CREATE TABLE facts_states (session_id TEXT PRIMARY KEY REFERENCES sessions(id), value TEXT NOT NULL)')
                    self.connection.execute('CREATE TABLE checkpoints (id TEXT PRIMARY KEY,session_id TEXT NOT NULL REFERENCES sessions(id),title TEXT NOT NULL,snapshot TEXT NOT NULL,created_at TEXT NOT NULL)')
                    self.connection.execute('CREATE TABLE branches (session_id TEXT PRIMARY KEY REFERENCES sessions(id),checkpoint_id TEXT NOT NULL REFERENCES checkpoints(id))')
                    self.connection.execute('PRAGMA user_version=4')
            if self.connection.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise StoreError('Нарушена целостность базы. Работа остановлена.')
            self.connection.execute('PRAGMA foreign_keys=ON')
            self.connection.execute('PRAGMA journal_mode=WAL')
            self.connection.execute('PRAGMA synchronous=FULL')
            with self.transaction() as connection:
                connection.execute("UPDATE attempts SET status='interrupted', error='Приложение остановлено до завершения хода' WHERE status='running'")
                for call in connection.execute('SELECT id,record FROM calls').fetchall():
                    data = json.loads(call['record'])
                    if data.get('status') == 'pending':
                        data['status'] = 'uncertain'
                        connection.execute('UPDATE calls SET record=? WHERE id=?', (encode(data), call['id']))
        except (OSError, sqlite3.Error, StoreError, ValueError, TypeError) as exc:
            self.close()
            raise StoreError('База недоступна, повреждена, имеет другую версию или уже открыта другим сервером.') from None

    @contextmanager
    def transaction(self):
        with self.mutex:
            try:
                with self.connection:
                    yield self.connection
            except (sqlite3.Error, OSError):
                raise StoreError('Не удалось прочитать или сохранить историю SQLite.') from None

    def create(self, title, profile):
        session_id = uuid4().hex
        with self.transaction() as connection:
            connection.execute('INSERT INTO sessions(id,title,profile) VALUES(?,?,?)',
                               (session_id, title, encode(profile)))
        return session_id

    def list(self):
        with self.transaction() as connection:
            return [dict(row) for row in connection.execute('SELECT s.id,s.title,s.revision,b.checkpoint_id FROM sessions s LEFT JOIN branches b ON b.session_id=s.id ORDER BY s.rowid')]

    def load(self, session_id):
        with self.transaction() as connection:
            row = connection.execute('SELECT * FROM sessions WHERE id=?', (session_id,)).fetchone()
            if row is None:
                return None
            try:
                data = dict(row)
                for field in ('profile', 'messages', 'turns'):
                    data[field] = json.loads(data[field])
                validate_messages(data['messages'])
                if not isinstance(data['turns'], list) or not isinstance(data['profile'], dict):
                    raise ValueError
                if len(data['turns']) != sum(message['role'] == 'user' for message in data['messages']):
                    raise ValueError
                memory_row = connection.execute('SELECT value FROM memories WHERE session_id=?', (session_id,)).fetchone()
                data['memory'] = json.loads(memory_row['value']) if memory_row else Memory().to_dict()
                validate_memory(data['messages'], Memory(**data['memory']))
                facts_row = connection.execute('SELECT value FROM facts_states WHERE session_id=?',(session_id,)).fetchone()
                data['facts_state'] = json.loads(facts_row['value']) if facts_row else Facts().to_dict()
                validate_facts(data['messages'],Facts(**data['facts_state']))
                data['attempts'] = []
                for attempt in connection.execute('SELECT * FROM attempts WHERE session_id=? ORDER BY rowid', (session_id,)):
                    item = dict(attempt)
                    item['messages'] = json.loads(item['messages'])
                    data['attempts'].append(item)
                data['calls'] = []
                for call in connection.execute('SELECT * FROM calls WHERE session_id=? ORDER BY rowid', (session_id,)):
                    record = json.loads(call['record'])
                    if not isinstance(record, dict):
                        raise ValueError
                    data['calls'].append({**record, 'call_id': call['id'], 'attempt_id': call['attempt_id']})
                return data
            except (ValueError, TypeError, KeyError):
                raise StoreError('Сохранённый протокол повреждён. Исходные данные сохранены.') from None

    def begin(self, session_id, prompt):
        attempt_id = uuid4().hex
        with self.transaction() as connection:
            row = connection.execute('SELECT revision FROM sessions WHERE id=?', (session_id,)).fetchone()
            if row is None:
                raise StoreError('Разговор не найден.')
            active = connection.execute("SELECT 1 FROM attempts WHERE session_id=? AND status='running'", (session_id,)).fetchone()
            if active:
                raise StoreError('В разговоре уже есть незавершённый запрос.')
            connection.execute('INSERT INTO attempts(id,session_id,revision,prompt,status,created_at) VALUES(?,?,?,?,?,?)',
                (attempt_id, session_id, row['revision'], prompt, 'running', datetime.now(timezone.utc).isoformat()))
        return attempt_id

    def checkpoint(self, attempt_id, messages):
        with self.transaction() as connection:
            result = connection.execute("UPDATE attempts SET messages=? WHERE id=? AND status='running'",
                                        (encode(messages), attempt_id))
            if result.rowcount != 1:
                raise StoreError('Незавершённая попытка не найдена.')

    def complete(self, attempt_id, messages, turn, *, memory=None, facts_state=None):
        try:
            validate_messages(messages)
            if memory is not None:
                validate_memory(messages, Memory(**memory))
            if facts_state is not None:
                validate_facts(messages,Facts(**facts_state))
        except (ValueError, TypeError, KeyError):
            raise StoreError('Не удалось сохранить некорректный протокол. Прежняя история сохранена.') from None
        with self.transaction() as connection:
            attempt = connection.execute("SELECT * FROM attempts WHERE id=? AND status='running'", (attempt_id,)).fetchone()
            if attempt is None:
                raise StoreError('Попытка уже завершена или не найдена.')
            row = connection.execute('SELECT * FROM sessions WHERE id=?', (attempt['session_id'],)).fetchone()
            if row['revision'] != attempt['revision']:
                raise StoreError('История изменилась во время запроса.')
            if memory is not None:
                connection.execute('INSERT OR REPLACE INTO memories(session_id,value) VALUES(?,?)', (attempt['session_id'],encode(memory)))
            if facts_state is not None:
                connection.execute('INSERT OR REPLACE INTO facts_states(session_id,value) VALUES(?,?)',(attempt['session_id'],encode(facts_state)))
            turns = json.loads(row['turns']) + [turn]
            connection.execute('UPDATE sessions SET messages=?,turns=?,revision=revision+1 WHERE id=?',
                               (encode(messages), encode(turns), attempt['session_id']))
            connection.execute("UPDATE attempts SET messages=?,status='complete' WHERE id=?", (encode(messages), attempt_id))

    def fail(self, attempt_id, error):
        with self.transaction() as connection:
            connection.execute("UPDATE attempts SET status='failed',error=? WHERE id=? AND status='running'", (error, attempt_id))

    def begin_call(self, attempt_id, record):
        call_id = uuid4().hex
        with self.transaction() as connection:
            attempt = connection.execute("SELECT session_id FROM attempts WHERE id=? AND status='running'", (attempt_id,)).fetchone()
            if attempt is None:
                raise StoreError('Активная попытка для учёта метрик не найдена.')
            connection.execute('INSERT INTO calls(id,session_id,attempt_id,record) VALUES(?,?,?,?)',
                               (call_id, attempt['session_id'], attempt_id, encode(record)))
        return call_id

    def finish_call(self, call_id, record):
        with self.transaction() as connection:
            changed = connection.execute('UPDATE calls SET record=? WHERE id=?', (encode(record), call_id))
            if changed.rowcount != 1:
                raise StoreError('Запись обращения к модели не найдена.')

    def save_checkpoint(self,session_id,title):
        if not isinstance(title,str) or not title.strip() or len(title)>120:raise StoreError('Введите имя checkpoint до 120 символов.')
        with self.transaction() as connection:
            row=self.load(session_id)
            if row is None:raise StoreError('Исходный разговор не найден.')
            if any(attempt['status']=='running' for attempt in row['attempts']):raise StoreError('Дождитесь завершения текущего хода.')
            snapshot={key:row[key] for key in ('profile','messages','turns','revision','memory','facts_state')}
            checkpoint_id=uuid4().hex
            connection.execute('INSERT INTO checkpoints VALUES(?,?,?,?,?)',(checkpoint_id,session_id,title.strip(),encode(snapshot),datetime.now(timezone.utc).isoformat()))
            return checkpoint_id

    def list_checkpoints(self):
        with self.transaction() as connection:
            return [dict(row) for row in connection.execute('SELECT id,session_id,title,created_at FROM checkpoints ORDER BY rowid')]

    def fork(self,checkpoint_id,title):
        if not isinstance(title,str) or not title.strip() or len(title)>120:raise StoreError('Введите имя ветки до 120 символов.')
        with self.transaction() as connection:
            row=connection.execute('SELECT snapshot FROM checkpoints WHERE id=?',(checkpoint_id,)).fetchone()
            if row is None:raise StoreError('Checkpoint не найден.')
            try:
                snapshot=json.loads(row['snapshot'])
                validate_messages(snapshot['messages'])
                validate_memory(snapshot['messages'],Memory(**snapshot['memory']))
                validate_facts(snapshot['messages'],Facts(**snapshot['facts_state']))
                if (type(snapshot['revision']) is not int or snapshot['revision']<0 or not isinstance(snapshot['profile'],dict)
                    or not isinstance(snapshot['turns'],list) or len(snapshot['turns'])!=sum(m['role']=='user' for m in snapshot['messages'])):raise ValueError
            except (ValueError,TypeError,KeyError):raise StoreError('Checkpoint повреждён. Исходные данные сохранены.') from None
            session_id=uuid4().hex
            connection.execute('INSERT INTO sessions(id,title,profile,revision,messages,turns) VALUES(?,?,?,?,?,?)',
                (session_id,title.strip(),encode(snapshot['profile']),snapshot['revision'],encode(snapshot['messages']),encode(snapshot['turns'])))
            connection.execute('INSERT INTO memories VALUES(?,?)',(session_id,encode(snapshot['memory'])))
            connection.execute('INSERT INTO facts_states VALUES(?,?)',(session_id,encode(snapshot['facts_state'])))
            connection.execute('INSERT INTO branches VALUES(?,?)',(session_id,checkpoint_id))
            return session_id

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.lock_file is not None:
            self.lock_file.close()
            self.lock_file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
