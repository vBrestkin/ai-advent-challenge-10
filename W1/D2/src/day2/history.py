"""В истории остаются только успешно полученные и сохранённые пары ходов."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path

from .core import atomic_json


class HistoryError(Exception):
    pass


class Conversation:
    def __init__(self, path, data):
        self.path, self.data = Path(path), data
        self.snapshot = self.path.read_bytes()

    @property
    def messages(self):
        return deepcopy(self.data['messages'])

    @property
    def model(self):
        return self.data['model']

    @classmethod
    def create(cls, path, model, parameters=None):
        path = Path(path)
        if path.exists():
            raise HistoryError('История уже существует; используйте продолжение разговора.')
        data = dict(version=1, model=model, parameters={'max_tokens': 4096, 'temperature': .7, **(parameters or {})}, messages=[])
        try:
            atomic_json(path, data)
            return cls(path, data)
        except OSError:
            raise HistoryError('Не удалось создать файл истории.') from None

    @classmethod
    def load(cls, path):
        try:
            data = json.loads(Path(path).read_text(encoding='utf-8'))
            if not (type(data['version']) is int and data['version'] == 1
                    and isinstance(data['model'], str) and data['model'].strip()):
                raise ValueError
            params = data['parameters']
            if not (isinstance(params, dict) and type(params['max_tokens']) is int and params['max_tokens'] > 0
                    and type(params['temperature']) in (int, float) and math.isfinite(params['temperature'])
                    and 0 <= params['temperature'] <= 2 and set(params) == {'max_tokens', 'temperature'}):
                raise ValueError
            messages = data['messages']
            if not isinstance(messages, list) or len(messages) % 2:
                raise ValueError
            for i, message in enumerate(messages):
                if not (message['role'] == ('user' if i % 2 == 0 else 'assistant')
                        and isinstance(message['content'], str) and message['content'].strip()):
                    raise ValueError
            return cls(path, data)
        except (OSError, ValueError, KeyError, TypeError):
            raise HistoryError('Файл истории повреждён или недоступен; исходный файл сохранён.') from None

    def ask(self, prompt, api):
        if not prompt.strip():
            raise HistoryError('Сообщение не должно быть пустым.')
        if api.model != self.model:
            raise HistoryError('Модель не совпадает с сохранёнными настройками разговора.')
        if self.path.read_bytes() != self.snapshot:
            raise HistoryError('История изменена другим процессом; откройте её заново.')
        messages = self.messages + [{'role': 'user', 'content': api.journal.scrub(prompt)}]
        response = api.send(messages, self.data['parameters'])
        if not response['complete']:
            raise HistoryError(f"Ответ не завершён (finish_reason={response['finish_reason']}); ход остался только в журнале.")
        data = {**self.data, 'messages': messages + [{'role': 'assistant', 'content': response['content']}]}
        try:
            atomic_json(self.path, data)
            snapshot = self.path.read_bytes()
        except OSError:
            raise HistoryError('Ответ получен, но история не сохранена. Посмотрите журнал; автоматического повтора нет.') from None
        self.data, self.snapshot = data, snapshot
        api.journal.record('HISTORY_SAVED', str(self.path))
        return response['content']
