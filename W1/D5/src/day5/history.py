"""Атомарный стек успешных ходов с фиксированным профилем D5."""
from copy import deepcopy
import json
from pathlib import Path

from .core import atomic_json
from .models import MODELS, PROVIDER, PARAMETERS


class HistoryError(Exception):pass


class Conversation:
    def __init__(self,path,data):
        self.path,self.data=Path(path),data
        self.snapshot=self.path.read_bytes()

    @property
    def model(self):return self.data['model']

    @property
    def messages(self):return deepcopy(self.data['messages'])

    @classmethod
    def create(cls,path,model,*,provider=PROVIDER):
        path=Path(path)
        if path.exists():raise HistoryError('История уже существует; используйте продолжение.')
        if model not in MODELS.values() or provider!=PROVIDER:raise HistoryError('Неизвестный профиль модели.')
        data={'version':1,'model':model,'provider':provider,'parameters':deepcopy(PARAMETERS),'messages':[]}
        try:
            atomic_json(path,data)
            return cls(path,data)
        except OSError:raise HistoryError('Не удалось создать историю.') from None

    @classmethod
    def load(cls,path):
        try:
            data=json.loads(Path(path).read_text(encoding='utf-8'))
            if not (type(data['version']) is int and data['version']==1 and data['model'] in MODELS.values()
                    and data['provider']==PROVIDER and data['parameters']==PARAMETERS):raise ValueError
            messages=data['messages']
            if not isinstance(messages,list) or len(messages)%2:raise ValueError
            for i,message in enumerate(messages):
                if not (message['role']==('user' if i%2==0 else 'assistant')
                        and isinstance(message['content'],str) and message['content'].strip()):raise ValueError
            return cls(path,data)
        except (OSError,ValueError,TypeError,KeyError):
            raise HistoryError('История повреждена или недоступна; исходный файл сохранён.') from None

    def ask(self,prompt,api):
        if not prompt.strip():raise HistoryError('Пустое сообщение не отправлено.')
        if self.model!=api.model or self.data['provider']!=api.provider:
            raise HistoryError('Модель или маршрут отличаются от сохранённых; начните новый разговор.')
        if self.path.read_bytes()!=self.snapshot:raise HistoryError('История изменена другим процессом.')
        messages=self.messages+[{'role':'user','content':api.journal.scrub(prompt)}]
        result=api.send(messages,self.data['parameters'])
        if not result['complete'] or result['actual_model']!=self.model:
            raise HistoryError('Ответ не завершён или получен от другой модели; ход остался в журнале.')
        data={**self.data,'messages':messages+[{'role':'assistant','content':result['content']}]}
        try:
            if self.path.read_bytes()!=self.snapshot:raise HistoryError('История изменилась во время запроса.')
            atomic_json(self.path,data)
            snapshot=self.path.read_bytes()
        except OSError:raise HistoryError('Ответ получен, но историю не удалось сохранить; посмотрите журнал.') from None
        self.data,self.snapshot=data,snapshot
        api.journal.record('HISTORY_SAVED',str(self.path))
        return result['content']
