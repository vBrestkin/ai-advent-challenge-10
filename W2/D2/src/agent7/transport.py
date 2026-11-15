"""HTTP-клиент Polza.ai с инструментами, журналом и без повторов."""
import json,time
import httpx
from .agent import AgentError
from .tools import TOOLS

MODEL='google/gemini-2.5-flash-lite'
BASE='https://polza.ai/api/v1'

class Client:
    def __init__(self,key,journal,http):
        self.key,self.journal,self.http=key,journal,http
        self.usage=[]

    def send(self,messages):
        payload={'model':MODEL,'messages':messages,'tools':TOOLS,'tool_choice':'auto',
                 'stream':False,'temperature':.2,'max_tokens':4096,
                 'reasoning':{'enabled':True,'max_tokens':1024},
                 'provider':{'only':['google-ai-studio/flex'],'allow_fallbacks':False,'require_parameters':True}}
        self.journal.record('REQUEST',payload)
        start=time.perf_counter()
        try:
            response=self.http.post(BASE+'/chat/completions',json=payload,
                headers={'Authorization':'Bearer '+self.key},timeout=httpx.Timeout(180,connect=10),follow_redirects=False)
            self.journal.record('RESPONSE',{'status':response.status_code,'body':response.text})
            response.raise_for_status()
            body=response.json()
            usage=body.get('usage') or {}
            self.usage.append(usage)
            self.journal.record('USAGE',usage)
            choice=body['choices'][0]; message=choice['message']
            if body.get('model')!=MODEL: raise AgentError('Получен ответ другой модели.')
            if choice['finish_reason'] not in ('stop','tool_calls') or message.get('refusal'):
                raise AgentError('Ответ не завершён или отклонён моделью.')
            if not isinstance(message,dict) or message.get('role','assistant')!='assistant': raise ValueError
            if choice['finish_reason']=='tool_calls' and not message.get('tool_calls'): raise ValueError
            if not message.get('tool_calls') and not isinstance(message.get('content'),str): raise ValueError
            return self.journal.scrub({'role':'assistant', **message})
        except (httpx.HTTPError,ValueError,KeyError,IndexError,TypeError,AttributeError):
            self.journal.record('ERROR','Ошибка сети, HTTP или структуры ответа. Повтор не выполнялся.')
            raise AgentError('Ошибка провайдера или соединения. Подробности в журнале.') from None
        finally: self.journal.record('DURATION',round(time.perf_counter()-start,6))
