"""Агент управляет циклом инструментов и передаёт полные снимки хранилищу."""
from copy import deepcopy
import json
from .tools import execute

SYSTEM = ('Ты помощник с учебными заказами. Для сумм заказов обращайся к find_order, '
          'для вычислений к calculate. Используй реальные результаты инструментов. '
          'Если данных недостаточно, уточни вопрос. Отвечай по-русски.')

class AgentError(Exception): pass


class Agent:
    def __init__(self,model,*,max_rounds=6):
        self.model=model
        self.max_rounds=max_rounds
        self.messages=[]

    def run(self,prompt,*,checkpoint=None):
        if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>500000:
            raise AgentError('Введите сообщение от 1 до 500000 символов.')
        messages=deepcopy(self.messages)+[{'role':'user','content':prompt}]
        steps=[]
        llm_calls=0
        seen=set()
        for _ in range(self.max_rounds):
            llm_calls+=1
            reply=self.model.send([{'role':'system','content':SYSTEM}]+messages)
            calls=reply.get('tool_calls')
            if calls is None:
                calls=[]
            if not isinstance(calls,list):
                raise AgentError('Некорректный список вызовов инструментов.')
            messages.append(reply)
            if checkpoint is not None:
                checkpoint(deepcopy(messages))
            if not calls:
                answer=reply.get('content')
                if not isinstance(answer,str) or not answer.strip(): raise AgentError('Модель вернула пустой ответ.')
                self.messages=messages
                return {'answer':answer,'steps':steps,'messages':deepcopy(messages),'llm_calls':llm_calls}
            if not isinstance(calls,list) or len(calls)>4: raise AgentError('Слишком много вызовов инструментов.')
            for call in calls:
                try:
                    call_id=call['id']; name=call['function']['name']; raw=call['function']['arguments']
                    if not isinstance(call_id,str) or not call_id or call_id in seen: raise ValueError
                    if call.get('type')!='function' or not isinstance(name,str) or not isinstance(raw,str): raise ValueError
                except (KeyError,TypeError,ValueError): raise AgentError('Некорректный протокол вызова инструмента.') from None
                seen.add(call_id)
                try: args=json.loads(raw)
                except (ValueError,TypeError): args=None
                result=execute(name,args)
                step={'id':call_id,'tool':name,'arguments':args,'result':result}
                steps.append(step)
                messages.append({'role':'tool','tool_call_id':call_id,'content':json.dumps(result,ensure_ascii=False)})
                if checkpoint is not None:
                    checkpoint(deepcopy(messages))
                if hasattr(self.model,'journal'): self.model.journal.record('TOOL',step)
        raise AgentError('Достигнут лимит шагов агента. Уточните запрос.')
