"""Агент владеет циклом LLM → инструменты → LLM и историей текущего процесса."""
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

    def run(self,prompt):
        if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>20000:
            raise AgentError('Введите сообщение от 1 до 20000 символов.')
        messages=deepcopy(self.messages)+[{'role':'user','content':prompt}]
        steps=[]
        seen=set()
        for _ in range(self.max_rounds):
            reply=self.model.send([{'role':'system','content':SYSTEM}]+messages)
            calls=reply.get('tool_calls') or []
            messages.append(reply)
            if not calls:
                answer=reply.get('content')
                if not isinstance(answer,str) or not answer.strip(): raise AgentError('Модель вернула пустой ответ.')
                self.messages=messages
                return {'answer':answer,'steps':steps,'messages':deepcopy(messages)}
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
                if hasattr(self.model,'journal'): self.model.journal.record('TOOL',step)
        raise AgentError('Достигнут лимит шагов агента. Уточните запрос.')
