"""HTTP-клиент с учётом каждого обращения, включая ошибки и неизвестные списания."""
from copy import deepcopy
import hashlib
import json
import time

import httpx

from .agent import AgentError
from .metrics import TokenCounter, normalize_usage, BudgetStop, number
from .tools import TOOLS
from .strategies import FACTS_RESPONSE_FORMAT

MODEL = 'qwen/qwen3-14b'
PROVIDER = 'deepinfra/fp8'
BASE = 'https://polza.ai/api/v1'
PARAMETERS = {'temperature': .2, 'max_tokens': 4096, 'reasoning': {'enabled': False}}


class Client:
    def __init__(self, key, journal, http, *, counter=None, budget=None, allow_overflow=False):
        self.key, self.journal, self.http = key, journal, http
        self.counter = counter or TokenCounter()
        self.budget = budget
        self.allow_overflow = allow_overflow
        self.context_limit = 40960
        self.rates = {'prompt': 14.302656, 'completion': 28.605312}
        self.observer = None
        self.records = []
        self.usage = []
        self.catalog_snapshot = None

    def send(self, messages, *, purpose="chat", max_tokens=None):
        output_limit = max_tokens if max_tokens is not None else PARAMETERS["max_tokens"]
        if purpose not in ("chat", "summary", "facts") or type(output_limit) is not int or not 1 <= output_limit <= PARAMETERS["max_tokens"]:
            raise AgentError("Некорректные параметры обращения к модели.")
        tools = TOOLS if purpose == "chat" else []
        payload = {'model': MODEL, 'messages': messages, 'tools': TOOLS, 'tool_choice': 'auto',
                   'stream': False, **deepcopy(PARAMETERS),
                   'provider': {'only': [PROVIDER], 'allow_fallbacks': False, 'require_parameters': True,
                                'max_price': {key: value * 1.01 for key, value in self.rates.items()}}}
        payload['max_tokens'] = output_limit
        if purpose=='facts':payload['response_format']=deepcopy(FACTS_RESPONSE_FORMAT)
        if not tools:
            payload.pop('tools'); payload.pop('tool_choice')
        serialized = json.dumps(payload, ensure_ascii=False)
        estimate = self.counter.estimate(messages, tools)
        reservation = ((len(serialized.encode()) + 2048) * self.rates['prompt'] * 1.01 +
                       output_limit * self.rates['completion'] * 1.01) / 1e6
        record = {**normalize_usage(None), 'status': 'pending', 'sent': True, 'purpose': purpose,
                  'request_tokens_estimate': estimate, 'counter_method': self.counter.method,
                  'context_limit': self.context_limit, 'http_status': None, 'response': None,
                  'request_sha256': hashlib.sha256(serialized.encode()).hexdigest(),
                  'budget_reserve_rub': reservation, 'duration_seconds': None}
        blocked = None
        if not self.allow_overflow and estimate + output_limit > self.context_limit:
            blocked = ('local_context_limit', 'Локальная оценка контекста с резервом ответа превышает окно модели. Запрос не отправлен.')
        if self.budget is not None:
            try:
                self.budget.before(reservation)
            except BudgetStop as exc:
                blocked = ('budget_limit', str(exc))
        if blocked:
            record.update(status=blocked[0], sent=False, cost_rub=0., budget_reserve_rub=0.)
        self.records.append(record)
        call_id = self.observer('start', record, None) if self.observer else None
        if blocked:
            if self.observer:
                self.observer('finish', record, call_id)
            self.journal.record('BLOCKED', record)
            raise AgentError(blocked[1])
        self.journal.record('REQUEST', payload)
        started = time.perf_counter()
        try:
            response = self.http.post(BASE + '/chat/completions', json=payload,
                headers={'Authorization': 'Bearer ' + self.key},
                timeout=httpx.Timeout(180, connect=10), follow_redirects=False)
            record['http_status'] = response.status_code
            record['response'] = self.journal.scrub(response.text)
            self.journal.record('RESPONSE', {'status': response.status_code, 'body': response.text})
            if response.status_code != 200:
                raise AgentError(f'HTTP {response.status_code}: ' + self.journal.scrub(response.text[:1500]))
            body = response.json()
            # Исключаем NaN/Infinity во всём ответе до сохранения в JSON SQLite.
            json.dumps(body, allow_nan=False)
            usage = body.get('usage') or {}
            self.usage.append(usage)
            record.update(normalize_usage(usage))
            record['response'] = self.journal.scrub(body)
            self.journal.record('USAGE', usage)
            choice = body['choices'][0]
            message = choice['message']
            if body.get('model') != MODEL:
                raise AgentError('Получен ответ другой модели.')
            if choice['finish_reason'] not in ('stop', 'tool_calls') or message.get('refusal'):
                raise AgentError('Ответ не завершён или отклонён моделью.')
            if not isinstance(message, dict) or message.get('role', 'assistant') != 'assistant':
                raise ValueError
            if choice['finish_reason'] == 'tool_calls' and not message.get('tool_calls'):
                raise ValueError
            if not message.get('tool_calls') and not isinstance(message.get('content'), str):
                raise ValueError
            if purpose != 'chat' and message.get('tool_calls'):
                raise AgentError('Инструменты при обновлении памяти недопустимы.')
            record['status'] = 'complete'
            return self.journal.scrub({'role': 'assistant', **message})
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError, AttributeError):
            record['status'] = 'api_error'
            raise AgentError('Ошибка сети или структуры ответа. Повторов нет, расход может быть неизвестен.') from None
        except AgentError:
            record['status'] = 'api_error'
            raise
        except KeyboardInterrupt:
            record['status'] = 'interrupted'
            raise
        finally:
            record['duration_seconds'] = round(time.perf_counter() - started, 6)
            if number(record.get('cost_rub')) is not None:
                record['budget_reserve_rub'] = 0.
            if self.budget is not None:
                self.budget.after(record['cost_rub'], reservation)
            if self.observer:
                self.observer('finish', self.journal.scrub(record), call_id)
            self.journal.record('CALL_METRICS', record)


def preflight(client):
    client.journal.record('CATALOG_REQUEST', BASE + '/models')
    try:
        response = client.http.get(BASE + '/models', params={'include_providers': 'true', 'include_capabilities': 'true'}, timeout=30)
        response.raise_for_status()
        catalog = next(model for model in response.json()['data'] if model['id'] == MODEL)
        provider = next(route for route in catalog.get('providers') or [] if route['name'] == PROVIDER)
        required = {'tools', 'tool_choice', 'max_tokens', 'temperature', 'reasoning','response_format','structured_outputs'}
        if not required <= set(provider.get('supported_parameters') or []):
            raise ValueError
        if provider.get('capabilities', {}).get('tools', {}).get('status') != 'supported':
            raise ValueError
        if provider.get('capabilities',{}).get('json_schema_strict',{}).get('status')!='supported':
            raise ValueError
        limit = provider.get('context_length') or catalog.get('context_length')
        if type(limit) is not int or limit <= PARAMETERS['max_tokens']:
            raise ValueError
        completion_limit = provider.get('max_completion_tokens') or catalog.get('max_completion_tokens')
        if type(completion_limit) is not int or completion_limit < PARAMETERS['max_tokens']:
            raise ValueError
        pricing = provider['pricing']
        rates = {'prompt': float(pricing['prompt_per_million']), 'completion': float(pricing['completion_per_million'])}
        if pricing.get('currency') != 'RUB' or any(number(value) is None or value <= 0 for value in rates.values()):
            raise ValueError
    except (httpx.HTTPError, StopIteration, KeyError, ValueError, TypeError):
        raise AgentError('Каталог не подтвердил маршрут, его параметры, контекст и рублёвые тарифы.') from None
    client.context_limit = limit
    client.rates = rates
    checked = {'catalog': catalog, 'provider': provider, 'rates': rates, 'context_limit': limit}
    client.catalog_snapshot = checked
    client.journal.record('CATALOG_RESPONSE', checked)
    return checked
