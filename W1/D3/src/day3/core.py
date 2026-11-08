"""HTTP без повторов, безопасный журнал и атомарное сохранение JSON."""
from datetime import datetime
import json
import os
from pathlib import Path
import time
from uuid import uuid4

import httpx

BASE_URL = 'https://polza.ai/api/v1'


def identifier():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + uuid4().hex[:12]


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid4().hex + '.tmp')
    try:
        fd = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write('\n')
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Journal:
    def __init__(self, directory, secret=''):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.path = Path(directory) / (identifier() + '.log')
        self.secret = secret
        fd = os.open(str(self.path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.file = os.fdopen(fd, 'w', encoding='utf-8')
        self.record('START', 'Запуск day3')

    def scrub(self, value):
        if isinstance(value, str):
            return value.replace(self.secret, '[REDACTED]') if self.secret else value
        if isinstance(value, dict):
            return {self.scrub(k): self.scrub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.scrub(v) for v in value]
        return value

    def record(self, event, data):
        timestamp = datetime.now().astimezone().isoformat(timespec='seconds')
        self.file.write(f'{timestamp} {event} {json.dumps(self.scrub(data), ensure_ascii=False)}\n')
        self.file.flush()

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class ApiError(Exception):
    def __init__(self, message, result=None):
        super().__init__(message)
        self.result = result or {}


def numeric(value):
    # Не превращаем неизвестные расходы в ноль и не принимаем bool за число.
    import math
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


class ApiClient:
    def __init__(self, key, model, journal, client, *, provider='google-ai-studio/flex', connect_timeout=10, read_timeout=180):
        self.key, self.model, self.journal, self.client = key, model, journal, client
        self.provider = provider
        self.price_limits = None
        self.timeout = httpx.Timeout(read_timeout, connect=connect_timeout)

    def catalog(self):
        self.journal.record('CATALOG_REQUEST', BASE_URL + '/models')
        try:
            response = self.client.get(BASE_URL + '/models', params={'include_providers':'true', 'include_capabilities':'true'}, timeout=self.timeout)
            response.raise_for_status()
            model = next(m for m in response.json()['data'] if m['id'] == self.model)
        except (httpx.HTTPError, ValueError, KeyError, TypeError, StopIteration):
            raise ApiError('Не удалось получить возможности модели из каталога.') from None
        self.journal.record('CATALOG_RESPONSE', model)
        return self.journal.scrub(model)

    def send(self, messages, parameters=None):
        params = {'max_tokens': 32768, 'temperature': .7, 'reasoning': {'enabled': True, 'max_tokens': 512}, **(parameters or {})}
        if set(params) - {'max_tokens', 'temperature', 'reasoning', 'response_format'}:
            raise ApiError('Неподдерживаемые параметры: нельзя переопределять модель или маршрут в теле шага.')
        provider = {'only': [self.provider], 'allow_fallbacks': False, 'require_parameters': True}
        if self.price_limits is not None:
            provider['max_price'] = self.price_limits
        payload = {'model': self.model, 'messages': messages, 'stream': False,
                   'provider': provider, **params}
        self.journal.record('REQUEST', payload)
        started = time.perf_counter()
        result = dict(request=self.journal.scrub(payload), response=None, content='',
                      finish_reason=None, complete=False, actual_model=None, provider=None,
                      total_tokens=None, prompt_tokens=None, completion_tokens=None,
                      reasoning_tokens=None, cost_rub=None, http_status=None, api_error=None)
        try:
            response = self.client.post(BASE_URL + '/chat/completions',
                headers={'Authorization': 'Bearer ' + self.key}, json=payload,
                timeout=self.timeout, follow_redirects=False)
            result['http_status'] = response.status_code
            result['response'] = self.journal.scrub(response.text)
            self.journal.record('RESPONSE', dict(status=response.status_code, body=response.text))
            if response.status_code != 200:
                hints = {401: 'Неверный ключ.', 402: 'Недостаточно средств или исчерпан бюджет.',
                         429: 'Лимит запросов.', 404: 'Модель не найдена.'}
                raise ApiError(f'HTTP {response.status_code}. ' + hints.get(response.status_code, 'Ошибка провайдера.'))
            try:
                body = response.json()
                result['response'] = self.journal.scrub(body)
                choice = body['choices'][0]
                message, reason = choice['message'], choice['finish_reason']
                content = message.get('content')
                refusal = bool(message.get('refusal')) or reason == 'content_filter'
                if content is None and (refusal or reason == 'length'):
                    content = ''
                if not isinstance(content, str) or not isinstance(reason, str):
                    raise ValueError
                usage = body.get('usage') or {}
                if not isinstance(usage, dict):
                    usage = {}
                details = usage.get('completion_tokens_details') or {}
                reasoning_tokens = numeric(details.get('reasoning_tokens')) if isinstance(details, dict) else None
                result.update(content=self.journal.scrub(content), finish_reason=reason,
                    complete=reason == 'stop' and bool(content.strip()) and not refusal,
                    actual_model=body.get('model'), provider=body.get('provider'), reasoning_tokens=reasoning_tokens,
                    **{key: numeric(usage.get(key)) for key in ('total_tokens', 'prompt_tokens', 'completion_tokens', 'cost_rub')})
            except (ValueError, KeyError, TypeError, IndexError, AttributeError):
                raise ApiError('Некорректный JSON или структура ответа API.') from None
            if reason == 'error':
                raise ApiError('Ошибка генерации на стороне провайдера.')
        except httpx.TimeoutException:
            result['api_error'] = 'Таймаут подключения или обмена данными; повторов нет.'
        except httpx.RequestError:
            result['api_error'] = 'Ошибка сети (DNS, TLS, соединение или разрыв связи); повторов нет.'
        except ApiError as exc:
            result['api_error'] = str(exc)
        result['duration_seconds'] = round(time.perf_counter() - started, 6)
        self.journal.record('METRICS', {k: v for k, v in result.items() if k not in ('request', 'response', 'content')})
        if result['api_error']:
            self.journal.record('ERROR', result['api_error'])
            raise ApiError(result['api_error'], result)
        return self.journal.scrub(result)
