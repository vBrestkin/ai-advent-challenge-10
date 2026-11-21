"""Локальные оценки, фактический usage и консервативный бюджет запросов."""
import hashlib
import json
import math
from pathlib import Path

FIELDS = ('prompt_tokens', 'completion_tokens', 'total_tokens', 'reasoning_tokens', 'cost_rub')
TOKENIZER_URL = 'https://modelscope.cn/models/Qwen/Qwen3-14B/resolve/master/tokenizer.json'
TOKENIZER_SHA256 = 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4'


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


class TokenCounter:
    def __init__(self, path=None):
        self.tokenizer = None
        self.method = 'utf8-quarter-estimate'
        self.exact_text = False
        if path is not None and Path(path).is_file():
            payload = Path(path).read_bytes()
            if hashlib.sha256(payload).hexdigest() != TOKENIZER_SHA256:
                raise ValueError('Контрольная сумма токенизатора отличается.')
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(str(path))
            self.method = 'qwen3-json-estimate'
            self.exact_text = True

    def count(self, text):
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False).ids)
        return math.ceil(len(text.encode('utf-8')) / 4)

    def estimate(self, messages, tools):
        serialized = json.dumps({'messages': messages, 'tools': tools}, ensure_ascii=False, separators=(',', ':'))
        return self.count(serialized)


def normalize_usage(usage):
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get('completion_tokens_details')
    details = details if isinstance(details, dict) else {}
    result = {field: number(usage.get(field)) for field in FIELDS if field != 'reasoning_tokens'}
    result['reasoning_tokens'] = number(details.get('reasoning_tokens'))
    warnings = []
    if all(result[key] is not None for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')):
        if result['total_tokens'] != result['prompt_tokens'] + result['completion_tokens']:
            warnings.append('total отличается от prompt + completion')
    if result['reasoning_tokens'] is not None and result['completion_tokens'] is not None:
        if result['reasoning_tokens'] > result['completion_tokens']:
            warnings.append('reasoning больше completion')
    result['usage_warning'] = '; '.join(warnings) or None
    return result


def aggregate(records):
    sent = [row for row in records if row.get('sent', True)]
    result = {'calls': len(sent), 'attempts': len(records)}
    for key in FIELDS:
        values = [row[key] for row in sent if number(row.get(key)) is not None]
        result[key] = round(sum(values), 10) if values else None
        result['unknown_costs' if key == 'cost_rub' else 'unknown_' + key] = len(sent) - len(values)
    result['duration_seconds'] = round(sum(row.get('duration_seconds') or 0 for row in sent), 6)
    result['unknown_reserve_rub'] = round(sum(row.get('budget_reserve_rub') or 0 for row in sent if number(row.get('cost_rub')) is None), 10)
    return result


class BudgetStop(Exception):
    pass


class Budget:
    def __init__(self, limit):
        if number(limit) is None or limit <= 0:
            raise ValueError('Бюджет должен быть положительным конечным числом.')
        self.limit = limit
        self.spent = 0.
        self.unknown = 0.

    def before(self, reservation):
        if self.spent + self.unknown + reservation > self.limit:
            raise BudgetStop('Недостаточно остатка бюджета с учётом резервов неизвестных списаний.')

    def after(self, cost, reservation):
        if number(cost) is None:
            self.unknown += reservation
        else:
            self.spent += cost

    def accounting(self):
        return {'limit_rub': self.limit, 'known_cost_rub': round(self.spent, 10),
                'unknown_reserve_rub': round(self.unknown, 10),
                'accounted_rub': round(self.spent + self.unknown, 10)}
