"""Воспроизводимый эксперимент; отчёты строятся только из сохранённых ответов."""
from copy import deepcopy
import csv
from datetime import datetime
import io
import json
from pathlib import Path
import random

from jsonschema import Draft202012Validator

from .core import ApiError, atomic_json, identifier

PROMPT = 'Составь три идеи заданий для изучения Python: название, уровень сложности и длительность в минутах.'
FORMAT = ('Ответь только JSON без Markdown и пояснений. Объект с единственным ключом ideas: массив объектов '
          'с полями title (непустая строка), level (beginner, intermediate или advanced), '
          'duration_minutes (положительное целое число). Идей должно быть ровно три, названия разные.')
SCHEMA = {'type': 'object', 'properties': {'ideas': {'type': 'array', 'items': {
    'type': 'object', 'properties': {'title': {'type': 'string'},
    'level': {'type': 'string', 'enum': ['beginner', 'intermediate', 'advanced']},
    'duration_minutes': {'type': 'integer'}},
    'required': ['title', 'level', 'duration_minutes'], 'additionalProperties': False}}},
    'required': ['ideas'], 'additionalProperties': False}
VARIANTS = {
    'baseline': ('Без требования формата', {}),
    'prompt_json': ('JSON только в промпте', {}),
    'token_limit': ('JSON + лимит 32 токена', {'max_tokens': 32}),
    'stop_sequence': ('JSON + stop на ключе level', {'stop': ['"level"']}),
    'json_mode': ('JSON mode', {'response_format': {'type': 'json_object'}}),
    'strict_schema': ('Строгая JSON Schema', {'response_format': {'type': 'json_schema',
        'json_schema': {'name': 'python_ideas', 'strict': True, 'schema': SCHEMA}}}),
}


def reject_constant(value):
    raise ValueError('Недопустимая JSON-константа: ' + value)


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Повторяющийся JSON-ключ: ' + key)
        result[key] = value
    return result


def assess(content):
    result = dict(json_valid=False, schema_valid=False, task_valid=False, task_evaluated=False, validation_error=None)
    try:
        # Не вырезаем Markdown и не чиним JSON: проверяем именно полученный текст.
        data = json.loads(content, parse_constant=reject_constant, object_pairs_hook=unique_keys)
    except (ValueError, TypeError, RecursionError):
        result['validation_error'] = 'Исходный текст не разбирается как строгий JSON.'
        return result
    result['json_valid'] = True
    errors = list(Draft202012Validator(SCHEMA).iter_errors(data))
    if errors:
        result['validation_error'] = 'JSON не соответствует схеме: ' + errors[0].message
        return result
    result['schema_valid'] = True
    result['task_evaluated'] = True
    ideas = data['ideas']
    titles = [idea['title'].strip().casefold() for idea in ideas]
    result['task_valid'] = (len(ideas) == 3 and len(set(titles)) == 3 and all(titles)
                            and all(idea['duration_minutes'] > 0 for idea in ideas))
    if not result['task_valid']:
        result['validation_error'] = 'Нужно ровно три разных непустых названия и положительные длительности.'
    return result


def summarize(records):
    summary = []
    for variant in VARIANTS:
        rows = [r for r in records if r['variant'] == variant]
        if not rows:
            continue
        costs = [r['cost_rub'] for r in rows if r['cost_rub'] is not None]
        tokens = [r['total_tokens'] for r in rows if r['total_tokens'] is not None]
        summary.append(dict(variant=variant, runs=len(rows),
            **{key: sum(bool(r[key]) for r in rows) for key in ('json_valid', 'schema_valid', 'task_valid')},
            success=sum(bool(r['task_valid'] and r['complete'] and not r['api_error']) for r in rows),
            api_errors=sum(bool(r['api_error']) for r in rows),
            format_errors=sum(not r['json_valid'] and not r['api_error'] for r in rows),
            truncated=sum(r['finish_reason'] == 'length' for r in rows),
            duration_seconds=round(sum(r['duration_seconds'] for r in rows), 6),
            total_tokens=sum(tokens) if tokens else None,
            unknown_tokens=len(rows) - len(tokens),
            cost_rub=round(sum(costs), 10) if costs else None,
            unknown_costs=len(rows) - len(costs)))
    return summary


DETAIL_FIELDS = ['variant', 'repeat', 'http_status', 'finish_reason', 'complete',
    'json_valid', 'schema_valid', 'task_evaluated', 'task_valid', 'api_error', 'validation_error',
    'duration_seconds', 'prompt_tokens', 'completion_tokens', 'total_tokens', 'cost_rub', 'actual_model']


def csv_text(rows, fields):
    file = io.StringIO(newline='')
    writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    return file.getvalue()


def value(number):
    return '—' if number is None else f'{number:.8f}'.rstrip('0').rstrip('.') if isinstance(number, float) else str(number)


def render_report(config, records, summary):
    lines = ['# День 2 — формат ответа', '',
        f"Модель: `{config['model']}`. Дата: {config['started_at']}. Статус: {config['status']}.", '',
        f"Запрос: {PROMPT}", '',
        '## Сводная таблица', '',
        '| Вариант | JSON | Схема | Требования | Полный успех | Ошибки API | Обрезания | Токены | Время, с | Стоимость, ₽ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        n = row['runs']
        fractions = [f"{row[key]}/{n}" for key in ('json_valid', 'schema_valid', 'task_valid', 'success')]
        numbers = [value(row[key]) for key in ('api_errors', 'truncated', 'total_tokens', 'duration_seconds', 'cost_rub')]
        lines.append('| ' + ' | '.join([VARIANTS[row['variant']][0]] + fractions + numbers) + ' |')
    lines += ['', '## Выводы', '']
    lines += [f"Неизвестных стоимостей: {sum(r['unknown_costs'] for r in summary)}; "
              f"неизвестных значений токенов: {sum(r['unknown_tokens'] for r in summary)}.", '']
    for row in summary:
        lines.append(f"- **{VARIANTS[row['variant']][0]}:** JSON разбирается в {row['json_valid']}/{row['runs']} ответах; "
            f"полностью успешны {row['success']}/{row['runs']}. Ошибок API: {row['api_errors']}; "
            f"ошибок разбора среди полученных ответов: {row['format_errors']}; обрезаний по лимиту: {row['truncated']}.")
    prompted = [r for r in records if r['variant'] == 'prompt_json' and not r['api_error']]
    if prompted:
        fenced = sum(r['content'].lstrip().startswith('```') for r in prompted)
        lines += ['', f"**JSON по инструкции. Markdown-ограждения: {fenced}/{len(prompted)}.** "
            'Такая обёртка не является JSON и ломает прямой json.loads, даже если содержимое блока выглядит правильным. '
            'Просьба в промпте не обеспечивает соблюдение формата на уровне API.']
    by_variant = {r['variant']: r for r in summary}
    if 'json_mode' in by_variant and 'strict_schema' in by_variant:
        plain, strict = by_variant['json_mode'], by_variant['strict_schema']
        lines += ['', f"**JSON mode против строгой схемы:** по требованиям задания получено "
            f"{plain['success']}/{plain['runs']} и {strict['success']}/{strict['runs']} полных успешных ответов. "
            'Разница в числе успехов на этом запросе не доказывает превосходство одного способа на других задачах. '
            'Для машинной обработки при известной структуре предпочтительна схема с обязательной проверкой результата; '
            'JSON mode полезен, когда достаточно синтаксически корректного JSON без фиксированной структуры.']
    lines += ['', 'Лимит 32 токена и stop на строке `"level"` — намеренные стресс-проверки. '
        'Stop может завершить генерацию с причиной `stop` посередине JSON; причину завершения нужно проверять вместе с содержимым.', '',
        'JSON mode проверяет синтаксис, а схема — типы, обязательные поля и отсутствие лишних полей. '
        'Ровно три идеи, непустые разные названия и положительные длительности проверяются отдельно. '
        'Если структура не прошла проверку, требования к данным не оцениваются (task_evaluated=false); '
        'это не означает, что свободный текст содержательно неверен. '
        'Смысловая полезность заданий автоматически не оценивалась.', '',
        'Это наблюдения на одном запросе и одной модели, по пять повторов; они не доказывают надёжность на всех задачах. '
        'Проверки относятся к исходному ответу: Markdown не удаляется, повреждённый JSON не исправляется.', '',
        '## Подробная таблица', '',
        '| Вариант | № | Завершение | JSON | Схема | Требования | Токены | Секунды | ₽ |',
        '|---|---:|---|---|---|---|---:|---:|---:|']
    for row in records:
        values = [row['variant'], str(row['repeat']), row['finish_reason'] or 'API error']
        values += ['да' if row[key] else 'нет' for key in ('json_valid', 'schema_valid')]
        values += [('да' if row['task_valid'] else 'нет') if row.get('task_evaluated', row['schema_valid']) else '—']
        values += [value(row[key]) for key in ('total_tokens', 'duration_seconds', 'cost_rub')]
        lines.append('| ' + ' | '.join(values) + ' |')
    lines += ['', '## Примеры исходных ответов', '']
    for variant in VARIANTS:
        example = next((r for r in records if r['variant'] == variant), None)
        if example:
            preview = example['content'][:900] or example['api_error'] or '(пустой ответ)'
            lines += [f'### {VARIANTS[variant][0]}', '', '````text', preview, '````', '']
    lines += ['## Условия сравнения и источники', '',
        '- Каждый прогон — новая история с одним пользовательским сообщением. Условия не накапливают контекст.',
        '- Temperature 0.7; max_tokens 4096 во всех вариантах, кроме стресс-проверки с 32 токенами.',
        '- Варианты перемешиваются внутри каждого повтора с фиксированным seed=42.',
        '- В baseline отсутствует инструкция формата; во всех остальных она одинаковая. Системная инструкция хранится в results.json.',
        '- JSON Schema не ограничивает длину массива и положительность чисел: это отдельные требования задания.',
        '- Модель фиксирована; автоматические повторы и fallback отключены, require_parameters=true. '
        'Маршрут провайдера выбирает шлюз; точный backend и его стабильность не гарантируются.',
        '- Токены и рублёвая стоимость берутся из usage. Пропуски остаются неизвестными, а не нулями. '
        'При пропусках сумма отражает только известные значения; unknown_costs и unknown_tokens есть в summary.csv.',
        '- Время — ожидание HTTP-ответа и его разбор; это не вся длительность интерактивной сессии.',
        '- Все запросы, исходные ответы и метрики: results.json и raw/*.json. Каталог перед прогоном: catalog.json.',
        '- [Polza.ai: формат ответа](https://polza.ai/docs/gaidy/structured-output), '
        '[API](https://polza.ai/docs/api-reference/chat/completions).', '']
    return '\n'.join(lines)


def write_reports(directory, config, records):
    summary = summarize(records)
    atomic_json(directory / 'results.json', {'config': config, 'records': records})
    (directory / 'details.csv').write_text(csv_text(records, DETAIL_FIELDS), encoding='utf-8-sig')
    if summary:
        (directory / 'summary.csv').write_text(csv_text(summary, list(summary[0])), encoding='utf-8-sig')
    (directory / 'report.md').write_text(render_report(config, records, summary), encoding='utf-8')


def run_experiment(api, output_dir, repeats=5, notify=None):
    if type(repeats) is not int or repeats <= 0:
        raise ValueError('Число повторов должно быть положительным.')
    catalog = api.catalog()
    required = {'stop', 'temperature', 'max_tokens', 'response_format', 'structured_outputs'}
    supported = set(catalog.get('top_provider', {}).get('supported_parameters', []))
    if not required <= supported:
        raise ApiError('Каталог не подтверждает все необходимые параметры модели: ' + ', '.join(sorted(required - supported)))
    directory = Path(output_dir) / identifier()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'raw').mkdir()
    atomic_json(directory / 'catalog.json', catalog)
    config = dict(model=api.model, started_at=datetime.now().astimezone().isoformat(timespec='seconds'),
                  repeats=repeats, planned_requests=repeats * len(VARIANTS), prompt=PROMPT,
                  format_instruction=FORMAT, schema=SCHEMA, temperature=.7,
                  order_seed=42, status='running')
    records, randomizer = [], random.Random(42)
    write_reports(directory, config, records)
    api.journal.record('EXPERIMENT_START', config)
    try:
        for repeat in range(1, repeats + 1):
            order = list(VARIANTS)
            randomizer.shuffle(order)
            for variant in order:
                messages = [] if variant == 'baseline' else [{'role': 'system', 'content': FORMAT}]
                messages.append({'role': 'user', 'content': PROMPT})
                params = deepcopy(VARIANTS[variant][1])
                try:
                    row = api.send(messages, params)
                except ApiError as exc:
                    row = exc.result
                row = {**row, 'variant': variant, 'repeat': repeat,
                       'observed_at': datetime.now().astimezone().isoformat(timespec='seconds')}
                row.update(assess(row['content']))
                row = api.journal.scrub(row)
                records.append(row)
                atomic_json(directory / 'raw' / f'{variant}_{repeat:02d}.json', row)
                write_reports(directory, config, records)
                api.journal.record('EXPERIMENT_RESULT', row)
                if notify:
                    notify(f"{len(records)}/{config['planned_requests']}: {variant}; JSON={row['json_valid']}; schema={row['schema_valid']}; finish={row['finish_reason']}")
        config['status'] = 'complete'
    except BaseException:
        config['status'] = 'interrupted'
        raise
    finally:
        write_reports(directory, config, records)
        api.journal.record('EXPERIMENT_END', {'directory': str(directory), 'status': config['status'], 'rows': len(records)})
    return directory
