"""Реальные диалоги и переполнение провайдера в общем бюджете одного запуска."""
import csv
import io
import json
from pathlib import Path
from fastapi.testclient import TestClient

from .agent import AgentError
from .log import Journal, LoggingParser, identifier
from .metrics import Budget, aggregate, TOKENIZER_SHA256, TOKENIZER_URL
from .transport import MODEL, PROVIDER
from .web import ROOT, create_app, load_settings


NOTE = ('В учебном проекте команда обсуждает интерфейс, документацию и порядок проверки. '
        'Каждая запись сохраняется в полном виде, участники уточняют требования и примеры. '
        'После демонстрации результаты будут сопоставлены по токенам и стоимости. ')


def evaluate(records):
    overflow = next((row for row in records if row['scenario'] == 'overflow'), None)
    sent = [] if overflow is None else [call for call in overflow['calls'] if call.get('sent', True)]
    rejected = any(call.get('http_status') in (400, 413, 422) and any(
        word in str(call.get('response', '')).lower() for word in ('context', 'token', 'length', 'maximum')) for call in sent)
    successful = [row for row in records if row['scenario'] in ('short', 'long')]
    return {'short_and_long_completed': bool(successful) and all(row['http_status'] == 200 for row in successful),
            'overflow_rejected_by_api': rejected,
            'overflow_history_preserved': overflow is not None and overflow['history_before'] == overflow['history_after']}


def csv_text(rows):
    if not rows:
        return ''
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def show(value):
    if value is None:
        return '—'
    return f'{value:.6f}'.rstrip('0').rstrip('.') if isinstance(value, float) else str(value)


def write_report(directory, data):
    (directory / 'experiment.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = []
    cumulative = 0.
    unknown_costs = 0
    for index, row in enumerate(data['records'], 1):
        metrics = aggregate(row['calls'])
        cumulative += metrics.get('cost_rub') or 0
        unknown_costs += metrics['unknown_costs']
        rows.append({'turn': index, 'scenario': row['scenario'], 'http_status': row['http_status'],
            'new_text_tokens_local': row.get('prompt_tokens_local'), 'request_tokens_estimate': row.get('request_tokens_estimate'),
            'http_calls': metrics['calls'], 'prompt_tokens': metrics['prompt_tokens'],
            'completion_tokens': metrics['completion_tokens'], 'reasoning_tokens': metrics['reasoning_tokens'],
            'total_tokens': metrics['total_tokens'], 'cost_rub_known': metrics['cost_rub'],
            'unknown_costs': metrics['unknown_costs'], 'cumulative_known_cost_rub': round(cumulative, 10),
            'cumulative_unknown_costs': unknown_costs})
    (directory / 'turns.csv').write_text(csv_text(rows), encoding='utf-8-sig')
    summary = []
    for scenario in dict.fromkeys(row['scenario'] for row in data['records']):
        selected = [row for row in data['records'] if row['scenario'] == scenario]
        calls = [call for row in selected for call in row['calls']]
        summary.append({'scenario': scenario, 'turns': len(selected), 'successful_turns': sum(row['http_status'] == 200 for row in selected), **aggregate(calls)})
    (directory / 'summary.csv').write_text(csv_text(summary), encoding='utf-8-sig')
    budget = data['budget']
    lines = ['# День 8 — токены и переполнение контекста', '',
        f"Модель `{MODEL}`, запрошенный маршрут `{PROVIDER}`. Статус: {data['status']}.", '',
        f"Контекст маршрута по каталогу: {data['context_limit']} токенов. Общий лимит генерации 4096.", '',
        '## Сравнение сценариев', '',
        '| Сценарий | Ходов | Успешных | HTTP-вызовов | Вход, известные | Выход, известные | Всего, известные | Стоимость, ₽ (известная) | Неизвестных списаний |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        lines.append('| ' + ' | '.join(show(row[key]) for key in ('scenario', 'turns', 'successful_turns', 'calls', 'prompt_tokens', 'completion_tokens', 'total_tokens', 'cost_rub', 'unknown_costs')) + ' |')
    lines += ['', '## Рост по ходам', '',
        '| Ход | Сценарий | Новый текст, локально | Запрос, оценка | Вход API | Выход API | Известная цена, ₽ | Накоплено известных, ₽ |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| ' + ' | '.join(show(row[key]) for key in ('turn', 'scenario', 'new_text_tokens_local', 'request_tokens_estimate', 'prompt_tokens', 'completion_tokens', 'cost_rub_known', 'cumulative_known_cost_rub')) + ' |')
    lines += ['', '## Переполнение', '']
    for row in data['records']:
        if row['scenario'] in ('overflow', 'local_guard'):
            lines += [f"### {row['scenario']}", '', f"HTTP приложения: {row['http_status']}. История сохранена: {row['history_before'] == row['history_after']}.", '']
            for call in row['calls']:
                if call.get('status') != 'complete':
                    text = str(call.get('response') or call.get('status'))
                    lines += ['```text', text[:4000], '```', '']
    lines += ['## Проверки', '', json.dumps(data['checks'], ensure_ascii=False, indent=2), '',
        '## Выводы и условия', '',
        '- Каждый новый запрос передаёт всю историю, системную инструкцию, схемы инструментов и результаты их вызовов.',
        '- Локальный подсчёт текста использует фиксированный токенизатор Qwen. Оценка полного запроса токенизирует JSON, '
        'она отличается от серверного оформления ролей и инструментов. Фактические prompt/completion/total берутся из usage API.',
        '- Метрики хода суммируют все обращения к LLM, включая этапы вызова инструментов. Reasoning уже входит в completion и total.',
        '- Для long используется один растущий разговор. short и pilot_tools начинают независимые разговоры. '
        'Переполнение проверяется на продолжении long, поэтому можно проверить сохранность прежней истории.',
        '- В лабораторной проверке overflow явно разрешён выход за локальную оценку. Ошибка локального ограничителя '
        'local_guard показывается отдельно. HTTP 400 провайдера и HTTP 502 приложения сохранены раздельно.',
        '- При ошибке успешный контекст не дополняется незавершённым ходом. Сам запрос и метрики остаются в attempts/calls.',
        '- Отсутствующий usage не означает бесплатный вызов. При неизвестном списании сохраняется резерв, он не подставляется вместо фактической цены.',
        '- Сжатие и усечение истории отключены. Порядок запросов фиксирован, автоматических повторов и fallback нет.', '',
        '## Бюджет', '',
        f"Известная стоимость: {show(budget['known_cost_rub'])} ₽. Резерв неизвестных списаний: {show(budget['unknown_reserve_rub'])} ₽. "
        f"Учтено {show(budget['accounted_rub'])} ₽ из {show(budget['limit_rub'])} ₽.", '',
        'Резерв входа оценивается по UTF-8 байтам запроса с запасом 2048, выхода — по max_tokens и тарифам +1%. '
        'Это локальная защита, а не банковская гарантия. Стоимость берётся из usage.cost_rub.', '',
        'Полные данные — experiment.json, turns.csv, summary.csv и raw/*.json. Каталог и сведения о токенизаторе включены в experiment.json.']
    (directory / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(root=ROOT, *, budget_rub=20):
    root = Path(root)
    directory = root / 'results' / identifier()
    directory.mkdir(parents=True)
    budget = Budget(budget_rub)
    data = {'status': 'initializing', 'records': [], 'budget': budget.accounting(), 'checks': {},
            'context_limit': 40960, 'catalog': {}, 'tokenizer': {'url': TOKENIZER_URL, 'sha256': TOKENIZER_SHA256}}

    def save():
        data['budget'] = budget.accounting()
        data['checks'] = evaluate(data['records'])
        write_report(directory, data)

    save()
    try:
        app = create_app(root=root, data_dir=directory / 'data', budget=budget, allow_overflow=True)
        with TestClient(app) as client:
            if not app.state.ready:
                raise RuntimeError('Не настроен POLZA_API_KEY.')
            if not app.state.counter.exact_text:
                raise RuntimeError('Сначала установите фиксированный токенизатор: python -m agent8.setup_tokenizer')
            data['catalog'] = app.state.catalog
            data['context_limit'] = app.state.model.context_limit
            data['tokenizer']['method'] = app.state.counter.method
            app.state.journal.record('EXPERIMENT_START', {'directory': str(directory), 'budget': budget_rub})

            def turn(scenario, prompt, session_id=''):
                before = client.get('/api/sessions/' + session_id).json()['messages'] if session_id else []
                first_record = len(app.state.model.records)
                estimate = client.post('/api/estimate', json={'message': prompt, 'session_id': session_id}).json()
                response = client.post('/api/chat', json={'message': prompt, 'session_id': session_id})
                result = response.json()
                sid = result.get('session_id') or response.headers.get('X-Session-ID') or session_id
                if not sid:
                    raise RuntimeError('Не получен идентификатор экспериментального разговора.')
                snapshot = client.get('/api/sessions/' + sid)
                snapshot.raise_for_status()
                after = snapshot.json()
                row = {'scenario': scenario, 'session_id': sid, 'prompt': prompt,
                       'http_status': response.status_code, 'response': result,
                       'prompt_tokens_local': app.state.counter.count(prompt),
                       'request_tokens_estimate': estimate.get('request_tokens_estimate'),
                       'history_before': before, 'history_after': after['messages'],
                       'calls': app.state.model.records[first_record:]}
                data['records'].append(row)
                raw = directory / 'raw'
                raw.mkdir(exist_ok=True)
                (raw / f'{len(data["records"]):02d}_{scenario}.json').write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding='utf-8')
                save()
                message = f"{scenario}: HTTP {response.status_code}, API-вызовов {aggregate(row['calls'])['calls']}, учтено {budget.accounting()['accounted_rub']:.4f}/{budget_rub} ₽"
                app.state.journal.record('STDOUT', message)
                print(message, flush=True)
                return row

            pilot = turn('pilot_tools', 'Найди сумму заказа A-104 и рассчитай цену со скидкой 15%.')
            names = [step['tool'] for step in pilot['response'].get('steps', [])]
            if pilot['http_status'] != 200 or names != ['find_order', 'calculate']:
                raise RuntimeError('Пилот инструментов не пройден. Основной эксперимент остановлен.')
            short = turn('short', 'Объясни, что такое JSON, одним предложением.')
            if short['http_status'] != 200:
                raise RuntimeError('Короткий запрос не завершён.')
            sid = ''
            for index in range(1, 9):
                prompt = f'Учебная запись номер {index}.\n' + NOTE * 24 + '\nПодтверди получение одним словом ПРИНЯТО. Не перечисляй записи.'
                row = turn('long', prompt, sid)
                sid = row['session_id']
                if row['http_status'] != 200:
                    raise RuntimeError('Длинный разговор остановлен после ошибки, повторов нет.')
            overflow_text = 'Тест большого контекста.\n' + ' alpha' * 50000 + '\nОтветь одним словом ОК.'
            if app.state.counter.count(overflow_text) <= data['context_limit']:
                raise RuntimeError('Проверочный текст не превышает заявленный контекст.')
            turn('overflow', overflow_text, sid)
            app.state.model.allow_overflow = False
            turn('local_guard', overflow_text, sid)
            data['status'] = 'complete' if all(data['checks'].values()) else 'observed_unexpected_behavior'
    except (RuntimeError, AgentError, ValueError, OSError, httpx.HTTPError) as exc:
        data['status'] = 'stopped'
        data['error'] = str(exc)
    except BaseException:
        data['status'] = 'interrupted'
        raise
    finally:
        save()
    return directory


def main():
    key = (load_settings(ROOT).get('POLZA_API_KEY') or '').strip()
    with Journal(ROOT / 'logs', key) as journal:
        parser = LoggingParser(journal=journal, description='День 8: реальные короткий, длинный и переполненный диалоги')
        parser.add_argument('--budget-rub', type=float, default=20)
        args = parser.parse_args()
        try:
            directory = run(budget_rub=args.budget_rub)
            result = json.loads((directory / 'experiment.json').read_text())
            message = f"Отчёт: {directory / 'report.md'}; статус: {result['status']}"
            journal.record('STDOUT', message)
            print(message)
            return 0 if result['status'] == 'complete' else 1
        except (ValueError, OSError) as exc:
            journal.record('ERROR', str(exc))
            print('Эксперимент остановлен. Проверьте журнал.')
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
