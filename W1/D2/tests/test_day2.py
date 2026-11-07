import io
import json
from pathlib import Path

import httpx
import pytest
from pytest_bdd import given, when, then, parsers, scenarios

from day2.core import ApiClient, ApiError, Journal
from day2.history import Conversation, HistoryError
from day2.experiment import VARIANTS, SCHEMA, assess, run_experiment, summarize
from day2.cli import main

scenarios('features/day2.feature')
SECRET = 'private-test-key'
MODEL = 'google/gemini-2.5-flash-lite'
VALID = {'ideas': [dict(title=f'Задача {i}', level='beginner', duration_minutes=15) for i in range(3)]}


def completion(content=None, reason='stop'):
    return {'model': MODEL, 'choices': [{'finish_reason': reason, 'message': {
        'role': 'assistant', 'content': json.dumps(VALID, ensure_ascii=False) if content is None else content}}],
        'usage': {'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30, 'cost_rub': 0.01}}


@pytest.fixture
def state(tmp_path):
    journal = Journal(tmp_path / 'logs', SECRET)
    value = dict(root=tmp_path, journal=journal, calls=[], kind='ok')

    def respond(request):
        if request.method == 'GET':
            return httpx.Response(200, json={'data': [{'id': MODEL, 'top_provider': {
                'supported_parameters': ['stop', 'temperature', 'max_tokens', 'response_format', 'structured_outputs'],
                'pricing': {'prompt_per_million': '6', 'completion_per_million': '24', 'currency': 'RUB'}}}]})
        value['calls'].append(json.loads(request.content))
        kind = value['kind']
        if kind == 'timeout':
            raise httpx.ReadTimeout(SECRET, request=request)
        if kind == 'reset':
            raise httpx.ReadError(SECRET, request=request)
        if kind == 'http_503':
            return httpx.Response(503, text=SECRET)
        if kind == 'malformed':
            return httpx.Response(200, text='broken ' + SECRET)
        reason = {'length': 'length', 'refusal': 'content_filter'}.get(kind, 'stop')
        return httpx.Response(200, json=completion(reason=reason))

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        value['api'] = ApiClient(SECRET, MODEL, journal, client)
        yield value
    journal.close()


@given('создан разговор с успешным первым ходом')
def first_turn(state):
    state['conversation'] = Conversation.create(state['root'] / 'chat.json', MODEL)
    state['conversation'].ask('Меня зовут Анна.', state['api'])
    state['saved'] = (state['root'] / 'chat.json').read_bytes()


@when('разговор восстановлен и отправлено второе сообщение')
def continue_turn(state):
    state['conversation'] = Conversation.load(state['root'] / 'chat.json')
    state['conversation'].ask('Как меня зовут?', state['api'])


@then('API получает стек из трёх сообщений')
def stack(state):
    messages = state['calls'][-1]['messages']
    assert [m['role'] for m in messages] == ['user', 'assistant', 'user']
    assert messages[0]['content'] == 'Меня зовут Анна.'
    assert messages[2]['content'] == 'Как меня зовут?'


@then('история содержит два полных хода')
def two_turns(state):
    assert len(Conversation.load(state['root'] / 'chat.json').messages) == 4


@when(parsers.parse('следующий ответ завершился как "{kind}"'))
def failed_turn(state, kind):
    state['kind'] = kind
    with pytest.raises((ApiError, HistoryError)):
        state['conversation'].ask('Новое сообщение', state['api'])


@then('сохранённая история первого хода не меняется')
def unchanged(state):
    assert (state['root'] / 'chat.json').read_bytes() == state['saved']
    assert len(state['conversation'].messages) == 2
    assert len(state['calls']) == 2


@given('API возвращает валидные ответы для эксперимента')
def valid_api(state):
    pass


@when('выполнены пять повторов каждого варианта')
def experiment(state):
    state['result_dir'] = run_experiment(state['api'], state['root'] / 'results', repeats=5)


@then('сохранены тридцать исходных результатов')
def records(state):
    data = json.loads((state['result_dir'] / 'results.json').read_text())
    assert len(data['records']) == 30
    assert len(list((state['result_dir'] / 'raw').glob('*.json'))) == 30
    assert {r['variant'] for r in data['records']} == set(VARIANTS)
    assert all('request' in r and 'response' in r for r in data['records'])
    assert sum(r['cost_rub'] for r in data['records']) == pytest.approx(.30)


@then('созданы подробная и сводная таблицы с выводами')
def tables(state):
    import csv
    root = state['result_dir']
    assert len(list(csv.DictReader((root / 'details.csv').open()))) == 30
    summary = list(csv.DictReader((root / 'summary.csv').open()))
    assert len(summary) == 6
    assert all(int(row['schema_valid']) == 5 for row in summary)
    assert 'Выводы' in (root / 'report.md').read_text()
    assert '5/5' in (root / 'report.md').read_text()


@then('все прогоны начинаются с пустой истории')
def independent(state):
    assert len(state['calls']) == 30
    assert all(sum(m['role'] == 'user' for m in call['messages']) == 1 for call in state['calls'])
    assert all(not any(m['role'] == 'assistant' for m in call['messages']) for call in state['calls'])
    assert all(call['temperature'] == .7 for call in state['calls'])
    assert {call['max_tokens'] for call in state['calls']} == {32, 4096}
    assert sum('stop' in call for call in state['calls']) == 5
    strict = [c for c in state['calls'] if c.get('response_format', {}).get('type') == 'json_schema']
    assert len(strict) == 5 and strict[0]['response_format']['json_schema']['strict'] is True


@pytest.mark.parametrize('error', [httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
    httpx.WriteTimeout, httpx.PoolTimeout, httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError])
def test_network_errors_are_safe_and_never_retried(state, error):
    calls = []
    def respond(request):
        calls.append(request)
        raise error(SECRET, request=request)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        api = ApiClient(SECRET, MODEL, state['journal'], client)
        with pytest.raises(ApiError) as exc:
            api.send([{'role': 'user', 'content': 'hi'}])
    assert SECRET not in str(exc.value)
    assert len(calls) == 1
    assert SECRET not in state['journal'].path.read_text()


@pytest.mark.parametrize('status', [301, 400, 401, 402, 403, 404, 408, 413, 429, 500, 502, 503, 504])
def test_http_error_is_not_format_failure(state, status):
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(status, text=SECRET))) as client:
        api = ApiClient(SECRET, MODEL, state['journal'], client)
        with pytest.raises(ApiError) as exc:
            api.send([{'role': 'user', 'content': 'hi'}])
    assert str(status) in str(exc.value)
    assert SECRET not in state['journal'].path.read_text()


@pytest.mark.parametrize('body', [None, [], {}, {'choices': []}, {'choices': [{'message': {}}]},
    {'choices': [{'finish_reason': 'stop', 'message': {'content': ['bad']}}]}])
def test_malformed_api_body(state, body):
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=json.dumps(body)))) as client:
        with pytest.raises(ApiError):
            ApiClient(SECRET, MODEL, state['journal'], client).send([{'role': 'user', 'content': 'hi'}])


@pytest.mark.parametrize('text,json_ok,schema_ok,task_ok', [
    ('текст', False, False, False), ('```json\n{}\n```', False, False, False),
    ('{"ideas": []}', True, True, False),
    ('{"ideas": [{"title": "A", "level": "beginner", "duration_minutes": "15"}]}', True, False, False),
    (json.dumps(VALID), True, True, True),
    ('NaN', False, False, False),
    ('{"ideas": [], "ideas": []}', False, False, False)])
def test_parse_schema_and_task_are_distinct(text, json_ok, schema_ok, task_ok):
    result = assess(text)
    assert result['json_valid'] is json_ok
    assert result['schema_valid'] is schema_ok
    assert result['task_valid'] is task_ok


def test_length_can_contain_parseable_json_without_being_success(state):
    state['kind'] = 'length'
    result = state['api'].send([{'role': 'user', 'content': 'hi'}])
    assert result['finish_reason'] == 'length'
    assert result['complete'] is False
    assert assess(result['content'])['json_valid'] is True


def test_missing_usage_remains_unknown_not_zero(state):
    body = completion()
    body.pop('usage')
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body))) as client:
        result = ApiClient(SECRET, MODEL, state['journal'], client).send([{'role': 'user', 'content': 'hi'}])
    assert result['cost_rub'] is None and result['total_tokens'] is None


def test_failure_during_experiment_preserves_all_rows(state):
    state['kind'] = 'http_503'
    directory = run_experiment(state['api'], state['root'] / 'results', repeats=1)
    rows = json.loads((directory / 'results.json').read_text())['records']
    assert len(rows) == 6
    assert all(row['api_error'] and row['cost_rub'] is None for row in rows)
    assert all(not row['json_valid'] for row in rows)
    assert len(state['calls']) == 6


@pytest.mark.parametrize('body', ['broken', '{}', '{"version":1,"messages":[]}'])
def test_corrupt_history_is_not_overwritten(tmp_path, body):
    path = tmp_path / 'chat.json'
    path.write_text(body)
    with pytest.raises(HistoryError):
        Conversation.load(path)
    assert path.read_text() == body


def test_history_write_failure_does_not_change_memory_or_disk(state, monkeypatch):
    first_turn(state)
    import day2.history as history
    def fail(*args):
        raise OSError('disk full')
    monkeypatch.setattr(history.os, 'replace', fail)
    with pytest.raises(HistoryError):
        state['conversation'].ask('Не сохраняется', state['api'])
    unchanged(state)


def test_history_model_mismatch_does_not_call_api(state):
    chat = Conversation.create(state['root'] / 'chat.json', 'different/model')
    with pytest.raises(HistoryError):
        chat.ask('hi', state['api'])
    assert not state['calls']


def test_journal_same_second_unique_and_redacts(state):
    with Journal(state['root'] / 'logs', SECRET) as second:
        second.record('TEST', SECRET)
        assert second.path != state['journal'].path
        assert SECRET not in second.path.read_text()


def test_no_call_when_journal_unwritable(tmp_path):
    path = tmp_path / 'file'
    path.write_text('occupied')
    with pytest.raises(OSError):
        Journal(path, SECRET)


def cli(state, args, lines=()):
    out, err = io.StringIO(), io.StringIO()
    iterator = iter(lines)
    def read(prompt):
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError
    code = main(args, root=state['root'], env={'POLZA_API_KEY': SECRET, 'LLM_MODEL': MODEL},
                client=state['api'].client, read=read, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_cli_chat_commands_and_resume(state):
    code, out, err = cli(state, ['chat', '--session', 'demo'], ['hi', '/history', '/exit'])
    assert code == 0 and 'user' in out and not err
    code, out, err = cli(state, ['chat', '--session', 'demo'], ['remember?', '/new', '/exit'])
    assert code == 0 and not err
    assert len(state['calls'][-1]['messages']) == 3
    assert len(list((state['root'] / 'history').glob('*.json'))) == 2


@pytest.mark.parametrize('args', [[], ['unknown'], ['chat', '--session', '../outside'],
    ['experiment', '--repeats', '0'], ['chat', '--max-tokens', '0'],
    ['chat', '--read-timeout', 'nan'], ['chat', '--connect-timeout', '-1']])
def test_cli_invalid_args_logged_without_api(state, args):
    code, out, err = cli(state, args)
    assert code == 2 and 'Ошибка' in err and not state['calls']
    assert any('Ошибка' in p.read_text() for p in (state['root'] / 'logs').glob('*.log'))


def test_cli_help_has_log_and_no_api(state):
    code, out, err = cli(state, ['--help'])
    assert code == 0 and 'usage:' in out and not state['calls']


def test_cli_catalog_does_not_generate_paid_answer(state):
    code, out, err = cli(state, ['catalog'])
    assert code == 0 and 'supported_parameters' in out and not state['calls']


def test_cli_ctrl_c_is_logged(state):
    def interrupt(prompt):
        raise KeyboardInterrupt
    err = io.StringIO()
    code = main(['chat'], root=state['root'], env={'POLZA_API_KEY': SECRET, 'LLM_MODEL': MODEL},
        client=state['api'].client, read=interrupt, stdout=io.StringIO(), stderr=err)
    assert code == 130 and 'Прервано' in err.getvalue()
    assert any('Прервано' in p.read_text() for p in (state['root'] / 'logs').glob('*.log'))


def test_cli_empty_and_unknown_commands_do_not_call_api(state):
    code, out, err = cli(state, ['chat'], ['', '/wrong', '/exit'])
    assert code == 0 and not state['calls']


def test_cli_recovers_after_network_error(state):
    state['kind'] = 'timeout'
    code, out, err = cli(state, ['chat'], ['hi', '/exit'])
    assert code == 0 and 'Ошибка' in err
    assert all(not json.loads(p.read_text())['messages'] for p in (state['root'] / 'history').glob('*.json'))


def test_experiment_records_redact_secret(state):
    state['kind'] = 'malformed'
    directory = run_experiment(state['api'], state['root'] / 'results', repeats=1)
    for file in directory.rglob('*'):
        if file.is_file():
            assert SECRET not in file.read_text()


def test_unsupported_model_parameters_abort_before_paid_calls(state, monkeypatch):
    monkeypatch.setattr(state['api'], 'catalog', lambda: {'id': MODEL, 'top_provider': {
        'supported_parameters': ['max_tokens'], 'pricing': {}}})
    with pytest.raises(ApiError):
        run_experiment(state['api'], state['root'] / 'results', repeats=5)
    assert not state['calls']


def test_summary_keeps_unknown_cost_and_counts_api_separately():
    rows = [dict(variant='baseline', json_valid=False, schema_valid=False, task_valid=False,
        api_error='HTTP 503', finish_reason=None, complete=False,
        duration_seconds=1, cost_rub=None, total_tokens=None)]
    row = summarize(rows)[0]
    assert row['api_errors'] == 1 and row['format_errors'] == 0
    assert row['cost_rub'] is None and row['unknown_costs'] == 1


def test_history_changed_by_another_process_is_not_overwritten(state):
    first_turn(state)
    path = state['root'] / 'chat.json'
    changed = state['saved'] + b'\n'
    path.write_bytes(changed)
    with pytest.raises(HistoryError):
        state['conversation'].ask('hi', state['api'])
    assert path.read_bytes() == changed and len(state['calls']) == 1


@pytest.mark.parametrize('messages', [[{'role': 'user', 'content': 'orphan'}],
    [{'role': 'assistant', 'content': 'wrong'}, {'role': 'user', 'content': 'order'}],
    [{'role': 'user', 'content': ''}, {'role': 'assistant', 'content': 'answer'}]])
def test_invalid_history_stack_rejected(state, messages):
    path = state['root'] / 'invalid.json'
    path.write_text(json.dumps(dict(version=1, model=MODEL,
        parameters={'max_tokens': 4096, 'temperature': .7}, messages=messages)))
    with pytest.raises(HistoryError):
        Conversation.load(path)


def test_provider_generation_error_is_api_error(state):
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=completion(reason='error')))) as client:
        with pytest.raises(ApiError):
            ApiClient(SECRET, MODEL, state['journal'], client).send([{'role': 'user', 'content': 'hi'}])


def test_explicit_chat_limit_saved_and_reused(state):
    assert cli(state, ['chat', '--session', 'limit', '--max-tokens', '100'], ['hi', '/exit'])[0] == 0
    assert state['calls'][-1]['max_tokens'] == 100
    assert cli(state, ['chat', '--session', 'limit'], ['again', '/exit'])[0] == 0
    assert state['calls'][-1]['max_tokens'] == 100


def test_cli_missing_credentials_logs_error_without_call(state):
    err = io.StringIO()
    code = main(['chat'], root=state['root'], env={}, client=state['api'].client,
                stdout=io.StringIO(), stderr=err)
    assert code == 2 and 'POLZA_API_KEY' in err.getvalue() and not state['calls']


def test_interrupted_experiment_saves_partial_report(state, monkeypatch):
    original = state['api'].send
    def send(messages, params):
        if state['calls']:
            raise KeyboardInterrupt
        return original(messages, params)
    monkeypatch.setattr(state['api'], 'send', send)
    with pytest.raises(KeyboardInterrupt):
        run_experiment(state['api'], state['root'] / 'results', repeats=5)
    path = next((state['root'] / 'results').glob('*/results.json'))
    data = json.loads(path.read_text())
    assert data['config']['status'] == 'interrupted' and len(data['records']) == 1


def test_catalog_network_failure_never_sends_generation(state):
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503))) as client:
        with pytest.raises(ApiError):
            ApiClient(SECRET, MODEL, state['journal'], client).catalog()


def test_missing_costs_are_explicit_in_markdown_report(state, monkeypatch):
    import day2.experiment as experiment
    original = state['api'].send
    def send(*args):
        row = original(*args)
        row['cost_rub'] = None
        return row
    monkeypatch.setattr(state['api'], 'send', send)
    directory = experiment.run_experiment(state['api'], state['root'] / 'results', repeats=1)
    assert 'Неизвестных стоимостей: 6' in (directory / 'report.md').read_text()


def test_report_explains_observed_markdown_cause(state, monkeypatch):
    original = state['api'].send
    def send(messages, parameters):
        row = original(messages, parameters)
        if not parameters and messages[0]['role'] == 'system':
            row['content'] = '```json\n' + row['content'] + '\n```'
        return row
    monkeypatch.setattr(state['api'], 'send', send)
    directory = run_experiment(state['api'], state['root'] / 'results', repeats=1)
    text = (directory / 'report.md').read_text()
    assert 'Markdown-ограждения: 1/1' in text
    assert 'json.loads' in text


def test_task_checks_are_marked_not_evaluated_when_structure_is_invalid():
    assert assess('обычный текст')['task_evaluated'] is False
    assert assess('{"wrong": 1}')['task_evaluated'] is False
    assert assess('{"ideas": []}')['task_evaluated'] is True
