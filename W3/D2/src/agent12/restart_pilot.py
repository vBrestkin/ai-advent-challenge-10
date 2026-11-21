"""Платная проверка контекста через два отдельных процесса HTTP-сервера."""
from contextlib import contextmanager
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

from .log import Journal, identifier
from .web import ROOT

FIRST = 'Запомни: проект называется Маяк, менеджер Ирина. Найди сумму заказа A-104.'
SECOND = 'Напомни название проекта, имя менеджера и ранее найденную сумму заказа. Используй сохранённую историю, без повторного поиска.'


def verify(before, restored, after):
    answer = after['messages'][-1].get('content', '').casefold()
    return {
        'history_restored_exactly': before['messages'] == restored['messages'],
        'browser_turns_restored': before['turns'] == restored['turns'],
        'tool_result_restored': any(message.get('role') == 'tool' and '2390' in message.get('content', '')
                                    for message in restored['messages']),
        'details_recalled': all(detail in answer for detail in ('маяк', 'ирина', '2390')),
        'no_tool_reexecution': len(after['messages']) == len(before['messages']) + 2,
        'second_turn_saved': len(after['turns']) == len(before['turns']) + 1,
    }


@contextmanager
def server(directory, phase, port, journal):
    command = [sys.executable, '-m', 'agent12.web', '--port', str(port), '--data-dir', str(directory / 'data')]
    output = (directory / f'{phase}-server.txt').open('w', encoding='utf-8')
    process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
    journal.record('PROCESS_START', {'phase': phase, 'pid': process.pid, 'port': port})
    try:
        with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=5) as client:
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError('HTTP-сервер остановился при запуске. Посмотрите журнал сервера.')
                try:
                    health = client.get('/health')
                    if health.status_code == 200:
                        if not health.json()['ready']:
                            raise RuntimeError('Не настроен POLZA_API_KEY.')
                        break
                except httpx.RequestError:
                    pass
                time.sleep(.1)
            else:
                raise RuntimeError('Не удалось дождаться запуска HTTP-сервера.')
        yield process.pid
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        output.close()
        journal.record('PROCESS_STOP', {'phase': phase, 'pid': process.pid, 'returncode': process.returncode})


def main():
    directory = ROOT / 'results' / identifier()
    directory.mkdir(parents=True)
    report = {'verified': False, 'directory': str(directory)}
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    with Journal(ROOT / 'logs') as journal:
        report['pilot_log'] = journal.path.name
        try:
            with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=400) as client:
                with server(directory, 'before', port, journal) as first_pid:
                    response = client.post('/api/chat', json={'message': FIRST})
                    response.raise_for_status()
                    result = response.json()
                    session_id = result['session_id']
                    response = client.get('/api/sessions/' + session_id)
                    response.raise_for_status()
                    before = response.json()
                # Контекстный менеджер уже дождался завершения первого процесса.
                with server(directory, 'after', port, journal) as second_pid:
                    response = client.get('/api/sessions/' + session_id)
                    response.raise_for_status()
                    restored = response.json()
                    response = client.post('/api/chat', json={'session_id': session_id, 'message': SECOND})
                    response.raise_for_status()
                    response = client.get('/api/sessions/' + session_id)
                    response.raise_for_status()
                    after = response.json()
            checks = verify(before, restored, after)
            checks['different_processes'] = first_pid != second_pid
            report.update(verified=all(checks.values()), checks=checks, session_id=session_id,
                          before_pid=first_pid, after_pid=second_pid,
                          before=before, restored=restored, after=after)
        except (RuntimeError, httpx.HTTPError, KeyError, ValueError) as exc:
            report['error'] = str(exc)
        journal.record('PILOT_RESULT', report)
        (directory / 'restart.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'verified': report['verified'], 'report': str(directory / 'restart.json'),
                          'checks': report.get('checks'), 'error': report.get('error')}, ensure_ascii=False, indent=2))
    return 0 if report['verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
