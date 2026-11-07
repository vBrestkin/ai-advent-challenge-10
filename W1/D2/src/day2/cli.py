"""Команды chat, experiment и бесплатная проверка каталога catalog."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import re
import sys

import httpx
from dotenv import dotenv_values

from .core import ApiClient, ApiError, Journal, identifier
from .experiment import run_experiment
from .history import Conversation, HistoryError

PROJECT_DIR = Path(__file__).resolve().parents[2]


class ArgumentError(Exception):
    pass


class HelpRequested(Exception):
    def __init__(self, text):
        self.text = text


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ArgumentError(message)

    def print_help(self, file=None):
        raise HelpRequested(self.format_help())


def arguments(argv):
    parser = Parser(description='Диалог с историей и сравнение шести форматов ответа')
    commands = parser.add_subparsers(dest='command', required=True)
    chat = commands.add_parser('chat', help='Новый разговор или продолжение сохранённого')
    chat.add_argument('--session', help='Имя разговора; файл истории создаётся автоматически')
    chat.add_argument('--max-tokens', type=int, default=None, help='Лимит нового разговора (4096)')
    experiment = commands.add_parser('experiment', help='Платные API-прогоны и таблицы результатов')
    experiment.add_argument('--repeats', type=int, default=5)
    catalog = commands.add_parser('catalog', help='Возможности и цены модели без генерации ответа')
    for command in (chat, experiment, catalog):
        command.add_argument('--connect-timeout', type=float, default=10)
        command.add_argument('--read-timeout', type=float, default=60)
    args = parser.parse_args(argv)
    if any(not math.isfinite(v) or v <= 0 for v in (args.connect_timeout, args.read_timeout)):
        raise ArgumentError('Таймауты должны быть положительными конечными числами.')
    if args.command == 'chat':
        if args.session and not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', args.session):
            raise ArgumentError('Имя разговора: 1–100 латинских букв, цифр, дефисов или подчёркиваний.')
        if args.max_tokens is not None and args.max_tokens <= 0:
            raise ArgumentError('Лимит токенов должен быть положительным.')
    if args.command == 'experiment' and args.repeats <= 0:
        raise ArgumentError('Число повторов должно быть положительным.')
    return args


def chat_loop(api, args, root, read, emit):
    directory = root / 'history'
    path = directory / ((args.session or identifier()) + '.json')
    if path.exists():
        conversation = Conversation.load(path)
        if conversation.model != api.model:
            raise HistoryError('LLM_MODEL отличается от модели сохранённого разговора; верните настройку или начните новый.')
        if args.max_tokens is not None and args.max_tokens != conversation.data['parameters']['max_tokens']:
            raise HistoryError('Настройки сохранённого разговора отличаются; начните новый разговор для другого лимита.')
    else:
        conversation = Conversation.create(path, api.model, {'max_tokens': args.max_tokens or 4096})
    emit(f'Разговор: {conversation.path.stem}; модель: {conversation.model}; лимит: {conversation.data["parameters"]["max_tokens"]}')
    emit('Команды: /new — новый разговор; /history — история; /exit — выход.')
    while True:
        api.journal.record('INPUT_PROMPT', 'Вы> ')
        try:
            text = read('Вы> ')
        except EOFError:
            emit('Диалог завершён (EOF).')
            return
        api.journal.record('INPUT', text)
        text = text.strip()
        if text == '/exit':
            emit('Диалог завершён. История сохранена.')
            return
        if text == '/new':
            conversation = Conversation.create(directory / (identifier() + '.json'), api.model,
                                               conversation.data['parameters'])
            emit('Новый разговор: ' + conversation.path.stem)
        elif text == '/history':
            emit(json.dumps(conversation.messages, ensure_ascii=False, indent=2))
        elif not text:
            emit('Пустое сообщение не отправлено.')
        elif text.startswith('/'):
            emit('Неизвестная команда. Доступны /new, /history, /exit.')
        else:
            try:
                answer = conversation.ask(text, api)
                emit('Модель> ' + answer)
            except (ApiError, HistoryError) as exc:
                emit('Ошибка: ' + str(exc), error=True)


def main(argv=None, *, root=None, env=None, client=None, read=None, stdout=None, stderr=None):
    root = Path(root) if root is not None else PROJECT_DIR
    stdout, stderr = stdout or sys.stdout, stderr or sys.stderr
    read = input if read is None else read
    journal = None
    try:
        journal = Journal(root / 'logs')

        def emit(text, error=False):
            text = journal.scrub(text)
            journal.record('STDERR' if error else 'STDOUT', text)
            print(text, file=stderr if error else stdout, flush=True)

        settings = ({**dotenv_values(root.parent / 'D1' / '.env', interpolate=False),
                     **dotenv_values(root / '.env', interpolate=False), **os.environ}
                    if env is None else dict(env))
        key = (settings.get('POLZA_API_KEY') or '').strip()
        model = (settings.get('LLM_MODEL') or '').strip()
        journal.secret = key
        try:
            args = arguments(argv)
            if not key or not model:
                raise ArgumentError('Задайте POLZA_API_KEY и LLM_MODEL в .env D2, .env D1 или окружении.')
        except HelpRequested as exc:
            emit(exc.text.rstrip())
            journal.record('END', 0)
            return 0
        except ArgumentError as exc:
            emit('Ошибка настроек: ' + str(exc), error=True)
            journal.record('END', 2)
            return 2
        manager = nullcontext(client) if client is not None else httpx.Client(
            transport=httpx.HTTPTransport(retries=0), follow_redirects=False)
        with manager as connection:
            api = ApiClient(key, model, journal, connection,
                connect_timeout=args.connect_timeout, read_timeout=args.read_timeout)
            try:
                if args.command == 'chat':
                    chat_loop(api, args, root, read, emit)
                elif args.command == 'catalog':
                    emit(json.dumps(api.catalog(), ensure_ascii=False, indent=2))
                else:
                    emit(f'Эксперимент: {args.repeats * 6} платных запросов, модель {model}.')
                    result = run_experiment(api, root / 'results', repeats=args.repeats, notify=emit)
                    emit('Таблицы и выводы: ' + str(result / 'report.md'))
            except (ApiError, HistoryError) as exc:
                emit('Ошибка: ' + str(exc), error=True)
                journal.record('END', 1)
                return 1
        journal.record('END', 0)
        return 0
    except KeyboardInterrupt:
        if journal is not None:
            text = 'Прервано пользователем; автоматического повтора нет.'
            journal.record('STDERR', text)
            print(text, file=stderr)
            journal.record('END', 130)
        return 130
    except OSError:
        text = 'Ошибка: журнал, история или файлы результатов недоступны. Работа остановлена.'
        if journal is not None:
            try:
                journal.record('STDERR', text)
                journal.record('END', 1)
            except OSError:
                pass
        print(text, file=stderr)
        return 1
    except Exception:
        text = 'Ошибка: непредвиденный сбой. Сырые исключения скрыты для защиты ключа.'
        if journal is not None:
            journal.record('STDERR', text)
            journal.record('END', 1)
        print(text, file=stderr)
        return 1
    finally:
        if journal is not None:
            journal.close()


def entrypoint():
    raise SystemExit(main())
