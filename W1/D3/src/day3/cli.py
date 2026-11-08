"""Диалог с reasoning, проверка каталога, пилот и основной эксперимент D3."""
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
from .experiment import run_experiment, run_pilot, preflight
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
    parser = Parser(description='День 3: диалог, reasoning, пилот и сравнение четырёх методов')
    commands = parser.add_subparsers(dest='command', required=True)
    chat = commands.add_parser('chat', help='Новый разговор или продолжение сохранённого')
    chat.add_argument('--session', help='Имя разговора; файл истории создаётся автоматически')
    chat.add_argument('--max-tokens', type=int, default=None, help='Общий лимит нового разговора (32768)')
    chat.add_argument('--reasoning', choices=['off','min','max'], default=None)
    experiment = commands.add_parser('experiment', help='64 решения / до 80 платных API-вызовов')
    experiment.add_argument('--pilot', required=True, help='Путь к успешному pilot.json')
    experiment.add_argument('--budget-rub', required=True, type=float)
    experiment.add_argument('--resume', help='Папка частичного прогона; готовые попытки не повторяются')
    pilot = commands.add_parser('pilot', help='Ровно два платных вызова для проверки min и max')
    catalog = commands.add_parser('catalog', help='Возможности и цены модели без генерации ответа')
    for command in (chat, experiment, catalog, pilot):
        command.add_argument('--model', help='Переопределить LLM_MODEL')
        command.add_argument('--provider', help='Закреплённый маршрут (google-ai-studio/flex)')
        command.add_argument('--connect-timeout', type=float, default=10)
        command.add_argument('--read-timeout', type=float, default=180)
    args = parser.parse_args(argv)
    if any(not math.isfinite(v) or v <= 0 for v in (args.connect_timeout, args.read_timeout)):
        raise ArgumentError('Таймауты должны быть положительными конечными числами.')
    if args.command == 'chat':
        if args.session and not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', args.session):
            raise ArgumentError('Имя разговора: 1–100 латинских букв, цифр, дефисов или подчёркиваний.')
        if args.max_tokens is not None and not 24576 < args.max_tokens <= 65535:
            raise ArgumentError('Общий лимит должен быть больше бюджета thinking 24576 и не больше 65535.')
    if args.command == 'experiment' and (not math.isfinite(args.budget_rub) or args.budget_rub<=0):
        raise ArgumentError('Бюджет должен быть конечным положительным числом.')
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
        if conversation.data['provider'] != api.provider or (args.reasoning is not None and args.reasoning != conversation.data['mode']):
            raise HistoryError('Маршрут или режим reasoning отличается от сохранённого разговора.')
    else:
        conversation = Conversation.create(path, api.model, {'max_tokens': args.max_tokens or 32768},
                                           provider=api.provider,mode=args.reasoning or 'min')
    emit(f'Разговор: {conversation.path.stem}; модель: {conversation.model}; лимит: {conversation.data["parameters"]["max_tokens"]}')
    emit(f"Маршрут: {conversation.data['provider']}; reasoning: {conversation.data['mode']}")
    emit('Команды: /new — новый разговор; /history — история; /exit — выход.')
    while True:
        api.journal.record('INPUT_PROMPT', 'Вы> ')
        try:
            text = read('Вы> ')
        except (EOFError, StopIteration):
            emit('Диалог завершён (EOF).')
            return
        api.journal.record('INPUT', text)
        text = text.strip()
        if text == '/exit':
            emit('Диалог завершён. История сохранена.')
            return
        if text == '/new':
            conversation = Conversation.create(directory / (identifier() + '.json'), api.model,
                                               conversation.data['parameters'],provider=api.provider,
                                               mode=conversation.data['mode'])
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
                     **dotenv_values(root.parent / 'D2' / '.env', interpolate=False),
                     **dotenv_values(root / '.env', interpolate=False), **os.environ}
                    if env is None else dict(env))
        key = (settings.get('POLZA_API_KEY') or '').strip()
        model = (settings.get('LLM_MODEL') or '').strip()
        journal.secret = key
        try:
            args = arguments(argv)
            model = args.model or model
            provider = args.provider or (settings.get('LLM_PROVIDER') or 'google-ai-studio/flex').strip()
            if not key or not model:
                raise ArgumentError('Задайте POLZA_API_KEY и LLM_MODEL в .env или окружении.')
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
            api = ApiClient(key, model, journal, connection, provider=provider,
                connect_timeout=args.connect_timeout, read_timeout=args.read_timeout)
            try:
                if args.command == 'chat':
                    preflight(api)
                    chat_loop(api, args, root, read, emit)
                elif args.command == 'catalog':
                    emit(json.dumps(preflight(api), ensure_ascii=False, indent=2))
                elif args.command == 'pilot':
                    emit(f'Пилот: два платных запроса, модель {model}, маршрут {provider}.')
                    result = run_pilot(api, root / 'pilots', notify=emit)
                    emit('Результат пилота: ' + str(result))
                    data=json.loads(result.read_text(encoding='utf-8'))
                    emit('Режимы подтверждены: ' + ('да' if data['verified'] else 'нет'))
                    if not data['verified']:
                        for mode, errors in data['checks'].items():
                            for message in errors: emit(mode+': '+message,error=True)
                        journal.record('END',1)
                        return 1
                else:
                    emit(f'Эксперимент: до 80 платных запросов; модель {model}, reasoning min/max, бюджет {args.budget_rub} ₽.')
                    result = run_experiment(api, root / 'results', pilot_path=args.pilot,budget_rub=args.budget_rub,
                                            notify=emit,resume_dir=args.resume)
                    emit('Таблицы и выводы: ' + str(result / 'report.md'))
                    status=json.loads((result/'results.json').read_text())['config']['status']
                    if status!='complete':
                        emit('Эксперимент остановлен: '+status,error=True)
                        journal.record('END',1)
                        return 1
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
