"""CLI: каталог трёх моделей, диалог и эксперимент с общим бюджетом."""
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
from .models import MODELS, PROVIDER, preflight
from .history import Conversation, HistoryError
from .experiment import run_experiment

PROJECT_DIR=Path(__file__).resolve().parents[2]


class ArgumentError(Exception):pass


class HelpRequested(Exception):
    def __init__(self,text):self.text=text


class Parser(argparse.ArgumentParser):
    def error(self,message):raise ArgumentError(message)
    def print_help(self,file=None):raise HelpRequested(self.format_help())


def arguments(argv):
    parser=Parser(description='День 5: три модели, качество, время, токены и стоимость')
    commands=parser.add_subparsers(dest='command',required=True)
    chat=commands.add_parser('chat',help='Разговор с выбранной моделью и сохранением истории')
    chat.add_argument('--model',choices=[*MODELS,*MODELS.values()],default=None)
    chat.add_argument('--session')
    experiment=commands.add_parser('experiment',help='Три пилота + 8 задач × 3 модели × 3 повтора')
    experiment.add_argument('--repeats',type=int,default=3)
    experiment.add_argument('--budget-rub',type=float,default=50)
    experiment.add_argument('--resume',help='Папка прогона; учтённые попытки не повторяются')
    catalog=commands.add_parser('catalog',help='Проверить три маршрута и цены без платной генерации')
    for command in (chat,experiment,catalog):
        command.add_argument('--connect-timeout',type=float,default=10)
        command.add_argument('--read-timeout',type=float,default=180)
    args=parser.parse_args(argv)
    if any(not math.isfinite(v) or v<=0 for v in (args.connect_timeout,args.read_timeout)):
        raise ArgumentError('Таймауты должны быть положительными конечными числами.')
    if args.command=='chat' and args.session and not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',args.session):
        raise ArgumentError('Имя разговора: 1–100 латинских букв, цифр, дефисов или подчёркиваний.')
    if args.command=='experiment' and (not math.isfinite(args.budget_rub) or args.budget_rub<=0 or not 1<=args.repeats<=100):
        raise ArgumentError('Бюджет должен быть положительным, число повторов — от 1 до 100.')
    return args


def chat_loop(api,args,root,read,emit):
    path=root/'history'/((args.session or identifier())+'.json')
    if path.exists():
        chat=Conversation.load(path)
        if chat.model!=api.model:raise HistoryError('Выбранная модель отличается от сохранённого разговора.')
    else:chat=Conversation.create(path,api.model,provider=api.provider)
    emit(f'Разговор: {chat.path.stem}; модель: {api.model}; маршрут: {api.provider}')
    emit('Температура 0.7; reasoning 1024; общий лимит 4096. Команды: /history, /new, /exit.')
    while True:
        api.journal.record('INPUT_PROMPT','Вы> ')
        try:text=read('Вы> ')
        except (EOFError,StopIteration):emit('Диалог завершён (EOF).'); return
        api.journal.record('INPUT',text); text=text.strip()
        if text=='/exit':emit('Диалог завершён. История сохранена.'); return
        if text=='/history':emit(json.dumps(chat.messages,ensure_ascii=False,indent=2))
        elif text=='/new':
            chat=Conversation.create(root/'history'/(identifier()+'.json'),api.model,provider=api.provider)
            emit('Новый разговор: '+chat.path.stem)
        elif not text:emit('Пустое сообщение не отправлено.')
        elif text.startswith('/'):emit('Неизвестная команда. Доступны /history, /new, /exit.')
        else:
            try:emit('Модель> '+chat.ask(text,api))
            except (ApiError,HistoryError) as exc:emit('Ошибка: '+str(exc),error=True)


def main(argv=None,*,root=None,env=None,client=None,read=None,stdout=None,stderr=None):
    root=Path(root) if root is not None else PROJECT_DIR
    stdout,stderr=stdout or sys.stdout,stderr or sys.stderr
    read=input if read is None else read
    journal=None
    def failure(text,status):
        try:
            if journal is not None:journal.record('STDERR',text); journal.record('END',status)
        except OSError:pass
        print(text,file=stderr)
        return status
    try:
        journal=Journal(root/'logs')
        settings={}
        if env is None:
            for day in range(1,6):settings.update(dotenv_values(root.parent/f'D{day}'/'.env',interpolate=False))
            settings.update(os.environ)
        else:settings=dict(env)
        key=(settings.get('POLZA_API_KEY') or '').strip(); journal.secret=key
        def emit(text,error=False):
            text=journal.scrub(text); journal.record('STDERR' if error else 'STDOUT',text)
            print(text,file=stderr if error else stdout,flush=True)
        try:
            args=arguments(argv)
            if args.command!='catalog' and not key:raise ArgumentError('Задайте POLZA_API_KEY в .env D1–D5 или окружении.')
            if args.command=='chat':
                model=args.model or (settings.get('LLM_MODEL') or MODELS['small'])
                model=MODELS.get(model,model)
                if model not in MODELS.values():raise ArgumentError('Выберите small, medium или flagship.')
        except HelpRequested as exc:emit(exc.text.rstrip()); journal.record('END',0); return 0
        except ArgumentError as exc:emit('Ошибка настроек: '+str(exc),error=True); journal.record('END',2); return 2
        manager=nullcontext(client) if client is not None else httpx.Client(transport=httpx.HTTPTransport(retries=0),follow_redirects=False)
        with manager as connection:
            clients={tier:ApiClient(key,mid,journal,connection,provider=PROVIDER,
                connect_timeout=args.connect_timeout,read_timeout=args.read_timeout) for tier,mid in MODELS.items()}
            try:
                if args.command=='catalog':emit(json.dumps({tier:preflight(api) for tier,api in clients.items()},ensure_ascii=False,indent=2))
                elif args.command=='chat':
                    api=next(api for api in clients.values() if api.model==model)
                    preflight(api); chat_loop(api,args,root,read,emit)
                else:
                    emit(f'До {3+24*args.repeats} платных вызовов, включая пилот. Общий бюджет: {args.budget_rub} ₽.')
                    path=run_experiment(clients,root/'results',repeats=args.repeats,budget_rub=args.budget_rub,
                        notify=emit,resume_dir=args.resume)
                    status=json.loads((path/'results.json').read_text())['config']['status']
                    emit('Отчёт: '+str(path/'report.md')); emit('Статус: '+status)
                    if status!='complete':journal.record('END',1); return 1
            except (ApiError,HistoryError) as exc:emit('Ошибка: '+str(exc),error=True); journal.record('END',1); return 1
        journal.record('END',0); return 0
    except KeyboardInterrupt:
        return failure('Прервано пользователем; автоматического повтора нет.',130)
    except OSError:
        return failure('Ошибка: журнал, история или результаты недоступны. Работа остановлена.',1)
    except Exception:
        return failure('Ошибка: непредвиденный сбой. Сырые исключения скрыты для защиты ключа.',1)
    finally:
        if journal is not None:journal.close()


def entrypoint():raise SystemExit(main())
