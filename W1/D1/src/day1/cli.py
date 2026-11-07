"""Один запрос без стриминга и автоматических повторов."""

import argparse
from contextlib import nullcontext
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

import httpx
from dotenv import dotenv_values

PROJECT_DIR = Path(__file__).resolve().parents[2]
ENDPOINT = "https://polza.ai/api/v1/chat/completions"


class CliError(Exception):
    """Сообщение, которое безопасно показать пользователю."""


class HelpRequested(Exception):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise CliError(message)

    def exit(self, status=0, message=None):
        raise HelpRequested()

    def _print_message(self, message, file=None):
        # Помощь выводится централизованно: одновременно в консоль и журнал.
        pass


def parse_args(argv):
    parser = Parser(description="Один запрос к Polza.ai с журналом запуска")
    parser.add_argument("prompt", help="Текст запроса в кавычках")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--connect-timeout", type=float, default=10)
    parser.add_argument("--read-timeout", type=float, default=60)
    try:
        args = parser.parse_args(argv)
    except HelpRequested:
        return None, parser.format_help()
    if not args.prompt.strip():
        raise CliError("Запрос не должен быть пустым.")
    if args.max_tokens <= 0:
        raise CliError("max-tokens должен быть положительным.")
    if any(not math.isfinite(v) or v <= 0 for v in (args.connect_timeout, args.read_timeout)):
        raise CliError("Таймауты должны быть конечными положительными числами.")
    return args, None


def response_text(response):
    if response.status_code != 200:
        messages = {400: "Неверный запрос или неподдерживаемые параметры.",
                    401: "Неверный API-ключ.", 402: "Недостаточно средств или исчерпан бюджет ключа.",
                    403: "Доступ запрещён.", 404: "Модель или ресурс не найдены.",
                    408: "Таймаут на стороне провайдера.", 413: "Слишком большой запрос.",
                    429: "Превышен лимит запросов. Повторите позже."}
        detail = messages.get(response.status_code, "Провайдер недоступен." if response.status_code >= 500
                              else "Неожиданный HTTP-статус.")
        raise CliError(f"HTTP {response.status_code}: {detail}")
    try:
        body = response.json()
        choice = body["choices"][0]
        message = choice["message"]
        reason = choice["finish_reason"]
        if message.get("refusal") or reason == "content_filter":
            raise CliError("Модель отказалась отвечать.")
        if reason == "length":
            raise CliError("Ответ обрезан по лимиту токенов; запрос не завершён полностью.")
        if reason != "stop":
            raise CliError("Провайдер не завершил генерацию обычного текстового ответа.")
        content = message["content"]
        if not isinstance(content, str) or not content.strip():
            raise CliError("Провайдер вернул пустой или нетекстовый ответ.")
        return content, body.get("usage")
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        raise CliError("Некорректный JSON или неожиданная структура ответа.") from None


def main(argv=None, *, env=None, log_dir=None, client=None, stdout=None, stderr=None):
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    journal = None
    secret = ""

    def redact(text):
        text = str(text)
        return text.replace(secret, "[REDACTED]") if secret else text

    def record(event, text):
        # JSON внутри текстового .log экранирует переводы строк и управляющие символы.
        stamp = datetime.now().astimezone().isoformat(timespec="seconds")
        journal.write(f"{stamp} {event} {json.dumps(redact(text), ensure_ascii=False)}\n")
        journal.flush()

    def emit(text, *, error=False):
        record("STDERR" if error else "STDOUT", text)
        print(redact(text), file=stderr if error else stdout, flush=True)

    try:
        directory = Path(log_dir) if log_dir is not None else PROJECT_DIR / "logs"
        directory.mkdir(parents=True, exist_ok=True)
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + uuid4().hex + ".log"
        journal = (directory / name).open("x", encoding="utf-8")
        record("START", "Запуск CLI")
        settings = ({**dotenv_values(PROJECT_DIR / ".env", interpolate=False), **os.environ}
                    if env is None else dict(env))
        secret = (settings.get("POLZA_API_KEY") or "").strip()
        try:
            args, help_text = parse_args(argv)
            if help_text is not None:
                emit(help_text.rstrip())
                record("END", "Код завершения: 0")
                return 0
            model = (settings.get("LLM_MODEL") or "").strip()
            if not secret:
                raise CliError("Не задан POLZA_API_KEY в .env или окружении.")
            if not model:
                raise CliError("Не задан LLM_MODEL в .env или окружении.")
        except CliError as exc:
            emit(f"Ошибка настроек: {exc}", error=True)
            record("END", "Код завершения: 2")
            return 2

        record("REQUEST", json.dumps({"model": model, "prompt": args.prompt,
                                      "max_tokens": args.max_tokens,
                                      "connect_timeout": args.connect_timeout,
                                      "read_timeout": args.read_timeout}, ensure_ascii=False))
        started = time.perf_counter()
        try:
            # HTTPTransport(retries=0) явно исключает скрытые повторы подключения.
            manager = nullcontext(client) if client is not None else httpx.Client(
                transport=httpx.HTTPTransport(retries=0), follow_redirects=False)
            with manager as connection:
                response = connection.post(ENDPOINT,
                    headers={"Authorization": f"Bearer {secret}"},
                    json={"model": model, "messages": [{"role": "user", "content": args.prompt}],
                          "max_tokens": args.max_tokens, "stream": False,
                          "provider": {"allow_fallbacks": False, "require_parameters": True}},
                    timeout=httpx.Timeout(args.read_timeout, connect=args.connect_timeout))
            record("HTTP", str(response.status_code))
            record("RESPONSE", response.text)
            answer, usage = response_text(response)
            emit(answer)
            if usage is not None:
                record("USAGE", json.dumps(usage, ensure_ascii=False))
            code = 0
        except httpx.TimeoutException:
            emit("Ошибка: истекло время ожидания подключения или обмена данными. Запрос не повторялся.", error=True)
            code = 1
        except httpx.RequestError:
            emit("Ошибка сети: проверьте соединение, DNS и TLS; возможен разрыв связи или недоступность провайдера. Запрос не повторялся.", error=True)
            code = 1
        except CliError as exc:
            emit(f"Ошибка: {exc}", error=True)
            code = 1
        except KeyboardInterrupt:
            emit("Прервано пользователем. Запрос не повторялся.", error=True)
            code = 130
        except OSError:
            raise
        except Exception:
            # Не выводим сырое исключение: оно может содержать токен или заголовки.
            emit("Ошибка: непредвиденный сбой обработки запроса.", error=True)
            code = 1
        record("DURATION", f"{time.perf_counter() - started:.3f} секунд")
        record("END", f"Код завершения: {code}")
        return code
    except OSError:
        print("Ошибка: не удалось записать журнал или вывести результат. Работа остановлена.", file=stderr)
        return 1
    finally:
        if journal is not None:
            journal.close()


def entrypoint():
    raise SystemExit(main())
