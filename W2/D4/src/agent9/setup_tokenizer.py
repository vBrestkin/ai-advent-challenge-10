"""Явная загрузка фиксированного JSON-токенизатора без исполнения кода модели."""
import hashlib
from pathlib import Path
from uuid import uuid4
import httpx
from .log import Journal, LoggingParser
from .metrics import TOKENIZER_URL, TOKENIZER_SHA256
from .web import ROOT, load_settings


def install(path, http, journal):
    journal.record('TOKENIZER_REQUEST', TOKENIZER_URL)
    response = http.get(TOKENIZER_URL, timeout=90, follow_redirects=True)
    response.raise_for_status()
    payload = response.content
    digest = hashlib.sha256(payload).hexdigest()
    journal.record('TOKENIZER_RESPONSE', {'bytes': len(payload), 'sha256': digest})
    if digest != TOKENIZER_SHA256:
        raise ValueError('Контрольная сумма JSON-токенизатора отличается. Старый файл сохранён.')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name('.tokenizer-' + uuid4().hex + '.tmp')
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def main():
    key = (load_settings(ROOT).get('POLZA_API_KEY') or '').strip()
    with Journal(ROOT / 'logs', key) as journal, httpx.Client(transport=httpx.HTTPTransport(retries=0)) as http:
        parser = LoggingParser(journal=journal, description='Установить фиксированный JSON-токенизатор Qwen')
        parser.parse_args()
        try:
            path = install(ROOT / 'data/tokenizer.json', http, journal)
            message = 'Токенизатор сохранён: ' + str(path)
            journal.record('STDOUT', message)
            print(message)
        except (httpx.HTTPError, OSError, ValueError) as exc:
            journal.record('ERROR', str(exc))
            print('Не удалось установить токенизатор. Проверьте журнал.')
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
