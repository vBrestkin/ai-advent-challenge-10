import argparse
from datetime import datetime

from pathlib import Path

from uuid import uuid4

import os,json

def identifier():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + uuid4().hex[:12]

class Journal:
    def __init__(self, directory, secret=''):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.path = Path(directory) / (identifier() + '.log')
        self.secret = secret
        fd = os.open(str(self.path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.file = os.fdopen(fd, 'w', encoding='utf-8')
        self.record('START', 'Запуск agent12')

    def scrub(self, value):
        if isinstance(value, str):
            return value.replace(self.secret, '[REDACTED]') if self.secret else value
        if isinstance(value, dict):
            return {self.scrub(k): self.scrub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.scrub(v) for v in value]
        return value

    def record(self, event, data):
        timestamp = datetime.now().astimezone().isoformat(timespec='seconds')
        self.file.write(f'{timestamp} {event} {json.dumps(self.scrub(data), ensure_ascii=False)}\n')
        self.file.flush()

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()



class LoggingParser(argparse.ArgumentParser):
    def __init__(self, *args, journal, **kwargs):
        self.journal = journal
        super().__init__(*args, **kwargs)

    def _print_message(self, message, file=None):
        safe = self.journal.scrub(message or '')
        self.journal.record('CLI_OUTPUT', safe)
        super()._print_message(safe, file)
