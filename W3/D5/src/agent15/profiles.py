"""Typed preferences and explicit dialogue updates."""
import re
from typing import Literal
from pydantic import BaseModel, ConfigDict

LENGTHS = {
    'brief': 'Отвечай кратко, не более 45 слов.',
    'detailed': 'Отвечай подробно: минимум 100 слов пояснения вне блоков кода, добавь детали и пример.',
}
CODE = {
    'with': 'Если задача программная, включай пример кода в отдельном блоке Markdown с тройными обратными кавычками.',
    'without': 'Не включай блоки кода и программные примеры.',
}


class Profile(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    length: Literal['neutral', 'brief', 'detailed'] = 'neutral'
    code: Literal['neutral', 'with', 'without'] = 'neutral'

    def system_message(self):
        rules = [rules[value] for rules, value in ((LENGTHS, self.length), (CODE, self.code)) if value in rules]
        return {'role': 'system', 'content': 'Сохранённые предпочтения пользователя. ' +
                (' '.join(rules) if rules else 'Особых предпочтений нет.')}


CHOICES = {'кратко': ('length','brief'), 'подробно': ('length','detailed'),
           'без кода': ('code','without'), 'с кодом': ('code','with'),
           'с примером кода': ('code','with')}
OPTION = '|'.join(sorted(map(re.escape, CHOICES), key=len, reverse=True))
PREFERENCE = re.compile(
    rf'(?:(?:теперь|всегда|впредь|дальше) )?(?:отвечай|отвечайте) '
    rf'(?P<first>{OPTION})(?:(?: и |, )(?P<second>{OPTION}))?[.!]?',
    re.IGNORECASE,
)


def preference_patch(message):
    normalized = re.sub(r'\s+', ' ', message.strip().lower())
    match = PREFERENCE.fullmatch(normalized)
    if not match:
        return {}
    result = {}
    for option in match.groups():
        if option is None:
            continue
        field, value = CHOICES[option]
        if field in result and result[field] != value:
            return {}
        result[field] = value
    return result
