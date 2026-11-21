"""Fixed dialogue and answer key from day 9, frozen before model requests."""
import json,re,csv,io
# Frozen before any model request. Expected answers are never included in summary or quiz inputs.
PROMPTS = [
 'Проект называется «Маяк», руководитель — Ирина. Найди сумму заказа A-104 и вычисли цену после скидки 15%. Это контрольная сумма проекта. Ответь кратко.',
 'Запомни дату демонстрации: 18 декабря 2026 года. Бюджет проекта сейчас 180000 рублей. Подтверди кратко.',
 'Для хранения диалогов выбрана SQLite. Развёртывание разрешено только в России. Подтверди кратко.',
 'Ограничение: нельзя отправлять персональные данные во внешние облака. Решение по интерфейсу: локальный веб-чат. Подтверди кратко.',
 'Перейдём к оформлению: фон светлый, крупный текст, понятные подписи кнопок. Обсуди одним предложением.',
 'Для показа нужны примеры простого запроса и запроса с инструментом. Предложи порядок в одном предложении.',
 'Добавим раздел документации о запуске и остановке сервера. Подтверди одним предложением.',
 'Важное исправление: бюджет увеличен с 180000 до 210000 рублей. Актуальный бюджет — 210000 рублей. Остальные договорённости прежние. Подтверди кратко.',
 'На репетиции проверим восстановление диалога после перезапуска. Ответь одним предложением.',
 'Для записи видео подготовим чистое окно браузера и терминал. Подтверди одним предложением.',
 'Перед показом прогоним автотесты и проверим логи. Подтверди одним предложением.',
 'Подготовка завершена. Дальше я задам вопросы о конкретных прежних договорённостях. Ответь только «Готово».',
]
FACTS = [
 {'id':'project','question':'Как называется проект?','answers':['Маяк']},
 {'id':'manager','question':'Как зовут руководителя проекта?','answers':['Ирина']},
 {'id':'deadline','question':'Какова дата демонстрации? Используй YYYY-MM-DD.','answers':['2026-12-18']},
 {'id':'budget','question':'Каков актуальный бюджет в рублях? Ответь числом без валюты.','answers':['210000']},
 {'id':'storage','question':'Какую базу данных выбрали для хранения диалогов?','answers':['SQLite']},
 {'id':'region','question':'В какой стране разрешено развёртывание?','answers':['Россия','в России','России']},
 {'id':'constraint','question':'Какие данные запрещено отправлять во внешние облака?','answers':['персональные данные']},
 {'id':'interface','question':'Какой интерфейс решили сделать?','answers':['локальный веб-чат','локальный веб чат']},
 {'id':'order_amount','question':'Какая исходная сумма заказа A-104 уже была получена в этой беседе? Только число.','answers':['2390','2390.00']},
 {'id':'discount_amount','question':'Какая контрольная сумма после скидки 15% уже была вычислена? Только число.','answers':['2031.5','2031.50']},
]


def grade(answer, expected):
    try:
        value=json.loads(answer)
        if not isinstance(value,dict) or set(value)!={'value'} or not isinstance(value['value'],(str,int,float)) or isinstance(value['value'],bool): return False
        normalize=lambda s: re.sub(r'\s+',' ',str(s).strip().lower().replace('ё','е'))
        return normalize(value['value']) in {normalize(item) for item in expected}
    except (ValueError,TypeError): return False


def csv_text(rows):
    if not rows:return ''
    stream=io.StringIO(newline=''); writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(rows);return stream.getvalue()
