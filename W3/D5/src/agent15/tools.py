"""Два локальных инструмента с проверкой аргументов до исполнения."""
from decimal import Decimal, InvalidOperation
from jsonschema import Draft202012Validator

SCHEMAS = {
    'find_order': {'type':'object','properties':{'order_id':{'type':'string','maxLength':50}},'required':['order_id'],'additionalProperties':False},
    'calculate': {'type':'object','properties':{
        'operation':{'type':'string','enum':['add','subtract','multiply','divide']},
        'a':{'type':'string','maxLength':40},'b':{'type':'string','maxLength':40}},
        'required':['operation','a','b'],'additionalProperties':False}}
TOOLS = [{'type':'function','function':{'name':name,'description':description,'parameters':SCHEMAS[name]}}
         for name,description in [('find_order','Найти сумму учебного заказа по ID. Валюта RUB.'),
                                  ('calculate','Точная арифметика. Передавай числа строками. Используй для всех вычислений.')]]


def find_order(order_id):
    orders={'A-104':{'amount':'2390','currency':'RUB'},'B-205':{'amount':'5800','currency':'RUB'}}
    if order_id not in orders: raise ValueError('Заказ не найден.')
    return dict(orders[order_id])


def calculate(operation,a,b):
    left,right=Decimal(a),Decimal(b)
    if not all(n.is_finite() and abs(n)<=Decimal('1e12') and -12<=n.as_tuple().exponent<=12 for n in (left,right)):
        raise ValueError('Числа должны быть конечными, модуль не больше 1e12.')
    operations={'add':lambda:left+right,'subtract':lambda:left-right,
                'multiply':lambda:left*right,'divide':lambda:left/right}
    return {'result':format(operations[operation](),'f')}


def execute(name,args):
    if name not in SCHEMAS: return {'error':'Неизвестный инструмент.'}
    if not Draft202012Validator(SCHEMAS[name]).is_valid(args): return {'error':'Аргументы не соответствуют схеме.'}
    try: return {'find_order':find_order,'calculate':calculate}[name](**args)
    except (ValueError,InvalidOperation,ZeroDivisionError): return {'error':'Некорректные данные, деление на ноль или заказ не найден.'}
