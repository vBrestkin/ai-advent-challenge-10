"""Проверенные профили Gemini 2.5; одна конфигурация генерации для всех моделей."""
from copy import deepcopy
import math

from .core import ApiError

MODELS = {'small':'google/gemini-2.5-flash-lite',
          'medium':'google/gemini-2.5-flash', 'flagship':'google/gemini-2.5-pro'}
PROVIDER = 'google-ai-studio/flex'
PARAMETERS = {'max_tokens':4096,'temperature':.7,'reasoning':{'enabled':True,'max_tokens':1024}}


def preflight(api):
    if api.model not in MODELS.values() or api.provider != PROVIDER:
        raise ApiError('Для D5 проверены только три профиля Gemini 2.5 и google-ai-studio/flex.')
    catalog=api.catalog()
    provider=next((p for p in catalog.get('providers',[]) if p.get('name')==api.provider),None)
    if provider is None:
        raise ApiError('Модель не имеет выбранного маршрута.')
    required={'temperature','reasoning','max_tokens','response_format','structured_outputs'}
    if not required <= set(provider.get('supported_parameters',[])):
        raise ApiError('Маршрут не поддерживает общие параметры эксперимента.')
    if (provider.get('capabilities') or {}).get('json_schema_strict',{}).get('status')!='supported':
        raise ApiError('Маршрут не подтверждает строгую JSON Schema.')
    limit=provider.get('max_completion_tokens') or catalog.get('max_completion_tokens',0)
    if not isinstance(limit,int) or limit < PARAMETERS['max_tokens']:
        raise ApiError('Лимит модели меньше общего лимита ответа.')
    pricing=provider.get('pricing') or {}
    try:
        rates={k:float(pricing[k]) for k in ('prompt_per_million','completion_per_million')}
        if pricing.get('currency')!='RUB' or any(not math.isfinite(v) or v<=0 for v in rates.values()):
            raise ValueError
    except (KeyError,TypeError,ValueError):
        raise ApiError('Неизвестна рублёвая цена выбранного маршрута.') from None
    api.price_limits={'prompt':rates['prompt_per_million']*1.01,
                      'completion':rates['completion_per_million']*1.01}
    return {'catalog':catalog,'provider':provider,'rates':rates,'price_limits':deepcopy(api.price_limits)}
