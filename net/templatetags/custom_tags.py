from django import template
from django.utils.http import urlquote
import urllib
import types

register = template.Library()

@register.simple_tag(takes_context=True)
def query_string(context, key, value):
    GET = context['request'].GET.copy()
    GET[key] = value
    
    strings = []
    for k, v in GET.items():
        strings += [urlquote(k) + '=' + urlquote(v)]
        
    return '&'.join(strings)
    
def create_list(obj):
    return [obj]

register.filter('list', create_list)

    
