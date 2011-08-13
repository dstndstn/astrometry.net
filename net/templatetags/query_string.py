from django import template
import urllib

register = template.Library()

@register.simple_tag(takes_context=True)
def query_string(context, key, value):
    GET = context['request'].GET.copy()
    GET[key] = value
    return urllib.urlencode(GET)
    
