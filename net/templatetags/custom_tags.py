from django import template
from django.utils.http import urlquote
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

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

def paginator(context, page, text="", total_pages=9):
    number = page.number
    num_pages = page.paginator.num_pages
    start = number - (total_pages-1)/2
    end   = number + (total_pages-1)/2
    if start <= 0:
        end += 1-start
    if end > num_pages:
        start -= end-num_pages
    
    page_numbers = [n for n in range(start, end+1) if n > 0 and n <= num_pages]

    if page.has_next():
        nextpage = page.next_page_number()
    else:
        nextpage = None

    if page.has_previous():
        prevpage = page.previous_page_number()
    else:
        prevpage = None

    return {
        "results_per_page": len(page.object_list),
        "total_results": page.paginator.count,
        "page": number,
        "pages": num_pages,
        "page_numbers": page_numbers,
        "next": nextpage,
        "previous": prevpage,
        "has_next": page.has_next(),
        "has_previous": page.has_previous(),
        "show_first": 1 not in page_numbers,
        "show_last": num_pages not in page_numbers,
        "text": text,
        "request": context['request'],
    }

register.inclusion_tag("pagination.html", takes_context=True)(paginator)

