from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django import forms
from django.utils.safestring import mark_safe

class HorizontalRenderer(forms.RadioSelect.renderer):
    def render(self):
        return mark_safe(u'\n'.join([u'%s' % w for w in self]))

class NoBulletsRenderer(forms.RadioSelect.renderer):
    def render(self):
        return mark_safe(u'<br />\n'.join([u'%s' % w for w in self]))

def store_session_form(session, form_class, data):
    session[form_class.__name__] = data

def get_session_form(session, form_class, **kwargs):
    if session.get(form_class.__name__):
        form = form_class(session[form_class.__name__], **kwargs)
        form.is_valid()
        del session[form_class.__name__]
    else:
        form = form_class(**kwargs)
    return form 

def dict_pack(struct_tuple, data_tuple):
    pack = []
    for data in data_tuple:
        index = 0
        packed_data = {}
        for key in struct_tuple:
            packed_data.update({key:data[index]})
            index += 1
        pack += [packed_data]
    return tuple(pack)

def choicify(choice_dict_list, database_value, human_readable_value):
    choice_list = []
    for d in choice_dict_list:
        choice_list.append((d[database_value],d[human_readable_value]))
    return tuple(choice_list)
    
def get_page(object_list, page_size, page_number):
    paginator = Paginator(object_list, page_size)
    try:
        page = paginator.page(page_number)
    except PageNotAnInteger:
        page = paginator.page(1)
    except EmptyPage:
        page = paginator.page(paginator.num_pages)
    return page

