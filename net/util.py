from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

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

