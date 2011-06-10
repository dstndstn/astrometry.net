def dict_pack(struct_tuple, data_tuple):
    pack = {}
    for data in data_tuple:
        index = 0
        for key in struct_tuple:
            pack.update({key:data[index]})
            index += 1

def choicify(choice_dict_list, database_value, human_readable_value):
    choice_list = []
    for dict in choice_dict_list:
        choice_list.append((dict[database_value],dict[human_readable_value]))
    return tuple(choice_list)
    

