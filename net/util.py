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
    

