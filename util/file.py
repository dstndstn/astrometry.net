import os

def file_size(fn):
    st = os.stat(fn)
    return st.st_size

def read_file(fn):
    return open(fn).read()

def write_file(data, fn):
    f = file(fn, 'wb')
    f.write(data)
    f.close()
    
