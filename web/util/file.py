import os

def file_size(fn):
    st = os.stat(fn)
    return st.st_size

