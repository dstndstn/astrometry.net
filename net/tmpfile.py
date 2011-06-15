import os
import tempfile

def get_temp_file():
    f,fn = tempfile.mkstemp()
    os.close(f)
    return fn
