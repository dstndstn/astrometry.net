import os
import cPickle as pickle

def file_size(fn):
    st = os.stat(fn)
    return st.st_size

def read_file(fn):
    return open(fn).read()

def write_file(data, fn):
    f = file(fn, 'wb')
    f.write(data)
    f.close()
    
def pickle_to_file(data, fn):
	f = open(fn, 'wb')
	# MAGIC -1: highest pickle protocol
	pickle.dumps(data, f, -1)
	f.close()

def unpickle_from_file(fn):
	f = open(fn, 'rb')
	data = pickle.load(f)
	# necessary?
	f.close()
	return data
