import os
#import cPickle as pickle
#import pickle
import cPickle

def _filesize(fn):
	st = os.stat(fn)
	return st.st_size

def file_size(fn, error=None):
	if error is None:
		return _filesize(fn)
	try:
		return _filesize(fn)
	except OSError:
		return error

def read_file(fn):
    return open(fn).read()

def write_file(data, fn):
    f = file(fn, 'wb')
    f.write(data)
    f.close()
    
def pickle_to_file(data, fn):
	f = open(fn, 'wb')
	# MAGIC -1: highest pickle protocol
	cPickle.dump(data, f, -1)
	f.close()

def unpickle_from_file(fn):
	f = open(fn, 'rb')
	data = cPickle.load(f)
	# necessary?
	f.close()
	return data

def get_svn_version():
    from run_command import run_command
    version = {}
    rtn,out,err = run_command('svn info')
    assert(rtn == 0)
    lines = out.split('\n')
    lines = [l for l in lines if len(l)]
    for l in lines:
        words = l.split(':', 1)
        words = [w.strip() for w in words]
        version[words[0]] = words[1]
    return version
