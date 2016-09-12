# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from .fits import text_table_fields

def gator2fits(fn):
    f = open(fn, 'rb')
    while True:
        line = f.readline()
        #print 'Read:', line
        if not line.startswith('\\'):
            break
    names = line
    types = f.readline()
    units = f.readline()
    xxx = f.readline()

    #print 'names:', names
    #print 'types:', types
    #print 'xxx:', xxx

    hdr = names.replace('|', ' ')

    #print 'hdr:', hdr
    tmap = {'double':float, 'int':int, 'char':str}
    types = [tmap.get(t) for t in types.replace('|',' ').split()]
    #print 'Types:', types
        
    data = f.read()
    #print 'Data:', data
    data = data.replace('|', ' ')
    data = data.replace('null', 'NaN')
    
    T = text_table_fields(None, text=data, headerline=hdr, coltypes=types)
    #T.about()
    #print 'Got', T
    return T
