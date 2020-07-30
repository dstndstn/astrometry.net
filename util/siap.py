# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
from xml.dom import minidom, Node
import numpy as np
from astrometry.util.fits import *

def siap_parse_result(fn=None, txt=None):
    if fn is not None:
        dom1 = minidom.parse(fn)
    elif txt is not None:
        dom1 = minidom.parseString(txt)
    else:
        raise RuntimeError('Need filename or text to parse!')

    tables = dom1.getElementsByTagName('TABLE')
    assert(len(tables) == 1)
    table = tables[0]
    
    fields = table.getElementsByTagName('FIELD')
    print('%i fields' % len(fields))
    fieldnames = []
    fieldtypes = []
    fieldparser = []
    fieldisarray = []
    for f in fields:
        name = f.getAttribute('name').lower().replace('[]', '')
        print('field:', name, end=' ')
        ftype = f.getAttribute('datatype').lower()
        print('(%s)' % ftype)
        farray = f.hasAttribute('arraysize')

        ftmap = {'int':int,
                 'double':float,
                 'float':float,
                 'char':str,
                 }

        fieldnames.append(name)
        fieldtypes.append(ftype)
        fieldparser.append(ftmap.get(ftype))
        fieldisarray.append(farray)

    data = table.getElementsByTagName('TABLEDATA')
    assert(len(data) == 1)
    data = data[0]

    rows = data.getElementsByTagName('TR')
    print('%i rows' % len(rows))

    datarows = []
    for r in rows:
        cols = r.getElementsByTagName('TD')
        assert(len(cols) == len(fields))
        datacol = []
        for c,ft,fp,fa in zip(cols, fieldtypes, fieldparser, fieldisarray):
            if len(c.childNodes) == 0:
                # blank
                if ft in ['double']:
                    datacol.append(np.nan)
                elif ft in ['int']:
                    datacol.append(0)
                else:
                    datacol.append('')
                continue
            assert(c.firstChild)
            c = c.firstChild
            # print('Node:', c)
            # print('Node value:', c.nodeValue)
            assert(c.nodeType in [Node.TEXT_NODE, Node.CDATA_SECTION_NODE])
            c = c.nodeValue
            datum = None
            if fa and ft in ['int','double']:
                #elements = c.split(',')
                c = c.replace(',', ' ')
                elements = c.split()
                datum = np.array([fp(x) for x in elements])
            elif fp:
                datum = fp(c)
            else:
                datum = c
            datacol.append(datum)
        datarows.append(datacol)

    t = tabledata()
    for i,f in enumerate(fieldnames):
        t.set(str(f), np.array([r[i] for r in datarows]))
    t._length = len(datarows)

    return t
