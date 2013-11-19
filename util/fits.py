import os
import pyfits
import numpy as np

def pyfits_writeto(p, filename, **kwargs):
    '''
    *p*: HDUList object
    *filename*: uh, the filename to write to
    '''
    # old pyfits versions (eg the one in Ubuntu 10.04)
    # fail when used with python2.7 (warning.showwarning changed)
    # so work-around pyfits printing a warning when it overwrites an
    # existing file.
    if os.path.exists(filename):
        os.remove(filename)
    p.writeto(filename, **kwargs)

def merge_tables(TT, columns=None):
    assert(len(TT) > 0)
    if columns in [None, 'minimal']:
        cols = set(TT[0].get_columns())
        for T in TT[1:]:
            if columns == 'minimal' and len(cols.symmetric_difference(T.get_columns())):
                cols = cols.intersection(T.get_columns())
                continue
    
            # They must have the same set of columns
            if len(cols.symmetric_difference(T.get_columns())):
                print 'Tables to merge must have the same set of columns.'
                print 'First table columns:', cols
                print 'Target table columns:', T.get_columns()
                print 'Difference:', cols.symmetric_difference(T.get_columns())
            assert(len(cols.symmetric_difference(T.get_columns())) == 0)
        cols = list(cols)

        # Reorder the columns to match their order in TT[0].
        ocols = []
        for c in TT[0].get_columns():
            if c in cols and not c in ocols:
                ocols.append(c)
        cols = ocols

    else:
        for i,T in enumerate(TT):
            # ensure they all have the requested columns
            if not set(columns).issubset(set(T.get_columns())):
                print 'Each table to be merged must have the requested columns'
                print 'Table', i, 'is missing columns:', set(columns)-set(T.get_columns())
                print 'columns', columns
                print 'T.columns', T.get_columns()
                assert(False)
        cols = columns
    N = sum([len(T) for T in TT])
    td = tabledata()
    for col in cols:
        if col.startswith('_'):
            continue
        v0 = TT[0].getcolumn(col)
        if isinstance(v0, np.ndarray):
            V = np.concatenate([T.getcolumn(col) for T in TT])
        elif type(v0) is list:
            V = v0
            for T in TT[1:]:
                V.extend(T.getcolumn(col))
        elif np.isscalar(v0):
            #print 'merge_tables: copying scalar from first table:', col, '=', v0
            V = v0
        else:
            raise RuntimeError("pyfits_utils.merge_tables: Don't know how to concatenate type: %s" % str(type(v0)))
            
        td.set(col, V)
    #td._columns = cols
    assert(td._length == N)
    return td
    

def add_nonstructural_headers(fromhdr, tohdr):
    for card in fromhdr.ascardlist():
        if ((card.key in ['SIMPLE','XTENSION', 'BITPIX', 'END', 'PCOUNT', 'GCOUNT',
                          'TFIELDS',]) or
            card.key.startswith('NAXIS') or
            card.key.startswith('TTYPE') or
            card.key.startswith('TFORM')):
            #card.key.startswith('TUNIT') or
            #card.key.startswith('TDISP')):
            #print 'skipping card', card.key
            continue
        #if tohdr.has_key(card.key):
        #   #print 'skipping existing card', card.key
        #   continue
        #print 'adding card', card.key
        #tohdr.update(card.key, card.value, card.comment, before='END')
        #tohdr.ascardlist().append(
        cl = tohdr.ascardlist()
        if 'END' in cl.keys():
            i = cl.index_of('END')
        else:
            i = len(cl)
        cl.insert(i, pyfits.Card(card.key, card.value, card.comment))

def cut_array(val, I, name=None, to=None):
    if type(I) is slice:
        if to is None:
            return val[I]
        else:
            val[I] = to
            return

    if type(val) in [np.ndarray, np.core.defchararray.chararray]:
        #print 'slicing numpy array "%s": val shape' % name, val.shape
        #print 'slice shape:', I.shape
        # You can't slice a two-dimensional, length-zero, numpy array,
        # with an empty array.
        if len(val) == 0:
            return val
        if to is None:
            return val[I]
        else:
            val[I] = to
            return

    inttypes = [int, np.int64, np.int32, np.int]

    if type(val) in [list,tuple] and type(I) in inttypes:
        if to is None:
            return val[I]
        else:
            val[I] = to
            return

    # HACK -- emulate numpy's boolean and int array slicing
    # (when "val" is a normal python list)
    if type(I) is np.ndarray and hasattr(I, 'dtype') and ((I.dtype.type in [bool, np.bool])
                                                             or (I.dtype == bool)):
        try:
            if to is None:
                return [val[i] for i,b in enumerate(I) if b]
            else:
                for i,(b,t) in enumerate(zip(I,to)):
                    if b:
                        val[i] = t
                return
        except:
            print 'Failed to slice field', name
            #setattr(rtn, name, val)
            #continue

    if type(I) is np.ndarray and all(I.astype(int) == I):
        if to is None:
            return [val[i] for i in I]
        else:
            #[val[i] = t for i,t in zip(I,to)]
            for i,t in zip(I,to):
                val[i] = t
                
    if (np.isscalar(I) and hasattr(I, 'dtype') and
        I.dtype in inttypes):
        if to is None:
            return val[int(I)]
        else:
            val[int(I)] = to
            return

    if hasattr(I, '__len__') and len(I) == 0:
        return []

    print 'Error slicing array:'
    print 'array is'
    print '  type:', type(val)
    print '  ', val
    print 'cut is'
    print '  type:', type(I)
    print '  ', I
    raise Exception('Error in cut_array')

class tabledata(object):

    class td_iter(object):
        def __init__(self, td):
            self.td = td
            self.i = 0
        def __iter__(self):
            return self
        def next(self):
            if self.i >= len(self.td):
                raise StopIteration
            X = self.td[self.i]
            self.i += 1
            return X


    def __init__(self, header=None):
        self._length = 0
        self._header = header
        self._columns = []
    def __str__(self):
        return 'tabledata object with %i rows and %i columns' % (len(self), len([k for k in self.__dict__.keys() if not k.startswith('_')]))
    def about(self):
        keys = [k for k in self.__dict__.keys() if not k.startswith('_')]
        print 'tabledata object with %i rows and %i columns:' % (len(self),  len(keys))
        keys.sort()
        for k in keys:
            print '  ', k,
            v = self.get(k)
            print '(%s)' % (str(type(v))),
            if np.isscalar(v):
                print v,
            elif hasattr(v, 'shape'):
                print 'shape', v.shape,
            elif hasattr(v, '__len__'):
                print 'length', len(v),
            else:
                print v,

            if hasattr(v, 'dtype'):
                print 'dtype', v.dtype,
            print

    def __setattr__(self, name, val):
        object.__setattr__(self, name, val)
        #print 'set', name, 'to', val
        if (self._length == 0) and (not (name.startswith('_'))) and hasattr(val, '__len__') and len(val) != 0 and type(val) != str:
            self._length = len(val)
        if hasattr(self, '_columns') and not name in self._columns:
            self._columns.append(name)
    def set(self, name, val):
        self.__setattr__(name, val)
    def getcolumn(self, name):
        return self.__dict__[name]
        #except:
        #   return self.__dict__[name.lower()]
    def get(self, name):
        return self.getcolumn(name)
    # Returns the list of column names, as they were ordered in the input FITS or text table.
    def get_columns(self, internal=False):
        if internal:
            return self._columns[:]
        return [x for x in self._columns if not x.startswith('_')]
    # Returns the original FITS header.
    def get_header(self):
        return self._header

    def columns(self):
        return [k for k in self.__dict__.keys() if not k.startswith('_')]
    def __len__(self):
        return self._length
    def delete_column(self, c):
        del self.__dict__[c]
        self._columns.remove(c)

    def rename(self, c_old, c_new):
        setattr(self, c_new, getattr(self, c_old))
        self.delete_column(c_old)
        
    def __setitem__(self, I, O):

        #### TEST

        for name,val in self.__dict__.items():
            if name.startswith('_'):
                continue
            cut_array(val, I, name, to=O.get(name))
        return
        ####

        
        if type(I) is slice:
            print 'I:', I
            # HACK... "[:]" -> slice(None, None, None)
            if I.start is None and I.stop is None and I.step is None:
                I = np.arange(len(self))
            else:
                I = np.arange(I.start, I.stop, I.step)
        for name,val in self.__dict__.items():
            if name.startswith('_'):
                continue
            # ?
            if np.isscalar(val):
                self.set(name, O.get(name))
                continue
            try:
                val[I] = O.get(name)
            except Exception:
                # HACK -- emulate numpy's boolean and int array slicing...
                ok = False
                #if type(I) == np.ndarray and hasattr(I, 'dtype') and I.dtype == bool:
                #   for i,b in enumerate(I):
                #       if b:
                #           val[i] = O.get(val)
                #   ok = True
                #if type(I) == np.ndarray and hasattr(I, 'dtype') and I.dtype == 'int':
                #   rtn.set(name, [val[i] for i in I])
                #   ok = True
                #if len(I) == 0:
                #   rtn.set(name, [])
                #   ok = True
                if not ok:
                    print 'Error in slicing an astrometry.util.pyfits_utils.table_data object:'
                    #print '  -->', e

                    import pdb; pdb.set_trace()

                    print 'While setting member:', name
                    print ' setting elements:', I
                    print ' from obj', O
                    print ' target type:', type(O.get(name))
                    print ' dest type:', type(val)
                    print 'index type:', type(I)
                    #if hasattr(val, 'shape'):
                    #   print ' shape:', val.shape
                    #if hasattr(I, 'shape'):
                    #   print ' index shape:', I.shape
                    if hasattr(I, 'dtype'):
                        print '  index dtype:', I.dtype
                    print 'my length:', self._length
                    raise Exception('error in fits_table indexing')

    def copy(self):
        rtn = tabledata()
        for name,val in self.__dict__.items():
            if name.startswith('_'):
                continue
            if np.isscalar(val):
                #print 'copying scalar', name
                rtn.set(name, val)
                continue
            if type(val) in [np.ndarray, np.core.defchararray.chararray]:
                #print 'copying numpy array', name
                rtn.set(name, val.copy())
                continue
            if type(val) in [list,tuple]:
                #print 'copying list', name
                rtn.set(name, val[:])
                continue
            print 'in pyfits_utils: copy(): can\'t copy', name, '=', val[:10], 'type', type(val)
        rtn._header = self._header
        if hasattr(self, '_columns'):
            rtn._columns = [c for c in self._columns]
        return rtn

    def cut(self, I):
        for name,val in self.__dict__.items():
            if name.startswith('_'):
                continue
            if np.isscalar(val):
                continue
            #print 'cutting', name
            C = cut_array(val, I, name)
            self.set(name, C)
            self._length = len(C)

    def __getitem__(self, I):
        rtn = tabledata()
        for name,val in self.__dict__.items():
            if name.startswith('_'):
                continue
            if np.isscalar(val):
                rtn.set(name, val)
                continue
            try:
                C = cut_array(val, I, name)
            except:
                print 'Error in cut_array() via __getitem__, name', name
                raise
            rtn.set(name, C)

            if np.isscalar(I):
                rtn._length = 1
            else:
                rtn._length = len(getattr(rtn, name))
        rtn._header = self._header
        if hasattr(self, '_columns'):
            rtn._columns = [c for c in self._columns]
        return rtn
    def __iter__(self):
        return tabledata.td_iter(self)

    def append(self, X):
        for name,val in self.__dict__.items():
            if name.startswith('_'):
                continue
            if np.isscalar(val):
                continue
            try:
                val2 = X.getcolumn(name)
                if type(val) is list:
                    newX = val + val2
                else:
                    newX = np.append(val, val2, axis=0)
                self.set(name, newX)
                self._length = len(newX)
            except Exception:
                print 'exception appending element "%s"' % name
                raise
                
    def write_to(self, fn, columns=None, header='default', primheader=None,
                 use_fitsio=True, append=False):

        fitsio = None
        if use_fitsio:
            try:
                import fitsio
            except:
                pass

        if columns is None:
            columns = self.get_columns()

        if fitsio:
            arrays = [self.get(c) for c in columns]
            # fitsio has *strange* behavior when file already exists.
            if os.path.exists(fn):
                if not append:
                    os.unlink(fn)
            fits = fitsio.FITS(fn, 'rw')

            #for a,c in zip(arrays, columns):
            #   print 'Writing:', c, 'shape', getattr(a, 'shape', None), 'type', (getattr(a, 'dtype', type(a)))

            if header == 'default':
                header = None
            try:
                fits.write(arrays, names=columns, header=header)
            except:
                print 'Failed to write FITS table'
                print 'Columns:'
                for c,a in zip(columns, arrays):
                    print '  ', c, type(a),
                    try:
                        print a.dtype, a.shape,
                    except:
                        pass
                    print
                raise
            return


        fc = self.to_fits_columns(columns)
        #print 'FITS columns:', fc
        T = pyfits.new_table(fc)
        if header == 'default':
            header = self._header
        if header is not None:
            add_nonstructural_headers(header, T.header)
        if primheader is not None:
            P = pyfits.PrimaryHDU()
            add_nonstructural_headers(primheader, P.header)
            pyfits.HDUList([P, T]).writeto(fn, clobber=True)
        else:
            pyfits_writeto(T, fn)

    writeto = write_to

    def normalize(self, columns=None):
        if columns is None:
            columns = self.get_columns()
        for c in columns:
            X = self.get(c)
            X = normalize_column(X)
            self.set(c, X)

    def to_fits_columns(self, columns=None):
        cols = []

        fmap = {np.float64:'D',
                np.float32:'E',
                np.int32:'J',
                np.int64:'K',
                np.uint8:'B', #
                np.int16:'I',
                #np.bool:'X',
                #np.bool_:'X',
                np.bool:'L',
                np.bool_:'L',
                np.string_:'A',
                }

        if columns is None:
            columns = self.get_columns()
                
        for name in columns:
            if not name in self.__dict__:
                continue
            val = self.get(name)

            #print 'col', name, 'type', val.dtype, 'descr', val.dtype.descr
            #print repr(val.dtype)
            #print val.dtype.type
            #print repr(val.dtype.type)
            #print val.shape
            #print val.size
            #print val.itemsize

            if type(val) in [list, tuple]:
                val = np.array(val)

            try:
                val = normalize_column(val)
            except:
                pass

            try:
                fitstype = fmap.get(val.dtype.type, 'D')
            except:
                print 'Table column "%s" has no "dtype"; skipping' % name
                continue

            if fitstype == 'X':
                # pack bits...
                pass
            if len(val.shape) > 1:
                fitstype = '%i%s' % (val.shape[1], fitstype)
            elif fitstype == 'A' and val.itemsize > 1:
                # strings
                fitstype = '%i%s' % (val.itemsize, fitstype)
            else:
                fitstype = '1'+fitstype
            #print 'fits type', fitstype
            try:
                col = pyfits.Column(name=name, array=val, format=fitstype)
            except:
                print 'Error converting column', name, 'to a pyfits column:'
                print 'fitstype:', fitstype
                try:
                    print 'numpy dtype:'
                    print val.dtype
                    print val.dtype.type
                except:
                    pass
                print 'value:', val
                raise
            cols.append(col)
            #print 'fits type', fitstype, 'column', col
            #print repr(col)
            #print 'col', name, ': data length:', val.shape
        return cols

    def add_columns_from(self, X):
        assert(len(self) == len(X))
        mycols = self.get_columns()
        for c in X.get_columns():
            if c in mycols:
                print 'Not copying existing column', c
                continue
            self.set(c, X.get(c))

def normalize_column(X):
    try:
        dt = X.dtype
    except:
        return X
    if dt.byteorder in ['>','<']:
        # go native
        X = X.astype(dt.newbyteorder('N'))
    return X

def fits_table(dataorfn=None, rows=None, hdunum=1, hdu=None, ext=None,
               header='default',
               columns=None,
               column_map=None,
               lower=True,
               mmap=True,
               normalize=True,
               use_fitsio=True):
    '''
    If 'columns' (a list of strings) is passed, only those columns
    will be read; otherwise all columns will be read.
    '''
    if dataorfn is None:
        return tabledata(header=header)
    
    fitsio = None
    if use_fitsio:
        try:
            import fitsio
        except:
            pass

    pf = None
    hdr = None
    # aliases
    if hdu is not None:
        hdunum = hdu
    if ext is not None:
        hdunum = ext
    if isinstance(dataorfn, str):

        if fitsio:
            F = fitsio.FITS(dataorfn)
            data = F[hdunum]
            hdr = data.read_header()
        else:
            pf = pyfits.open(dataorfn, memmap=mmap)
            data = pf[hdunum].data
            if header == 'default':
                hdr = pf[hdunum].header
            del pf
            pf = None
    else:
        data = dataorfn

    if data is None:
        return None
    T = tabledata(header=hdr)

    T._columns = []

    if fitsio and not (type(data) == pyfits.core.FITS_rec):
        # fitsio sorts the rows and de-duplicates them, so compute
        # permutation vector 'I' to undo that.
        I = None
        if rows is not None:
            rows,I = np.unique(rows, return_inverse=True)

        if type(data) == np.ndarray:
            dd = data
            if columns is None:
                columns = data.dtype.fields.keys()
        else:
            dd = data.read(rows=rows, columns=columns, lower=True)
            if dd is None:
                return None

        if columns is None:
            try:
                columns = data.get_colnames()
            except:
                columns = data.colnames

            if lower:
                columns = [c.lower() for c in columns]

        for c in columns:
            X = dd[c.lower()]
            if I is not None:
                # apply permutation
                X = X[I]
            if column_map is not None:
                c = column_map.get(c, c)
            if lower:
                c = c.lower()
            T.set(c, X)
        
    else:
        if columns is None:
            columns = data.dtype.names
        for c in columns:
            #print 'reading column "%s"' % c
            col = data.field(c)
            if rows is not None:
                col = col[rows]
            if normalize:
                col = normalize_column(col)
            if column_map is not None:
                c = column_map.get(c, c)
            if lower:
                c = c.lower()
            T.set(c, col)

    return T

table_fields = fits_table

### FIXME -- it would be great to have a streaming text2fits as well!
### (fitsio does this fairly easily)
def streaming_text_table(forfn, skiplines=0, split=None, maxcols=None,
                         headerline=None, coltypes=None,
                         intvalmap={'NaN':-1000000, '':-1000000},
                         floatvalmap={'': np.nan},
                         skipcomments=True):
    # unimplemented
    assert(maxcols is None)

    f = None
    if isinstance(forfn, str):
        f = open(forfn)
        print 'Reading file', forfn
    else:
        f = forfn

    for i in range(skiplines):
        x = f.readline()
        print 'Skipping line:', x

    if headerline is None:
        headerline = f.readline().strip()
        print 'Header:', headerline
    header = headerline

    if header[0] == '#':
        header = header[1:]

    if split is None:
        colnames = header.split()
    else:
        colnames = header.split(split)
    print 'Column names:', colnames

    if coltypes is not None:
        if len(coltypes) != len(colnames):
            print 'Column names:', len(colnames)
            print 'Column types:', len(coltypes)
            raise Exception('Column names vs types length mismatch: %i vs %i' %
                            (len(colnames), len(coltypes)))
    else:
        coltypes = [str] * len(colnames)

    Nchunk = 100000
    alldata = []
    ncomplain = 0
    i0 = 0
    while True:
        import time
        t0 = time.clock()

        # Create empty data arrays
        data = [[None] * Nchunk for t in coltypes]
        j = 0
        lines = []
        for i,line in zip(xrange(Nchunk), f):
            line = line.strip()
            if line.startswith('#') and skipcomments:
                print 'Skipping comment line:'
                print line
                print
                continue
            if split is None:
                words = line.split()
            else:
                words = line.split(split)
            if len(words) != len(colnames):
                ncomplain += 1
                if ncomplain > 10:
                    continue
                print ('Expected to find %i columns of data to match headers (%s) in row %i; got %i\n    "%s"\n(Skipping this row of the input file)' %
                       (len(colnames), ', '.join(colnames), i+i0, len(words), line))
                continue
            for d,w in zip(data, words):
                d[j] = w
            j += 1
        nread = i+1
        goodrows = j
        
        t1 = time.clock()

        floattypes = [float,np.float32,np.float64]
        inttypes = [int, np.int32, np.int64]

        for dat,typ in zip(data, coltypes):
            if typ in floattypes:
                valmap = floatvalmap
            elif typ in inttypes:
                valmap = intvalmap
            else:
                continue
            # HACK -- replace with stringified versions of bad-values
            valmap = dict([(k,str(v)) for k,v in valmap.items()])
            for i,d in enumerate(dat):
                #dat[i] = valmap.get(d,d)
                # try:
                #     dat[i] = valmap[d]
                # except KeyError:
                #     pass
                if d in valmap:
                    dat[i] = valmap[d]
        t2 = time.clock()
                    
        # trim to valid rows
        data = [dat[:goodrows] for dat in data]
        # convert
        data = [np.array(dat).astype(typ) for dat,typ in zip(data, coltypes)]
                    
        t3 = time.clock()

        #print 'Reading & splitting:', t1-t0
        #print 'Bad values:', t2-t1
        #print 'Conversion:', t3-t2
        #print 'Total:', t3-t0

        # print 'Read', i+1, 'lines'
        # print 'Read', j, 'valid lines'
        print 'Read line', i0 + nread
        
        alldata.append(data)
        i0 += nread

        if nread != Nchunk:
            break
        
    if ncomplain > 10:
        print 'Total of', ncomplain, 'bad lines'

    # merge chunks
    T = tabledata()
    for name in reversed(colnames):
        print 'Merging', name
        xx = [data.pop() for data in alldata]
        print 'lengths:', [len(x) for x in xx]
        xx = np.hstack(xx)
        print 'total:', len(xx)
        print 'type:', xx.dtype
        T.set(name, xx)

    return T
        
# ultra-brittle text table parsing.
def text_table_fields(forfn, text=None, skiplines=0, split=None, trycsv=True, maxcols=None, headerline=None, coltypes=None,
                      intvalmap={'NaN':-1000000, '':-1000000},
                      floatvalmap={}):
    if text is None:
        f = None
        if isinstance(forfn, str):
            f = open(forfn)
            print 'Reading file', forfn
            data = f.read()
            f.close()
        else:
            data = forfn.read()
            print 'Read', len(data), 'bytes'
    else:
        data = text

    # replace newline variations with a single newline character
    print 'Replacing line endings'
    data = data.replace('\r\n','\n') # windows
    data = data.replace('\r','\n') # mac os
    print 'Splitting lines'
    txtrows = data.split('\n')
    print 'Got', len(txtrows), 'lines'
    print 'First line:', txtrows[0]
    print 'Last line:', txtrows[-1]
    if txtrows[-1] == '':
        print 'Trimming last line.'
        txtrows = txtrows[:-1]
        print 'Last line now:', txtrows[-1]
    if skiplines != 0:
        txtrows = txtrows[skiplines:]
        print 'Skipped', skiplines, 'kept', len(txtrows)

    if headerline is None:
        # column names are in the first (un-skipped) line.
        header = txtrows.pop(0)
        if header[0] == '#':
            header = header[1:]
    else:
        header = headerline
    header = header.split()
    if len(header) == 0:
        raise Exception('Expected to find column names in the first row of text; got \"%s\".' % txt)
    print 'Header:', len(header), 'columns'
    #assert(len(header) >= 1)
    if trycsv and (split is None) and (len(header) == 1) and (',' in header[0]):
        # try CSV
        header = header[0].split(',')
    colnames = header

    if coltypes is not None:
        if len(coltypes) != len(colnames):
            raise Exception('Column types: length %i, vs column names, length %i' %
                            (len(coltypes), len(colnames)))

    fields = tabledata()
    txtrows = [r for r in txtrows if not r.startswith('#')]
    print 'Kept', len(txtrows), 'non-commented rows'
    coldata = [[] for x in colnames]
    ncomplain = 0
    for i,r in enumerate(txtrows):
        if i and (i % 1000000 == 0):
            print 'Row', i
        if maxcols is not None:
            r = r[:maxcols]
        if split is None:
            cols = r.split()
        else:
            cols = r.split(split)
        if len(cols) == 0:
            continue
        if trycsv and (split is None) and (len(cols) != len(colnames)) and (',' in r):
            # try to parse as CSV.
            cols = r.split(',')
            
        if len(cols) != len(colnames):
            #raise Exception('Expected to find %i columns of data to match headers (%s) in row %i; got %i\n "%s"' % (len(colnames), ', '.join(colnames), i, len(cols), r))
            ncomplain += 1
            if ncomplain > 10:
                continue
            print 'Expected to find %i columns of data to match headers (%s) in row %i; got %i\n    "%s"' % (len(colnames), ', '.join(colnames), i, len(cols), r)
            continue
        #assert(len(cols) == len(colnames))
        if coltypes is not None:
            floattypes = [float,np.float32,np.float64]
            for i,(cd,c,t) in enumerate(zip(coldata, cols, coltypes)):
                if t in floattypes:
                    if len(c) == 0:
                        cd.append(np.nan)
                        continue
                    c = floatvalmap.get(c, c)

                if t in [int, np.int32, np.int64]:
                    try:
                        cd.append(t(c))
                    except:
                        if c in intvalmap:
                            cd.append(intvalmap[c])
                        else:
                            raise
                else:
                    cd.append(t(c))
        else:
            for cd,c in zip(coldata, cols):
                cd.append(c)

    if ncomplain > 10:
        print 'Total of', ncomplain, 'bad lines'
                
    if coltypes is None:
        for i,col in enumerate(coldata):
            isint = True
            isfloat = True
            for x in col:
                try:
                    float(x)
                except:
                    isfloat = False
                    #isint = False
                    #break
                try:
                    int(x, 0)
                except:
                    isint = False
                    #break
                if not isint and not isfloat:
                    break
            if isint:
                isfloat = False
    
            if isint:
                vals = [int(x, 0) for x in col]
            elif isfloat:
                vals = [float(x) for x in col]
            else:
                vals = col
    
            fields.set(colnames[i].lower(), np.array(vals))
            fields._length = len(vals)
    else:
        for i,(col,ct) in enumerate(zip(coldata, coltypes)):
            fields.set(colnames[i].lower(), np.array(col)) #, dtype=ct))

    fields._columns = [c.lower() for c in colnames]

    return fields
