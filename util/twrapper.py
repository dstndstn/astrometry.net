import numpy as np

class Twrapper(object):
    def __init__(self, table, lower=True):
        '''
        *table*: astropy.table.Table object.
        '''
        self._table = table
        self._lower = lower
        self._colmap = dict()
        self._revmap = dict()

        if lower:
            for c in table.colnames:
                c2 = c.lower()
                self._colmap[c2] = c
                self._revmap[c] = c2

    def get_column_name(self, c):
        return self._colmap.get(c, c)

    def get_columns(self):
        return [self._revmap.get(c, c) for c in self._table.colnames]

    def cut(self, I):
        keep = np.zeros(len(self), bool)
        keep[I] = True
        I = np.flatnonzero(keep == False)
        self._table.remove_rows(I)

    def rename(self, k, knew):
        k2 = self.get_column_name(k)
        del self._colmap[k]
        if self._lower and k2 in self._revmap:
            del self._revmap[k2]
        self._table.rename_column(k2, knew)

    def __len__(self):
        return len(self._table)

    def __getattr__(self, k):
        k = self.get_column_name(k)
        return self._table.field(k).data

    def __setattr__(self, k, val):
        if k[0] == '_':
            return super().__setattr__(k, val)
        k2 = self.get_column_name(k)
        if k2 in self._table.colnames:
            self._table[k2] = val
        else:
            self._table.add_column(val, name=k)

    def __getitem__(self, I):
        return Twrapper(self._table[I], lower=self._lower)
