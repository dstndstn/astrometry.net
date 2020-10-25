import os
import tempfile

from astrometry.net.settings import TEMPDIR

def get_temp_file(suffix='', tempfiles=None):
    f,fn = tempfile.mkstemp(dir=TEMPDIR, suffix=suffix)
    os.close(f)
    if tempfiles is not None:
        tempfiles.append(fn)
    return fn
