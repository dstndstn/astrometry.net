import os
import tempfile

from astrometry.net.settings import TEMPDIR

def get_temp_file(suffix=''):
    f,fn = tempfile.mkstemp(dir=TEMPDIR, suffix=suffix)
    os.close(f)
    return fn
