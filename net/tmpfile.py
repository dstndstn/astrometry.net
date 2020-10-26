import os
import tempfile

from astrometry.net import settings

def get_temp_file(suffix='', tempfiles=None):
    f,fn = tempfile.mkstemp(dir=settings.TEMPDIR, suffix=suffix)
    os.close(f)
    if tempfiles is not None:
        tempfiles.append(fn)
    return fn
