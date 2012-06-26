import os
import tempfile

from astrometry.net.settings import TEMPDIR

def get_temp_file():
    f,fn = tempfile.mkstemp(dir=TEMPDIR)
    os.close(f)
    return fn
