import os

from astrometry.util.shell import shell_escape

# DEBUG
import logging
def logverb(*msg):
    logging.debug(' '.join([str(m).decode('latin_1', 'backslashreplace') for m in msg]))

# Returns a list (usually with just one element) of 2-tuples:
#  [ (filetype, detail), (filetype, detail) ]
# eg
#  [ ('Minix filesystem', 'version 2'),
#    ('JPEG image data', 'JFIF standard 1.01') ]
def filetype(fn):
    filecmd = 'file -b -N -L -k -r %s'

    cmd = filecmd % shell_escape(fn)
    #logverb('Running: "%s"' % cmd)
    (filein, fileout) = os.popen2(cmd)
    out = fileout.read().strip()

    logverb('File: "%s"' % out)

    parts = [line.split(', ', 1) for line in out.split('\n- ')]
    lst = []
    for p in parts:
        if len(p) == 2:
            lst.append(tuple(p))
        else:
            lst.append((p[0], ''))
    #lst = [tuple(line.split(', ', 1)) for line in out.split('\n- ')]
    #logverb('Trimmed: "%s"' % typeinfo)
    return lst

# Returns a list (usually with just one element) of filetypes:
# eg
#  [ 'Minix filesystem', 'JPEG image data' ]
def filetype_short(fn):
    return [t for (t,nil) in filetype(fn)]
