# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE

import os
import sys

from astrometry.util.shell import shell_escape
from astrometry.util.run_command import run_command

# DEBUG
import logging
def logverb(*msg):
    if (sys.version_info > (3,0)):
        logging.debug(' '.join(msg))
    else:
        logging.debug(' '.join([str(m).decode('latin_1', 'backslashreplace') for m in msg]))

# Returns a list (usually with just one element) of 2-tuples:
#  [ (filetype, detail), (filetype, detail) ]
# eg
#  [ ('Minix filesystem', 'version 2'),
#    ('JPEG image data', 'JFIF standard 1.01') ]
def filetype(fn):
    filecmd = 'file -b -N -L -k %s'

    cmd = filecmd % shell_escape(fn)
    (rtn,out,err) = run_command(cmd)
    if rtn:
        logverb('"file" command failed.  Command: "%s"' % cmd)
        logverb('  ', out)
        logverb('  ', err)
        return None

    out = out.strip()
    logverb('File: "%s"' % out)
    lst = []

    # The "file -r" flag, removed in some Ubuntu versions, used to
    # tell it not to convert non-printable characters to octal.  Without -r,
    # some versions print the string r'\012- ' instead of "\n- ".  Do that
    # manually here.
    out = out.replace(r'\012- ', '\n- ')

    for line in out.split('\n- '):
        if line.endswith('\n-'):
            line = line[:-2]
        if len(line) == 0:
            continue
        p = line.split(', ', 1)
        if len(p) == 2:
            lst.append(tuple(p))
        else:
            lst.append((p[0], ''))
    return lst

# Returns a list (usually with just one element) of filetypes, or None if no filetypes are found:
# eg
#  [ 'Minix filesystem', 'JPEG image data' ]
def filetype_short(fn):
    ft = filetype(fn)
    if ft is None:
        return None
    return [t for (t,nil) in ft]
