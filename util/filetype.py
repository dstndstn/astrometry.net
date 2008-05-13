import os

from astrometry.util.shell import shell_escape

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
    #logverb('Result: "%s"' % typeinfo)

    lst = [line.split(', ', 1) for line in out.split('\n- ')]
    #logverb('Trimmed: "%s"' % typeinfo)
    return list

# Returns a list (usually with just one element) of filetypes:
# eg
#  [ 'Minix filesystem', 'JPEG image data' ]
def filetype_short(fn):
    return [t for (t,nil) in filetype(fn)]
