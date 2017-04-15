# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
import subprocess
import os
import select
import sys
from subprocess import PIPE
import sys

py3 = (sys.version_info[0] >= 3)

# Returns (rtn, out, err)
def run_command(cmd, timeout=None, callback=None, stdindata=None,
                tostring=True):
    """
    Run a command and return the text written to stdout and stderr, plus
    the return value.

    In python3, if *tostring* is True, the output and error streams
    will be converted to unicode, otherwise will be returned as bytes.
    
    Returns: (int return value, string out, string err)
    """
    child = subprocess.Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True)
    (fin, fout, ferr) = (child.stdin, child.stdout, child.stderr)

    stdin = fin.fileno()
    stdout = fout.fileno()
    stderr = ferr.fileno()
    outdata = []
    errdata = []
    ineof = outeof = erreof = False
    block = 1024
    while True:
        readers = []
        writers = []
        if not ineof: writers.append(stdin)
        if not outeof: readers.append(stdout)
        if not erreof: readers.append(stderr)
        if not len(readers):
            break
        (ready_readers, ready_writers, _) = select.select(readers, writers, [], timeout)
        if stdin in ready_writers and stdindata:
            bytes_written = os.write(stdin, stdindata[:block])
            stdindata = stdindata[bytes_written:]
            if not stdindata:
                fin.close()
                ineof = True
        if stdout in ready_readers:
            outchunk = os.read(stdout, block)
            if len(outchunk) == 0:
                outeof = True
            outdata.append(outchunk)
        if stderr in ready_readers:
            errchunk = os.read(stderr, block)
            if len(errchunk) == 0:
                erreof = True
            errdata.append(errchunk)
        if callback:
            callback()
    fout.close()
    ferr.close()
    w = child.wait()
    if py3:
        out = b''.join(outdata)
        err = b''.join(errdata)
        if tostring:
            out = out.decode()
            err = err.decode()
    else:
        out = ''.join(outdata)
        err = ''.join(errdata)

    if not os.WIFEXITED(w):
        return (-100, out, err)
    rtn = os.WEXITSTATUS(w)
    return (rtn, out, err)
