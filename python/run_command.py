import subprocess
import os
import select
from subprocess import PIPE

# Returns (rtn, out, err)
def run_command(cmd, timeout=None, callback=None, stdindata=None):
    """
    Run a command and return the text written to stdout and stderr, plus
    the return value.

    Returns: (int return value, string out, string err)
    """
    child = subprocess.Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True)
    (fin, fout, ferr) = (child.stdin, child.stdout, child.stderr)

    stdin = fin.fileno()
    stdout = fout.fileno()
    stderr = ferr.fileno()
    outbl = []
    errbl = []
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
            outbl.append(outchunk)
        if stderr in ready_readers:
            errchunk = os.read(stderr, block)
            if len(errchunk) == 0:
                erreof = True
            errbl.append(errchunk)
        if callback:
            callback()
    fout.close()
    ferr.close()
    w = child.wait()
    out = ''.join(outbl)
    err = ''.join(errbl)
    if not os.WIFEXITED(w):
        return (-100, out, err)
    rtn = os.WEXITSTATUS(w)
    return (rtn, out, err)
