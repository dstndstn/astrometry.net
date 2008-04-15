import popen2
import os
import select

def run_command(cmd):
    """
    Run a command and return the text written to stdout and stderr, plus
    the return value.

    Returns: (int return value, string out, string err)
    """
    child = popen2.Popen3(cmd, True)
    (fout, fin, ferr) = (child.fromchild, child.tochild, child.childerr)
    fin.close()
    stdout = fout.fileno()
    stderr = ferr.fileno()
    outbl = []
    errbl = []
    outeof = erreof = 0
    block = 1024
    while True:
        s=[]
        if not outeof:
            s.append(stdout)
        if not erreof:
            s.append(stderr)
        if not len(s):
            break
        (ready, nil1, nil2) = select.select(s, [], [])
        if stdout in ready:
            outchunk = os.read(stdout, block)
            if len(outchunk) == 0:
                outeof = 1
            outbl.append(outchunk)
        if stderr in ready:
            errchunk = os.read(stderr, block)
            if len(errchunk) == 0:
                erreof = 1
            errbl.append(errchunk)
    fout.close()
    ferr.close()
    w = child.wait()
    if not os.WIFEXITED(w):
        return (-100, out, err)
    rtn = os.WEXITSTATUS(w)
    return (rtn, ''.join(outbl), ''.join(errbl))

