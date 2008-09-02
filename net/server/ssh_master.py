import os
import tempfile
import select
import subprocess
import time
import tarfile
import sys
from StringIO import StringIO

from urllib import urlencode

from django.http import HttpResponse

from astrometry.util.file import *
from astrometry.net.server.log import log

from astrometry.net.portal.job import Job

# return the ssh config names of the shards.
def get_shards():
    if True:
        return ['iceshard']
    elif False:
        return ['neuron0',
                'neuron1',
                'neuron2',
                'neuron3',
                'neuron4',
                ]
    else:
        return ['shard55',
                'shard56',
                'shard57',
                'shard58',
                'shard59',
                ]

class ShardRequest(object):
    pass

def solve(job, logfunc):
    log('ssh-master.solve', job.jobid)
    axypath = job.get_axy_filename()
    axy = read_file(axypath)
    
    shards = []
    for x in get_shards():
        s = ShardRequest()
        s.ssh = x
        s.command = ['ssh', '-x', '-T', s.ssh]
        s.proc = subprocess.Popen(s.command,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  stdin =subprocess.PIPE,
                                  close_fds=True)
        s.sin = s.proc.stdin
        s.out = s.proc.stdout
        s.err = s.proc.stderr

        # Send the jobid, axy length, and axy contents.
        s.sin.write('%s\n' % job.jobid)
        s.sin.write('%i\n' % len(axy))
        s.sin.write(axy)

        s.running = True
        s.solved = False
        s.tarfiles = []
        s.outdata = []
        shards.append(s)

    firstsolved = None
    while True:
        f = ([s.out for s in shards if s.running and not s.out.closed] +
             [s.err for s in shards if s.running and not s.err.closed])
        #log('selecting on %i files...' % len(f))
        if len(f) == 0:
            break
        (ready, nil1, nil2) = select.select(f, [], [], 1.)

        for i,s in enumerate(shards):
            if not s.running:
                continue
            if s.out in ready:
                # use os.read() rather than readline() because it
                # doesn't block.
                txt = os.read(s.out.fileno(), 102400)
                log('[out %i] --> %i bytes' % (i, len(txt)))
                if len(txt) == 0:
                    s.out.close()
                else:
                    s.outdata.append(txt)
            if s.err in ready:
                txt = os.read(s.err.fileno(), 102400)
                if len(txt) == 0:
                    s.err.close()
                else:
                    # we should log this directly...
                    #log('[err %i] --> "%s"' % (i, str(txt)))
                    lines = txt.split('\n')
                    for l in lines:
                        logfunc(('[%i] ' % i) + l + '\n')
                    
            if s.out.closed:
                s.proc.poll()
                if s.proc.returncode is None:
                    continue
                log('return code from shard %i is %i' % (i, s.proc.returncode))

                s.tardata = ''.join(s.outdata)

                log('tarfile contents:')
                f = StringIO(s.tardata)
                tar = tarfile.open(name='', mode='r|', fileobj=f)
                for tarinfo in tar:
                    log('  ', tarinfo.name, 'is', tarinfo.size, 'bytes in size')
                    if tarinfo.name == 'wcs.fits':
                        s.solved = True
                    # read and save the file contents.
                    ff = tar.extractfile(tarinfo)
                    tarinfo.data = ff.read()
                    ff.close()
                    s.tarfiles.append(tarinfo)
                tar.close()

                if s.solved and firstsolved is None:
                    firstsolved = s

                    # send cancel requests to others.
                    for j,ss in enumerate(shards):
                        if i == j:
                            continue
                        if not ss.running:
                            continue
                        # Needed?
                        #if ss.sin.closed:
                        #    continue
                        ss.sin.write('cancel\n')
                
                # this proc is done!
                s.running = False



    # merge all the resulting tar files into one big tar file.
    # the firstsolved results will be in the base dir, the other
    # shards will be in 1/, 2/, etc.
    f = StringIO()
    tar = tarfile.open(name='', mode='w', fileobj=f)
    i = 1
    for s in shards:
        if s == firstsolved:
            prefix = ''
        else:
            prefix = '%i/' % i
            i += 1
        for tf in s.tarfiles:
            tf.name = prefix + tf.name
            log('  adding (%i bytes) %s' % (tf.size, tf.name))
            ff = StringIO(tf.data)
            tar.addfile(tf, ff)
    tar.close()
    tardata = f.getvalue()
    log('tardata length is', len(tardata))
    f.close()

    return tardata
