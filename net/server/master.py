import os
import tempfile
#import popen2
import select
import subprocess
import time
import tarfile
import sys
from StringIO import StringIO

from urllib import urlencode

from astrometry.util.file import *
from astrometry.net.server.log import log

class ShardRequest(object):
    def __init__(self, url):
        self.url = url

def get_shard_urls():
    return ['http://oven.cosmo.fas.nyu.edu:8888/test/shard/solve/',
            'http://oven.cosmo.fas.nyu.edu:8888/test/shard/solve/',
            ]

def solve(request):
    log('master.solve')

    jobid = request.POST.get('jobid')
    if not jobid:
        return HttpResponse('no jobid')
    axy = request.POST.get('axy')
    if not axy:
        return HttpResponse('no axy')

    # FIXME
    #axy = axy.decode('base64_codec')

    # list of shard URLs
    #  (all cluster machines will have same apache config,
    #   so it'll just be different machine names)

    reqs = [ShardRequest(url) for url in get_shard_urls()]

    # write the encoded POST data to a temp file...
    data = urlencode({ 'axy': axy, 'jobid': jobid })

    (f, postfile) = tempfile.mkstemp('', 'postdata')
    os.close(f)
    write_file(data, postfile)

    for req in reqs:
        # for each index / shard, wget the solve request URL
        req.command = ['wget', '-S', '-nv', '-O', '-', '--post-file', postfile, req.url]
        req.proc = subprocess.Popen(req.command,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    close_fds=True)
        req.out = req.proc.stdout
        req.err = req.proc.stderr
        req.outdata = []
        req.running = True

    while True:
        s = []
        for req in reqs:
            if req.running:
                s.append(req.out)
            if req.running:
                s.append(req.err)
        log('selecting on %i files...' % len(s))
        (ready, nil1, nil2) = select.select(s, [], [], 1.)
        # how much can we read without blocking?
        for i,req in enumerate(reqs):
            if req.out in ready:
                if req.out.closed:
                    log('stdout from shard %i is closed.' % i)
                else:
                    # use os.read() rather than readline() because it
                    # doesn't block.
                    txt = os.read(req.out.fileno(), 102400)
                    log('[out %i] --> %i bytes' % (i, len(txt)))
                    req.outdata.append(txt)
            if req.err in ready:
                if req.err.closed:
                    log('stderr from shard %i is closed.' % i)
                else:
                    txt = os.read(req.err.fileno(), 102400)
                    log('[err %i] --> "%s"' % (i, str(txt)))
            req.proc.poll()
            if req.proc.returncode is not None:
                log('return code from shard %i is %i' % (i, req.proc.returncode))
                while True:
                    txt = os.read(req.out.fileno(), 102400)
                    log('[out %i] --> %i bytes' % (i, len(txt)))
                    if len(txt) == 0:
                        break
                    req.outdata.append(txt)
                req.out.close()
                req.err.close()

                req.tardata = ''.join(req.outdata)

                f = StringIO(req.tardata)
                tar = tarfile.open(mode="r|", fileobj=f)
                for tarinfo in tar:
                    log(tarinfo.name, "is", tarinfo.size, "bytes in size")
                    #tar.extract(tarinfo)
                tar.close()

        time.sleep(1)

    #for req in reqs:
    #    req.outdata = ''.join(req.outdata)

    # select on the processes finishing
    # -> (a) crash
    # -> (b) finished, solved
    # -> (c) finished, unsolved
    # (b), wget(?) the cancel URLs - should lead to processes
    #     finishing and connections being closed.


def cancel(request):
    jobid = request.GET.get('jobid')
    if not jobid:
        return HttpResponse('no jobid')



from urllib import urlencode
from urllib2 import urlopen
from astrometry.util.file import *

def test(request):
    axy = read_file('/home/gmaps/test/astrometry/perseus_cfht.axy')
    axy = axy.encode('base64_codec')
    p = request.GET.copy()
    p['axy'] = axy
    p['jobid'] = '123456'
    request.GET = None
    request.POST = p
    return solve(request)
