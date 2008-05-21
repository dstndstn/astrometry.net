import os
import tempfile
#import popen2
import select
import subprocess
import time

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
        req.command = ['wget', '-nv', '-O', '-', '--post-file', postfile, req.url]
        req.proc = subprocess.Popen(req.command,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    close_fds=True)
        req.out = req.proc.stdout
        req.err = req.proc.stderr

    X=0
    while X<10:
        X+=1
    #while True:
        s = []
        for req in reqs:
            if not req.out.closed:
                s.append(req.out)
            if not req.err.closed:
                s.append(req.err)
        log('select()...')
        (ready, nil1, nil2) = select.select(s, [], [])
        # how much can we read without blocking?
        for i,req in enumerate(reqs):
            if req.out in ready:
                if req.out.closed:
                    log('stdout from shard %i is closed.' % i)
                else:
                    #log('stdout from shard %i is ready.' % i)
                    #txt = req.out.readline()
                    txt = os.read(req.out.fileno(), 102400)
                    # FIXME - do something!
                    #log('--> "%s"' % str(txt))
                    log('[out %i] --> %i bytes' % (i, len(txt)))
            if req.err in ready:
                if req.err.closed:
                    log('stderr from shard %i is closed.' % i)
                else:
                    #log('reading stderr from shard %i' % i)
                    #txt = req.err.readline()
                    txt = os.read(req.err.fileno(), 102400)
                    log('[err %i] --> "%s"' % (i, str(txt)))
            req.proc.poll()
            if req.proc.returncode is not None:
                log('return code from shard %i is %i' % (i, req.proc.returncode))
                # FIXME - read the rest of the data...
                req.out.close()
                req.err.close()

        time.sleep(1)
        
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
