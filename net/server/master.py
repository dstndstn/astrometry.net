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

from django.http import HttpResponse

from astrometry.util.file import *
from astrometry.net.server.log import log

class ShardRequest(object):
    def __init__(self, url):
        self.url = url

def get_shard_urls():
    return ['http://localhost:9058/test/shard/',
            'http://localhost:9059/test/shard/',
            ]

def get_shard_solve_urls():
    return [url + 'solve/' for url in get_shard_urls()]

def get_shard_cancel_urls(jobid):
    return [url + 'cancel/?jobid=%s' % jobid for url in get_shard_urls()]

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

    reqs = [ShardRequest(url) for url in get_shard_solve_urls()]
    for (r,u) in zip(reqs, get_shard_cancel_urls(jobid)):
        r.cancelurl = u

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
        req.solved = False
        req.tarfiles = []

    firstsolved = None

    while True:
        s = []
        for req in reqs:
            if req.running and not req.out.closed:
                s.append(req.out)
            if req.running and not req.err.closed:
                s.append(req.err)
        log('selecting on %i files...' % len(s))
        if len(s) == 0:
            break
        (ready, nil1, nil2) = select.select(s, [], [], 1.)
        # how much can we read without blocking?
        for i,req in enumerate(reqs):
            if not req.running:
                continue
            if req.out in ready:
                # use os.read() rather than readline() because it
                # doesn't block.
                txt = os.read(req.out.fileno(), 102400)
                log('[out %i] --> %i bytes' % (i, len(txt)))
                if len(txt) == 0:
                    req.out.close()
                else:
                    req.outdata.append(txt)
            if req.err in ready:
                txt = os.read(req.err.fileno(), 102400)
                if len(txt) == 0:
                    req.err.close()
                else:
                    log('[err %i] --> "%s"' % (i, str(txt)))

            if req.out.closed:
                req.proc.poll()
                if req.proc.returncode is None:
                    continue
                log('return code from shard %i is %i' % (i, req.proc.returncode))

                req.tardata = ''.join(req.outdata)

                log('tarfile contents:')
                f = StringIO(req.tardata)
                tar = tarfile.open(mode='r|', fileobj=f)
                for tarinfo in tar:
                    log('  ', tarinfo.name, 'is', tarinfo.size, 'bytes in size')
                    if tarinfo.name == 'solved':
                        req.solved = True
                    # read and save the file contents.
                    ff = tar.extractfile(tarinfo)
                    tarinfo.data = ff.read()
                    ff.close()
                    req.tarfiles.append(tarinfo)
                tar.close()

                if req.solved:
                    if firstsolved is None:
                        firstsolved = req
                if req == firstsolved:
                    # send cancel requests to others.
                    # these should return very quickly...
                    for r in reqs:
                        if r == req:
                            continue
                        log('sending cancel request to url', r.cancelurl)
                        r.cancommand = ['wget', '-nv', '-O', '-', r.cancelurl]
                        r.canproc = subprocess.Popen(r.cancommand, close_fds = True)

                # this proc is done!
                req.running = False

        time.sleep(1)

    for i,req in enumerate(reqs):
        if req.solved:
            log('request %i solved.' % i)

    # select on the processes finishing
    # -> (a) crash
    # -> (b) finished, solved
    # -> (c) finished, unsolved
    # (b), wget(?) the cancel URLs - should lead to processes
    #     finishing and connections being closed.


    # merge all the resulting tar files into one big tar file.
    # the firstsolved results will be in the base dir, the other
    # shards will be in 1/, 2/, etc.
    f = StringIO()
    tar = tarfile.open(mode='w', fileobj=f)
    tar.debug = 3
    i = 1
    for r in reqs:
        if r == firstsolved:
            prefix = ''
        else:
            prefix = '%i/' % i
            i += 1
        for tf in r.tarfiles:
            tf.name = prefix + tf.name
            log('  adding (%i bytes) %s' % (tf.size, tf.name))
            ff = StringIO(tf.data)
            tar.addfile(tf, ff)
    tar.close()
    tardata = f.getvalue()
    log('tardata length is', len(tardata))
    f.close()
    res = HttpResponse()
    res['Content-type'] = 'application/x-tar'
    res.write(tardata)

    #if firstsolved is None:
    #    res.write('unsolved')
    #else:
    #    log('returning tar data from shard %s' % firstsolved.url)
    #    res.write(firstsolved.tardata)

    return res

def cancel(request):
    jobid = request.GET.get('jobid')
    if not jobid:
        return HttpResponse('no jobid')




def test(request):
    axy = read_file('/home/gmaps/test/astrometry/perseus_cfht.axy')
    axy = axy.encode('base64_codec')
    p = request.GET.copy()
    p['axy'] = axy
    p['jobid'] = '123456'
    request.GET = None
    request.POST = p
    return solve(request)
