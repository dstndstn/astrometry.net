# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE

## This file includes code from Python 3.8's multiprocessing module.
#
# Copyright (c) 2006-2008, R Oudkerk
# Licensed to PSF under a Contributor Agreement.
#

'''
This file provides a subclass of the multiprocessing.Pool class that
tracks the CPU and Wall time of worker processes, as well as time and
I/O spent in pickling data.

It also provides an astrometry.util.ttime Measurement class,
TimingPoolMeas, that records & reports the pool's worker CPU time.

    dpool = TimingPool(4, taskqueuesize=4)
    dmup = multiproc.multiproc(pool=dpool)
    Time.add_measurement(TimingPoolMeas(dpool))
'''

#
# Pool has an _inqueue (_quick_put) and _outqueue (_quick_get)
# and _taskqueue:
#
#   pool.map()  ---> sets cache[]
#               ---> put work on taskqueue
#       handle_tasks thread  ---> gets work from taskqueue
#                            ---> puts work onto inqueue
#       worker threads       ---> get work from inqueue
#                            ---> put results into outqueue
#       handle_results thread --> gets results from outqueue
#                             --> sets cache[]
#
#       meanwhile, handle_workers thread creates new workers as needed.
#
# map() etc add themselves to cache[jobid] = self
# and place work on the task queue.
#
# _handle_tasks pulls tasks from the _taskqueue and puts them
#     on the _inqueue.
#
# _handle_results pulls results from the _outqueue (job,i,obj)
#     and calls cache[job].set(i, obj)
#     (cache[job] is an ApplyResult / MapResult, etc.)
#
# worker threads run the worker() function:
#   run initializer
#   while true:
#     pull task from inqueue
#     job,i,func,arg,kwargs = task
#     put (job,i,result) on outqueue
#
# _inqueue,_outqueue are SimpleQueue (queues.py)



# To capture the pickle traffic, we subclass the Connection, Pipe, and Queue
# classes.

# To capture the worker CPU time, we intercept the messages put on the inqueue
# and fetched from the outqueue --
# The inqueue includes job ids, the function to called, and args.  We wrap the
# function in time_func() object, so the worker records the CPU time, and
# returns a tuple of the original result and the timing information.
# When reading from the outqueue, we peel off the timing information.

import os
import sys
import time
import struct
import io

import multiprocessing
from multiprocessing.queues import SimpleQueue
from multiprocessing.connection import Connection
from multiprocessing.pool import Pool
from multiprocessing import context
try:
    ForkingPickler = context.reduction.ForkingPickler
except:
    # python3.5
    ForkingPickler = multiprocessing.reduction.ForkingPickler

from astrometry.util.ttime import CpuMeas

class TimingConnection(Connection):
    '''
    A multiprocessing *Connection* subclass that keeps track of how
    many objects and how many bytes are pickled, and how long that
    takes.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ptime = 0.
        self.uptime = 0.
        self.pbytes = 0
        self.upbytes = 0
        self.pobjs = 0
        self.upobjs = 0

    def stats(self):
        '''
        Returns statistics of the objects handled by this connection.

        This method is new in this subclass.
        '''
        return dict(pickle_objs = self.pobjs,
                    pickle_bytes = self.pbytes,
                    pickle_megabytes = 1e-6 * self.pbytes,
                    pickle_cputime = self.ptime,
                    unpickle_objs = self.upobjs,
                    unpickle_bytes = self.upbytes,
                    unpickle_megabytes = 1e-6 * self.upbytes,
                    unpickle_cputime = self.uptime)

    def recv(self):
        """Receive a (picklable) object.

        This is overriding a core method in the Connection superclass."""
        self._check_closed()
        self._check_readable()
        t0 = time.time()
        buf = self._recv_bytes()
        buf = buf.getbuffer()
        n = len(buf)
        obj = ForkingPickler.loads(buf)
        dt = time.time() - t0
        self.upbytes += n
        self.uptime += dt
        self.upobjs += 1
        return obj

    def send(self, obj):
        """Send a (picklable) object"""
        self._check_closed()
        self._check_writable()
        t0 = time.time()
        bb = ForkingPickler.dumps(obj)
        dt = time.time() - t0
        self.pbytes += len(bb)
        self.ptime += dt
        self.pobjs += 1
        return self._send_bytes(bb)

    # Borrowed from the python3.8 superclass -- handles > 2GB pickles.
    def _send_bytes(self, buf):
        n = len(buf)
        if n > 0x7fffffff:
            pre_header = struct.pack("!i", -1)
            header = struct.pack("!Q", n)
            self._send(pre_header)
            self._send(header)
            self._send(buf)
        else:
            # For wire compatibility with 3.7 and lower
            header = struct.pack("!i", n)
            if n > 16384:
                # The payload is large so Nagle's algorithm won't be triggered
                # and we'd better avoid the cost of concatenation.
                self._send(header)
                self._send(buf)
            else:
                # Issue #20540: concatenate before sending, to avoid delays due
                # to Nagle's algorithm on a TCP socket.
                # Also note we want to avoid sending a 0-length buffer separately,
                # to avoid "broken pipe" errors if the other end closed the pipe.
                self._send(header + buf)

    # Borrowed from the python3.8 superclass -- handles > 2GB pickles.
    def _recv_bytes(self, maxsize=None):
        buf = self._recv(4)
        size, = struct.unpack("!i", buf.getvalue())
        if size == -1:
            buf = self._recv(8)
            size, = struct.unpack("!Q", buf.getvalue())
        if maxsize is not None and size > maxsize:
            return None
        return self._recv(size)

## This is essential! -- register a special pickler for our
## Connection subclass.  Without this, the file descriptors don't
## get set correctly across the spawn call.
from multiprocessing import reduction
from multiprocessing.connection import reduce_connection
reduction.register(TimingConnection, reduce_connection)

def TimingPipe(track_input, track_output):
    '''
    Creates a pipe composed of Connection or TimingConnection objects,
    depending on what we want to record.
    '''
    fd1, fd2 = os.pipe()
    r = TimingConnection(fd1, writable=False)
    w = TimingConnection(fd2, readable=False)
    return r,w

def _sum_object_stats(O1, O2):
    ''' used to sum the statistics from two objects that may have
    a stats() function that returns a dict.'''
    S1 = S2 = {}
    if hasattr(O1, 'stats'):
        S1 = O1.stats()
    if hasattr(O2, 'stats'):
        S2 = O2.stats()
    # merge/sum
    for k,v in S2.items():
        if not k in S1:
            S1[k] = v
        else:
            S1[k] = S1[k] + S2[k]
    return S1

class TimingSimpleQueue(SimpleQueue):
    '''
    A *SimpleQueue* subclass that uses a *TimingPipe* object to keep
    stats on how much pickling objects costs.
    '''

    # new method
    def stats(self):
        '''
        Returns stats on the objects sent through this queue.
        '''
        return _sum_object_stats(self._reader, self._writer)

    def __init__(self, track_input, track_output, ctx):
        self._reader, self._writer = TimingPipe(track_input, track_output)
        self._rlock = ctx.Lock()
        self._poll = self._reader.poll
        if sys.platform == 'win32':
            self._wlock = None
        else:
            self._wlock = ctx.Lock()

class TimingPool(Pool):
    '''
    A python multiprocessing Pool subclass that keeps track of the
    resources used by workers, and tracks the expense of pickling
    objects.
    '''
    # New functions added:
    def get_worker_cpu(self):
        return self.beancounter.get_cpu()
    def get_worker_wall(self):
        return self.beancounter.get_wall()
    def get_pickle_traffic(self):
        return _sum_object_stats(self._inqueue, self._outqueue)
    def get_pickle_traffic_string(self):
        S = self.get_pickle_traffic()
        return (('  pickled %i objs, %g MB, using %g s CPU\n' +
                 'unpickled %i objs, %g MB, using %g s CPU') %
                 tuple(S.get(k,0) for k in [
                     'pickle_objs', 'pickle_megabytes', 'pickle_cputime',
                     'unpickle_objs', 'unpickle_megabytes', 'unpickle_cputime']))

    def __init__(self, *args,
                 track_send_pickles=True,
                 track_recv_pickles=True,
                 **kwargs):
        self.track_send_pickles = track_send_pickles
        self.track_recv_pickles = track_recv_pickles
        super().__init__(*args, **kwargs)
        self.beancounter = BeanCounter()

    def _setup_queues(self):
        if self.track_send_pickles:
            self._inqueue = TimingSimpleQueue(False, True, self._ctx)
        else:
            self._inqueue = self._ctx.SimpleQueue()
        if self.track_recv_pickles:
            self._outqueue = TimingSimpleQueue(True, False, self._ctx)
        else:
            self._outqueue = self._ctx.SimpleQueue()
        self._quick_get = self._quick_get_wrapper
        self._real_quick_get = self._outqueue._reader.recv
        self._quick_put = self._quick_put_wrapper
        self._real_quick_put = self._inqueue._writer.send

    def _quick_get_wrapper(self):
        # Peel off the timing results
        obj = self._real_quick_get()
        if obj is None:
            return obj
        job, i, (success, res) = obj
        if success:
            res,dt = res
            self.beancounter.add_time(dt)
        return job, i, (success, res)

    def _quick_put_wrapper(self, task):
        # Wrap tasks with timing results
        if task is not None:
            job, i, func, args, kwds = task
            func = time_func(func)
            task = job, i, func, args, kwds
        return self._real_quick_put(task)

class time_func(object):
    '''A wrapper that records the CPU time used by a call, and returns that
    along with the result.'''
    def __init__(self, func):
        self.func = func
    def __call__(self, *args, **kwargs):
        t1 = CpuMeas()
        R = self.func(*args, **kwargs)
        t2 = CpuMeas()
        dt = (t2.cpu_seconds_since(t1), t2.wall_seconds_since(t1))
        return R,dt

######################################################
# These classes handle tracking time & the Measurement framework.

class BeanCounter(object):
    '''
    A class to keep track of the CPU and Wall time used by workers.
    '''
    def __init__(self):
        self.cpu = 0.
        self.wall = 0.
        self.lock = multiprocessing.Lock()

    ### LOCKING
    def add_time(self, dt):
        self.lock.acquire()
        try:
            (cpu, wall) = dt
            self.cpu += cpu
            self.wall += wall
        finally:
            self.lock.release()
    def get_cpu(self):
        self.lock.acquire()
        try:
            return self.cpu
        finally:
            self.lock.release()
    def get_wall(self):
        self.lock.acquire()
        try:
            return self.wall
        finally:
            self.lock.release()
    def __str__(self):
        return 'CPU time: %.3fs s, Wall time: %.3fs' % (self.get_cpu(), self.get_wall())

class TimingPoolMeas(object):
    '''
    An astrometry.util.ttime Measurement object to measure the resources used
    by workers, and by pickling objects.
    '''
    def __init__(self, pool, pickleTraffic=True):
        self.pool = pool
        self.nproc = pool._processes
        self.pickleTraffic = pickleTraffic
    def __call__(self):
        return TimingPoolTimestamp(self.pool, self.pickleTraffic, self.nproc)

class TimingPoolTimestamp(object):
    '''
    The current resources used by a pool of workers, for
    astrometry.util.ttime
    '''
    def __init__(self, pool, pickleTraffic, nproc):
        self.pool = pool
        self.nproc = nproc
        self.t = self.now(pickleTraffic)
        self.cpu = CpuMeas()
    def format_diff(self, other):
        t1 = self.t
        t0 = other.t
        wall = self.cpu.wall_seconds_since(other.cpu)
        main_cpu = self.cpu.cpu_seconds_since(other.cpu)
        worker_cpu = t1['worker_cpu'] - t0['worker_cpu']
        worker_wall = t1['worker_wall'] - t0['worker_wall']
        use = (main_cpu + worker_cpu) / wall
        s = ('%.3f s worker CPU, %.3f s worker Wall, Wall: %.3f s, Cores in use: %.2f, Total efficiency (on %i cores): %.1f %%' %
             (worker_cpu, worker_wall, wall, use, self.nproc, 100.*use / float(self.nproc)))
        if 'pickle_objs' in self.t:
            s += (', pickled %i/%i objs, %.1f/%.1f MB' %
                  tuple(t1[k] - t0[k] for k in [
                        'pickle_objs', 'unpickle_objs',
                        'pickle_megabytes', 'unpickle_megabytes']))
        return s

    def now(self, pickleTraffic):
        if pickleTraffic:
            stats = self.pool.get_pickle_traffic()
        else:
            stats = dict()
        stats.update(worker_cpu = self.pool.get_worker_cpu(),
                     worker_wall = self.pool.get_worker_wall())
        return stats

######################################################



def test_func(N):
    import numpy as np
    r = []
    for i in range(N):
        arr = np.random.normal(size=1000000)
        r.append(arr)
    s = 0.
    for arr in r:
        s = s + arr**2
    return s

def test_func_2(arr):
    import numpy as np
    return np.sum(arr**2)

def test_func_3(arr):
    return arr[::2]**2

def test():
    #import logging
    #multiprocessing.log_to_stderr().setLevel(logging.DEBUG)

    # per multiprocess.Pool documentation (as of Python 3.8 at least), can't just
    # let a Pool fall out of scope; must close or you risk the process hanging.
    # This definitely happens on Linux!
    with TimingPool(4) as pool:
        R = pool.map(test_func, [10, 20, 30])
        print('worker cpu:', pool.get_worker_cpu())
        print('worker wall:', pool.get_worker_wall())
        print('pickles:', pool.get_pickle_traffic_string())

    import numpy as np
    print('Creating second pool')
    with TimingPool(4) as pool:
        print('Using second pool')
        R = pool.map(test_func_2, [np.random.normal(size=1000000) for x in range(5)])
        print('Got result from second pool')
        print('worker cpu:', pool.get_worker_cpu())
        print('worker wall:', pool.get_worker_wall())
        print('pickles:', pool.get_pickle_traffic_string())

    from astrometry.util.ttime import Time
    with TimingPool(4, track_send_pickles=False, track_recv_pickles=False) as pool:
        m = TimingPoolMeas(pool, pickleTraffic=False)
        Time.add_measurement(m)
        t0 = Time()
        R = pool.map(test_func, [20, 20, 20])
        print(Time()-t0)
        Time.remove_measurement(m)

    with TimingPool(4) as pool:
        Time.add_measurement(TimingPoolMeas(pool, pickleTraffic=True))
        t0 = Time()
        R = pool.map(test_func_3, [np.random.normal(size=1000000) for x in range(5)])
        print(Time()-t0)


def test_func_4(arr):
    arr = arr**2
    return arr

def test_jumbo():
    # Test jumbo (> 2 GB) args/results
    import numpy as np
    from astrometry.util.ttime import Time
    with TimingPool(2) as pool:
        Time.add_measurement(TimingPoolMeas(pool, pickleTraffic=True))
        t0 = Time()
        R = pool.map(test_func_4, [np.ones(int(2.1 * 1024 * 1024 * 1024 / 8))])
        print('Jumbo:', np.sum(R))
        print(Time()-t0)

def test_input_generator(n):
    for i in range(n):
        import numpy as np
        x = np.random.random((1000,1000))
        print('Yielding input', i)
        yield (i,x)

def test_sleep(x):
    import time
    time.sleep(3.)
    return x

def test_queue():
    # Generate a bunch of tasks with 1MB pickles...
    in_iter = test_input_generator(100)
    with TimingPool(4) as pool:
        out_iter = pool.imap_unordered(test_sleep, in_iter)
        while True:
            try:
                r = next(out_iter)
                i,x = r
                print('Got result', i)
            except StopIteration:
                print('StopIteration')
                break

if __name__ == '__main__':
    #test_jumbo()
    #test()
    test_queue()
    sys.exit()
