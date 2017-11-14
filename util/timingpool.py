# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function

import multiprocessing.queues
import multiprocessing.pool
import multiprocessing.synchronize

from multiprocessing.util import debug

import _multiprocessing
import threading
import time
import os

import sys
py3 = (sys.version_info[0] >= 3)

if py3:
    # py3 loads cPickle if available
    import pickle
    # py3 renames Queue to queue
    import queue
    Connection = multiprocessing.connection.Connection
else:
    # py2
    import cPickle as pickle
    import Queue as queue
    Connection = _multiprocessing.Connection

from astrometry.util.ttime import CpuMeas

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
# In Python 2.7 (and 2.6):
#
# Pool has an _inqueue (_quick_put) and _outqueue (_quick_get)
# and _taskqueue:
#
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

# _inqueue,_outqueue are SimpleQueue (queues.py)
# get->recv=_reader.recv and put->send=_writer.send
# _reader,_writer = Pipe(duplex=False)

# Pipe (connection.py)
# uses os.pipe() with a _multiprocessing.Connection()
# on each fd.

# _multiprocessing = /u32/python/src/Modules/_multiprocessing/pipe_connection.c
# -> connection.h : send() is connection_send_obj()
# which uses pickle.
#
# Only _multiprocessing/socket_connection.c is used on non-Windows platforms.

class TimingConnection():
    '''
    A *Connection* wrapper that keeps track of how many objects and
    how many bytes are pickled, and how long that takes.
    '''
    
    def __init__(self, fd, writable=True, readable=True):
        self.real = Connection(fd, writable=writable, readable=readable)
        self.ptime = 0.
        self.uptime = 0.
        self.pbytes = 0
        self.upbytes = 0
        self.pobjs = 0
        self.upobjs = 0

    def stats(self):
        '''
        Returns statistics of the objects handled by this connection.
        '''
        return dict(pickle_objs = self.pobjs,
                    pickle_bytes = self.pbytes,
                    pickle_megabytes = 1e-6 * self.pbytes,
                    pickle_cputime = self.ptime,
                    unpickle_objs = self.upobjs,
                    unpickle_bytes = self.upbytes,
                    unpickle_megabytes = 1e-6 * self.upbytes,
                    unpickle_cputime = self.uptime)

    def poll(self):
        return self.real.poll()

    # called by py2 (multiprocessing.queues.Queue.get())
    def recv(self):
        bb = self.real.recv_bytes()
        t0 = time.time()
        obj = pickle.loads(bb)
        dt = time.time() - t0
        self.upbytes += len(bb)
        self.uptime += dt
        self.upobjs += 1
        return obj

    # called by py2
    def send(self, obj):
        t0 = time.time()
        s = pickle.dumps(obj, -1)
        dt = time.time() - t0
        self.pbytes += len(s)
        self.ptime += dt
        self.pobjs += 1
        return self.real.send_bytes(s)

    # called by py3 (multiprocessing.queues.Queue.get())
    def recv_bytes(self):
        bb = self.real.recv_bytes()
        self.upbytes += len(bb)
        #self.uptime += ... unpickled by the Queue
        self.upobjs += 1
        return bb
    
    # called by py3
    def send_bytes(self, bb):
        self.pbytes += len(bb)
        #self.ptime += 
        self.pobjs += 1
        return self.real.send_bytes(bb)
    
    def close(self):
        return self.real.close()

def TimingPipe():
    '''
    Creates a pipe composed of two TimingConnection objects.
    '''
    fd1, fd2 = os.pipe()
    c1 = TimingConnection(fd1, writable=False)
    c2 = TimingConnection(fd2, readable=False)
    return c1,c2

class TimingSimpleQueue(multiprocessing.queues.SimpleQueue):
    '''
    A *SimpleQueue* subclass that uses a *TimingPipe* object to keep
    stats on how much pickling objects costs.
    '''
    # new method
    def stats(self):
        '''
        Returns stats on the objects sent through this queue.
        '''
        S1 = self._reader.stats()
        S2 = self._writer.stats()
        return dict([(k, S1[k]+S2[k]) for k in S1.keys()])

    def __init__(self):
        (self._reader, self._writer) = TimingPipe()
        self._rlock = multiprocessing.Lock()
        self._wlock = multiprocessing.Lock()
        # py2
        if hasattr(self, '_make_methods'):
            self._make_methods()
        # _make_methods creates two methods:
        #
        #  get:  self._rlock.acquire();
        #        self._reader.recv();
        #        self._rlock.release();
        #
        #  put:  self._wlock.acquire();
        #        self._write_send();
        #        self._wlock.release();
        #
        
def timing_worker(inqueue, outqueue, progressqueue,
                 initializer=None, initargs=(),
                 maxtasks=None):
    '''
    A modified worker thread that tracks how much CPU time is used.
    '''
    assert(maxtasks is None or (type(maxtasks) == int and maxtasks > 0))
    put = outqueue.put
    get = inqueue.get
    if hasattr(inqueue, '_writer'):
        inqueue._writer.close()
        outqueue._reader.close()
        if progressqueue is not None:
            progressqueue._reader.close()
        
    if initializer is not None:
        initializer(*initargs)

    mypid = os.getpid()
        
    completed = 0
    #t0 = time.time()
    while maxtasks is None or (maxtasks and completed < maxtasks):
        #print 'PID %i @ %f: get task' % (os.getpid(), time.time()-t0)
        try:
            # print 'Worker pid', os.getpid(), 'getting task'
            task = get()
        except (EOFError, IOError):
            debug('worker pid ' + os.getpid() +
                  ' got EOFError or IOError -- exiting')
            break
        except KeyboardInterrupt as e:
            print('timing_worker caught KeyboardInterrupt during get()')
            put((None, None, (None,(False,e))))
            raise SystemExit('ctrl-c')
            break

        if task is None:
            debug('worker got sentinel -- exiting')
            break

        # print 'PID %i @ %f: unpack task' % (os.getpid(), time.time()-t0)
        job, i, func, args, kwds = task

        if progressqueue is not None:
            try:
                # print 'Worker pid', os.getpid(), 'writing to progressqueue'
                progressqueue.put((job, i, mypid))
            except (EOFError, IOError):
                print('worker got EOFError or IOError on progress queue -- exiting')
                break

        t1 = CpuMeas()
        #print 'PID %i @ %f: run task' % (os.getpid(), time.time()-t0)
        try:
            success,val = (True, func(*args, **kwds))
        except Exception as e:
            success,val = (False, e)
            #print 'timing_worker: caught', e
        except KeyboardInterrupt as e:
            success,val = (False, e)
            #print 'timing_worker: caught ctrl-C during work', e
            #print type(e)
            put((None, None, (None,(False,e))))
            raise
        #print 'PID %i @ %f: ran task' % (os.getpid(), time.time()-t0)
        t2 = CpuMeas()
        dt = (t2.cpu_seconds_since(t1), t2.wall_seconds_since(t1))
        put((job, i, dt,(success,val)))
        completed += 1
        #print 'PID %i @ %f: sent result' % (os.getpid(), time.time()-t0)
    debug('worker exiting after %d tasks' % completed)

        
def timing_handle_results(outqueue, get, cache, beancounter, pool):
    '''
    A modified handle-results thread that tracks how much CPU time is
    used by workers.
    '''
    thread = threading.current_thread()
    while 1:
        try:
            task = get()
        except (IOError, EOFError):
            debug('result handler got EOFError/IOError -- exiting')
            return
        if thread._state:
            assert(thread._state == multiprocessing.pool.TERMINATE)
            debug('result handler found thread._state=TERMINATE')
            break
        if task is None:
            debug('result handler got sentinel')
            break
        #print 'Got task:', task
        (job, i, dt, obj) = task
        # ctrl-C -> (None, None, None, (False, KeyboardInterrupt()))
        if job is None:
            (success, val) = obj
            if not success:
                if isinstance(val, KeyboardInterrupt):
                    #print 'Terminating due to KeyboardInterrupt'
                    thread._state = multiprocessing.pool.TERMINATE
                    pool._state = multiprocessing.pool.CLOSE
                    break
        try:
            #print 'cache[job]:', cache[job], 'job', job, 'i', i
            cache[job]._set(i, obj)
        except KeyError:
            pass
        beancounter.add_time(dt)

    while cache and thread._state != multiprocessing.pool.TERMINATE:
        try:
            task = get()
        except (IOError, EOFError):
            debug('result handler got EOFError/IOError -- exiting')
            return

        if task is None:
            debug('result handler ignoring extra sentinel')
            continue
        (job, i, dt, obj) = task
        if job is None:
            #print 'Ignoring another KeyboardInterrupt'
            continue
        try:
            cache[job]._set(i, obj)
        except KeyError:
            pass
        beancounter.add_time(dt)

    if hasattr(outqueue, '_reader'):
        debug('ensuring that outqueue is not full')
        # If we don't make room available in outqueue then
        # attempts to add the sentinel (None) to outqueue may
        # block.  There is guaranteed to be no more than 2 sentinels.
        try:
            for i in range(10):
                if not outqueue._reader.poll():
                    break
                get()
        except (IOError, EOFError):
            pass
    debug('result handler exiting: len(cache)=%s, thread._state=%s',
          len(cache), thread._state)

    #print 'debug_handle_results finishing.'


def timing_handle_tasks(taskqueue, put, outqueue, progressqueue, pool,
                       maxnqueued):
    thread = threading.current_thread()
    if progressqueue is not None and hasattr(progressqueue, '_writer'):
        progressqueue._writer.close()
    
    nqueued = 0
    
    for taskseq, set_length in iter(taskqueue.get, None):
        i = -1
        #print 'handle_tasks: task sequence', taskseq
        for i, task in enumerate(taskseq):
            # print 'handle_tasks: got task', i
            if thread._state:
                debug('task handler found thread._state != RUN')
                break

            # print 'N queue:', nqueued, 'max', maxnqueued
            try:
                # print 'Queueing new task'
                put(task)
                nqueued += 1
            except IOError:
                debug('could not put task on queue')
                break

            # print 'N queue:', nqueued, 'max', maxnqueued
            if progressqueue is not None:
                while maxnqueued and nqueued >= maxnqueued:
                    try:
                        (job,i,pid) = progressqueue.get()
                        # print 'Job', job, 'element', i, 'pid', pid, 'started'
                        nqueued -= 1
                    except (IOError, EOFError):
                        break

        else:
            if set_length:
                debug('doing set_length()')
                set_length(i+1)
            continue
        break
    else:
        debug('task handler got sentinel')

    #print 'debug_handle_tasks got sentinel'

    try:
        # tell result handler to finish when cache is empty
        debug('task handler sending sentinel to result handler')
        outqueue.put(None)

        # tell workers there is no more work
        debug('task handler sending sentinel to workers')
        for p in pool:
            put(None)
    except IOError:
        debug('task handler got IOError when sending sentinels')

    # Empty the progressqueue to prevent blocking writing workers?
    if progressqueue is not None:
        # print 'task thread: emptying progressqueue'
        try:
            # print 'task thread: reading from progressqueue.  nqueued=', nqueued
            (job,i,pid) = progressqueue.get()
            # print 'Job', job, 'element', i, 'pid', pid, 'started'
            nqueued -= 1
        except (IOError,EOFError):
            pass
    # print 'Task thread done.'
    

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

class TimingPool(multiprocessing.pool.Pool):
    '''
    A python multiprocessing Pool subclass that keeps track of the
    resources used by workers, and tracks the expense of pickling
    objects.
    '''
    def _setup_queues(self):
        self._inqueue  = TimingSimpleQueue()
        self._outqueue = TimingSimpleQueue()
        self._quick_put = self._inqueue._writer.send
        self._quick_get = self._outqueue._reader.recv
        
    def get_pickle_traffic_string(self):
        S = self.get_pickle_traffic()
        return (('  pickled %i objs, %g MB, using %g s CPU\n' +
                 'unpickled %i objs, %g MB, using %g s CPU') %
                 (S[k] for k in [
                     'pickle_objs', 'pickle_megabytes', 'pickle_cputime',
                     'unpickle_objs', 'unpickle_megabytes', 'unpickle_cputime']))

    def get_pickle_traffic(self):
        S1 = self._inqueue.stats()
        S2 = self._outqueue.stats()
        return dict([(k, S1[k]+S2[k]) for k in S1.keys()])

    def get_worker_cpu(self):
        return self._beancounter.get_cpu()
    def get_worker_wall(self):
        return self._beancounter.get_wall()

    ### This just replaces the "worker" call with our "timing_worker".
    def _repopulate_pool(self):
        """Bring the number of pool processes up to the specified number,
        for use after reaping workers which have exited.
        """
        #print 'Repopulating pool with', (self._processes - len(self._pool)), 'workers'
        for i in range(self._processes - len(self._pool)):
            w = self.Process(target=timing_worker,
                             args=(self._inqueue, self._outqueue,
                                   self._progressqueue,
                                   self._initializer,
                                   self._initargs, self._maxtasksperchild)
                            )
            self._pool.append(w)
            w.name = w.name.replace('Process', 'PoolWorker')
            w.daemon = True
            w.start()
            debug('added worker')

    # This is just copied from the superclass; we call our routines:
    #  -handle_results -> timing_handle_results
    # And add _beancounter.
    def __init__(self, processes=None, initializer=None, initargs=(),
                 maxtasksperchild=None, taskqueuesize=0, context=None):
        '''
        taskqueuesize: maximum number of tasks to put on the queue;
          this is actually done by keeping a progressqueue, written-to
          by workers as they take work off the inqueue, and read by
          the handle_tasks thread.  (Can't use a limit on _taskqueue,
          because (a) multi-element tasks are written; and (b)
          taskqueue is between the caller and the handle_tasks thread,
          which then just transfers the work to the inqueue, where it
          piles up.  Can't easily use a limit on inqueue because it is
          implemented via pipes with unknown, OS-controlled capacity
          in units of bytes.)
        '''
        if context is None:
            # py3
            import multiprocessing.pool
            if 'get_context' in dir(multiprocessing.pool):
                context = multiprocessing.pool.get_context()
        self._ctx = context

        self._beancounter = BeanCounter()
        self._setup_queues()
        self._taskqueue = queue.Queue()
        self._cache = {}
        self._state = multiprocessing.pool.RUN
        self._initializer = initializer
        self._initargs = initargs
        self._maxtasksperchild = maxtasksperchild

        if taskqueuesize:
            self._progressqueue = TimingSimpleQueue()
        else:
            self._progressqueue = None
        
        if processes is None:
            try:
                processes = multiprocessing.cpu_count()
            except NotImplementedError:
                processes = 1

        if initializer is not None and not hasattr(initializer, '__call__'):
            raise TypeError('initializer must be a callable')

        self._processes = processes
        self._pool = []
        self._repopulate_pool()

        self._worker_handler = threading.Thread(
        target=multiprocessing.pool.Pool._handle_workers,
        args=(self, )
            )
        self._worker_handler.name = 'WorkerHandler'
        self._worker_handler.daemon = True
        self._worker_handler._state = multiprocessing.pool.RUN
        self._worker_handler.start()

        if True:
            self._task_handler = threading.Thread(
                target=timing_handle_tasks,
                args=(self._taskqueue, self._quick_put, self._outqueue,
                      self._progressqueue, self._pool,
                      taskqueuesize))
        else:
            self._task_handler = threading.Thread(
                target=multiprocessing.pool.Pool._handle_tasks,
                args=(self._taskqueue, self._quick_put, self._outqueue,
                      self._pool))
              
        self._task_handler.name = 'TaskHandler'
        self._task_handler.daemon = True
        self._task_handler._state = multiprocessing.pool.RUN
        self._task_handler.start()

        self._result_handler = threading.Thread(
            target=timing_handle_results,
            args=(self._outqueue, self._quick_get, self._cache,
                  self._beancounter, self)
            )
        self._result_handler.name = 'ResultHandler'
        self._result_handler.daemon = True
        self._result_handler._state = multiprocessing.pool.RUN
        self._result_handler.start()

        self._terminate = multiprocessing.util.Finalize(
            self, self._terminate_pool,
            args=(self._taskqueue, self._inqueue, self._outqueue, self._pool,
                  self._worker_handler, self._task_handler,
                  self._result_handler, self._cache),
            exitpriority=15
            )

if __name__ == '__main__':

    import sys
    from astrometry.util import multiproc
    from astrometry.util.ttime import *

    # import logging
    # lvl = logging.DEBUG
    # logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
    # import multiprocessing
    # multiprocessing.get_logger()
    
    def work(i):
        print('Doing work', i)
        time.sleep(2)
        print('Done work', i)
        return i
        
    def realwork(i):
        print('Doing work', i)
        import numpy as np
        X = 0
        for j in range(100 - 10*i):
            #print('work', i, j)
            X = X + np.random.normal(size=(1000,1000))
        print('Done work', i)
        return i
    
    class ywrapper(object):
        def __init__(self, y, n):
            self.n = n
            self.y = y
        def __str__(self):
            return 'ywrapper: n=%i; ' % self.n + self.y
        def __iter__(self):
            return self
        def next(self):
            return next(self.y)
        def __len__(self):
            return self.n
        __next__ = next
    def yielder(n):
        for i in range(n):
            print('Yielding', i)
            yield i

    N = 20
    y = yielder(N)
    args = ywrapper(y, N)
    
    dpool = TimingPool(4, taskqueuesize=4)
    dmup = multiproc.multiproc(pool=dpool)
    Time.add_measurement(TimingPoolMeas(dpool))

    # t0 = Time()
    # res = dmup.map(work, args)
    # print(Time()-t0)
    # print('Got result:', res)

    N = 20
    y = yielder(N)
    args = ywrapper(y, N)
    t0 = Time()
    print('Doing real work...')
    res = dmup.map(realwork, args)
    print(Time()-t0)
    print('Got result:', res)
    
    # Test taskqueuesize=1
    dpool = TimingPool(4, taskqueuesize=1)
    dmup = multiproc.multiproc(pool=dpool)
    Time.add_measurement(TimingPoolMeas(dpool))
    N = 20
    y = yielder(N)
    args = ywrapper(y, N)
    t0 = Time()
    print('Doing real work...')
    res = dmup.map(realwork, args)
    print(Time()-t0)
    print('Got result:', res)
