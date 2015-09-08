# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function

import os
import re
import resource

# split out of get_memusage for profiling purposes
def _read_proc_status(pid):
    procfn = '/proc/%d/status' % pid
    t = open(procfn).readlines()
    d = dict([(line.split()[0][:-1], line.split()[1:]) for line in t])
    return d

                 
def _read_proc_maps(pid):
    procfn = '/proc/%d/maps' % pid
    t = open(procfn).readlines()
    d = dict(mmaps=t)
    rex = re.compile(r'(?P<addrlo>[0-9a-f]+)-(?P<addrhi>[0-9a-f]+) .*')
    parsed = []
    addrsum = 0
    for line in t:
        m = rex.match(line)
        if m is not None:
            parsed.append(m.groupdict())
            try:
                addrsum += int(m.group('addrhi'), 16) - int(m.group('addrlo'), 16)
            except:
                pass
    return dict(mmaps=t, mmaps_parsed=parsed, mmaps_total=addrsum)

def get_memusage(mmaps=True):
    ru = resource.getrusage(resource.RUSAGE_SELF)
    pgsize = resource.getpagesize()
    maxrss = (ru.ru_maxrss * pgsize / 1e6)
    #print 'shared memory size:', (ru.ru_ixrss / 1e6), 'MB'
    #print 'unshared memory size:', (ru.ru_idrss / 1e6), 'MB'
    #print 'unshared stack size:', (ru.ru_isrss / 1e6), 'MB'
    #print 'shared memory size:', ru.ru_ixrss
    #print 'unshared memory size:', ru.ru_idrss
    #print 'unshared stack size:', ru.ru_isrss
    mu = dict(maxrss=[maxrss, 'MB'])

    pid = os.getpid()
    try:
        mu.update(_read_proc_status(pid))
    except:
        pass

    # /proc/meminfo ?

    if mmaps:
        try:
            mu.update(_read_proc_maps(pid))
        except:
            pass
    
    return mu

def get_procio():
    procfn = '/proc/%d/io' % os.getpid()
    dd = dict()
    try:
        t = open(procfn).readlines()
        for line in t:
            words = line.split()
            key = words[0].strip().strip(':')
            val = int(words[1])
            dd[key] = val
    except:
        pass
    return dd
    
def memusage():
    mu = get_memusage()
    print('Memory usage:')
    print('max rss:', mu['maxrss'], 'MB')
    for key in ['VmPeak', 'VmSize', 'VmRSS', 'VmData', 'VmStk'
                # VmLck, VmHWM, VmExe, VmLib, VmPTE
                ]:
        print(key, ' '.join(mu.get(key, [])))
    if 'mmaps' in mu:
        print('Number of mmaps:', len(mu['mmaps']))

def count_file_descriptors():
    procfn = '/proc/%d/fd' % os.getpid()
    try:
        # Linux: /proc/PID/fd/*
        t = os.listdir(procfn)
        return len(t)
    except:
        pass

    try:
        # OSX: "lsof"
        from run_command import run_command as rc
        cmd = 'lsof -p %i' % os.getpid()
        rtn,out,err = rc(cmd)
        if rtn == 0:
            return len(out.split('\n'))
    except:
        pass
    return 0

class FileDescriptorMeas(object):
    def __init__(self):
        self.fds = count_file_descriptors()
    def format_diff(self, other):
        return 'Open files: %i' % self.fds
            
class MemMeas(object):
    def __init__(self):
        self.mem0 = get_memusage(mmaps=False)
    def format_diff(self, other):
        txt = []
        for k in ['VmPeak', 'VmSize', 'VmRSS', 'VmData']:
            if not k in self.mem0:
                continue
            val,unit = self.mem0[k]
            if unit == 'kB':
                val = int(val, 10)
                val /= 1024.
                unit = 'MB'
                val = '%.0f' % val
            txt.append('%s: %s %s' % (k, val, unit))
        return ', '.join([] + txt)

class IoMeas(object):
    def __init__(self):
        self.io0 = get_procio()
    def format_diff(self, other):
        txt = []
        d1 = self.io0
        d0 = other.io0
        for k,knice in [('rchar',None), ('wchar',None),
                        ('read_bytes','rb'), ('write_bytes', 'wb')]:
            v1 = d1.get(k)
            v0 = d0.get(k)
            unit = 'b'
            dv = float(v1 - v0)
            for uu in ['kB', 'MB', 'GB']:
                if dv < 2048:
                    break
                dv = dv / 1024.
                unit = uu
            val = '%.3g' % dv
            if knice is None:
                kk = k
            else:
                kk = knice
            txt.append('%s: %s %s' % (kk, val, unit))
        return ', '.join([] + txt)

class CpuMeas(object):
    def __init__(self):
        import datetime
        from time import clock
        self.wall = datetime.datetime.now()
        self.cpu = clock()

    def cpu_seconds_since(self, other):
        return self.cpu - other.cpu
    def wall_seconds_since(self, other):
        dwall = (self.wall - other.wall)
        # python2.7
        if hasattr(dwall, 'total_seconds'):
            dwall = dwall.total_seconds()
        else:
            dwall = (dwall.microseconds + (dwall.seconds + dwall.days * 24. * 3600.) * 1e6) / 1e6
        return dwall
        
    def format_diff(self, other):
        dwall = self.wall_seconds_since(other)
        dcpu = self.cpu_seconds_since(other)
        return 'Wall: %.2f s, CPU: %.2f s' % (dwall, dcpu)
        
class Time(object):
    @staticmethod
    def add_measurement(m):
        Time.measurements.append(m)
    measurements = [CpuMeas]

    def __init__(self):
        self.meas = [m() for m in Time.measurements]

    def __sub__(self, other):
        '''
        Returns a string representation of the difference: self - other
        '''
        meas = ', '.join([m.format_diff(om) for m,om in zip(self.meas, other.meas)])
        return meas

