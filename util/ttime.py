import os
import resource

def get_memusage():
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

    procfn = '/proc/%d/status' % os.getpid()
    try:
        t = open(procfn).readlines()
        d = dict([(line.split()[0][:-1], line.split()[1:]) for line in t])
        mu.update(d)
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
    print 'Memory usage:'
    print 'max rss:', mu['maxrss'], 'MB'
    for key in ['VmPeak', 'VmSize', 'VmRSS', 'VmData', 'VmStk'
                # VmLck, VmHWM, VmExe, VmLib, VmPTE
                ]:
        print key, ' '.join(mu.get(key, []))

class MemMeas(object):
    def __init__(self):
        self.mem0 = get_memusage()
    def format_diff(self, other):
        #keys = self.mem0.keys()
        #keys.sort()
        txt = []
        #for k in keys:
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
    def format_diff(self, other):
        dwall = (self.wall - other.wall)
        # python2.7
        if hasattr(dwall, 'total_seconds'):
            dwall = dwall.total_seconds()
        else:
            dwall = (dwall.microseconds + (dwall.seconds + dwall.days * 24. * 3600.) * 1e6) / 1e6
        dcpu = (self.cpu - other.cpu)
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

