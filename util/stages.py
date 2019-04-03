# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
import os
from astrometry.util.file import *

'''

Stages: a utility for saving and resuming computation, savings
intermediate results as pickle files.


'''

class CallGlobal(object):
    def __init__(self, pattern, globals, *args, **kwargs):
        self.pat = pattern
        self.args = args
        self.kwargs = kwargs
        self.globals = globals
    def getfunc(self, stage):
        func = self.pat % stage
        func = eval(func, self.globals)
        return func
    def getkwargs(self, stage, **kwargs):
        kwa = self.kwargs.copy()
        kwa.update(kwargs)
        return kwa
    def __call__(self, stage, **kwargs):
        func = self.getfunc(stage)
        kwa = self.getkwargs(stage, **kwargs)
        return func(*self.args, **kwa)

class CallGlobalTime(CallGlobal):
    def __call__(self, stage, **kwargs):
        from astrometry.util.ttime import Time
        from datetime import datetime
        t0 = Time()
        print('Running stage', stage, 'at', datetime.now().isoformat())
        rtn = super(CallGlobalTime, self).__call__(stage, **kwargs)
        t1 = Time()
        print('Stage', stage, ':', t1-t0)
        print('Stage', stage, 'finished:', datetime.now().isoformat())
        return rtn

def runstage(stage, picklepat, stagefunc, force=[], forceall=False, prereqs={},
             update=True, write=True, initial_args={}, **kwargs):
    '''
    Run to a given *stage*.

    Each stage is a function.
    
    Each stage takes a dict and returns None or a dict.

    Each stage (except the first one!) has a prerequisite stage.

    Results after running each stage can be written to a pickle file,
    and later read instead of running the stage.

    Parameters
    ----------
    stage - int or string
        Stage name
    picklepat - string
        Filename pattern to which pickle data should be written
    stagefunc - function
        Function to call for each stage.
    force - list of stage name/number
        List of stages to run, ignoring existing pickle files
    forceall - boolean
        Force running all stages; ignore all existing pickle files
    prereqs - dict of stage->stage mappings
        Defines the dependencies between stages
    update - boolean
        Update prerequiste dict with results before writing out pickle
    write - boolean, or list of stages
        Write a pickle for this stage / all stages?
    initial_args - dict
        Arguments to pass to the first stage
    kwargs - dict
        Keyword args added to the prerequisites before running each stage
    '''
    # NOTE, if you add or change args here, be sure to update the recursive
    # "runstage" call below!!
    print('Runstage', stage)

    try:
        pfn = picklepat % stage
    except:
        pfn = picklepat % dict(stage=stage)
    
    if os.path.exists(pfn):
        if forceall or stage in force:
            print('Ignoring pickle', pfn, 'and forcing stage', stage)
        else:
            print('Reading pickle', pfn)
            try:
                R = unpickle_from_file(pfn)
                return R
            except Exception as e:
                print('Failed to read pickle file', pfn, ':', e, '; re-running stage')

    try:
        prereq = prereqs[stage]
    except KeyError:
        prereq = stage - 1

    if prereq is None:
        P = initial_args
    else:
        P = runstage(prereq, picklepat, stagefunc,
                     force=force, forceall=forceall, prereqs=prereqs, update=update,
                     write=write, initial_args=initial_args, **kwargs)

    #P.update(kwargs)
    Px = P.copy()
    Px.update(kwargs)

    print('Running stage', stage)
    # print('Prereq keys:', P.keys())
    # print('Adding kwargs keys:', kwargs.keys())
    # print('Combined keys:', Px.keys())

    R = stagefunc(stage, **Px)
    print('Stage', stage, 'finished')
    # if R is not None:
    #     print('Result keys:', R.keys())

    if update:
        if R is not None:
            P.update(R)
        R = P

    if not write:
        pass
    elif (write is True or stage in write):
        print('Saving pickle', pfn)
        #print('Pickling keys:', R.keys())
        # Create directory, if necessary.
        dirnm = os.path.dirname(pfn)
        if len(dirnm) and not os.path.exists(dirnm):
            try:
                os.makedirs(dirnm)
            except:
                pass
        tempfn = os.path.join(dirnm, 'tmp-' + os.path.basename(pfn))
        pickle_to_file(R, tempfn)
        os.rename(tempfn, pfn)
        print('Saved', pfn)
    return R
