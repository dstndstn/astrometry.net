# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE

#import sqlite3
#import commands
from __future__ import print_function
import os
import sys

def backtick(cmd):
    (childin, childout) = os.popen2(cmd)
    childin.close()
    out = childout.read()
    return out

def getval(fn, key):
    cmd = "sqlite " + fn + " \"select val from jobdata where key like '" + key + "'\""
    out = backtick(cmd)
    return out

#fn = '/home/gmaps/ontheweb-data/alpha/200705/00001872/jobdata.db'
#email = getval(fn, 'email');
#status = getval(fn, 'checked-done')
#print email, status

outdir = "/data2/apod-solves/"

for month in sys.argv[1:]:
    #print month
    jobs = os.listdir(month)
    for job in jobs:
        path = month + '/' + job + '/'
        fn = path + "jobdata.db"
        #print fn
        email = getval(fn, 'email').strip()
        #print email
        if email != 'ckochanek@astronomy.ohio-state.edu':
            continue
        solvedfn = path + 'solved'
        if not os.path.exists(solvedfn):
            continue
        #status = getval(fn, 'checked-done')
        #print job, status
        #wcsfn = path + "wcs.fits"
        #print wcsfn
        #print job
        print(path)
        cmd = "pngtopnm " + path + "fullsize.png | pnmtojpeg > " + outdir + "csk-" + job + ".jpg"
        backtick(cmd)
        cmd = "cp " + path + "wcs.fits " + outdir + "csk-" + job + ".wcs"
        backtick(cmd)


#conn = sqlite3.connect(fn)
#cur = conn.cursor()
#cur.execute("select val from jobdata where key like 'user-email'")

