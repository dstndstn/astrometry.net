#! /usr/bin/env python3
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE

import sys

import http.client
http.client.HTTPConnection.debuglevel = 1 

import urllib.request, urllib.parse, urllib.error
import urllib.request, urllib.error, urllib.parse
#from xml.parsers import expat
import xml.dom
from xml.dom import minidom
import pickle
import os
import os.path
import re
import time

from numpy import *

from astrometry.sdss.sqlcl import query as casquery
from astrometry.util.file import *

class Cas(object):
    def __init__(self, **kwargs):
        self.params = kwargs

    def get_url(self, relurl):
        return self.params['base_url'] + relurl

    def login_url(self):
        return self.get_url('login.aspx')

    def submit_url(self):
        return self.get_url(self.params['submiturl'])

    def mydb_url(self):
        return self.get_url('MyDB.aspx')

    def mydb_index_url(self):
        return self.get_url('mydbindex.aspx')

    def mydb_action_url(self):
        return self.get_url(self.params['actionurl'])

    def drop_url(self):
        return self.get_url('DropTable.aspx?tableName=%s')

    def output_url(self):
        return self.get_url('Output.aspx')

    def job_details_url(self):
        return self.get_url('jobdetails.aspx?id=%i')

    def cancel_url(self):
        return self.get_url('cancelJob.aspx')

    def submit_query(self, sql, table='', taskname='', dbcontext=None):
        if dbcontext is None:
            dbcontext = self.params['defaultdb']

        # MAGIC
        data = urllib.parse.urlencode({
            'targest': dbcontext,
            'sql': sql,
            'queue': 500,
            'syntax': 'false',
            'table': table,
            'taskname': taskname,
            })
        f = urllib.request.urlopen(self.submit_url(), data)
        doc = f.read()
        redirurl = f.geturl()
        # older CasJobs version redirects to the job details page:
        # just pull the jobid out of the redirected URL.
        print('Redirected to URL', redirurl)
        pat = re.escape(self.job_details_url().replace('%i','')) +  '([0-9]*)'
        #print 'pattern:', pat
        m = re.match(pat, redirurl)
        if m is not None:
            jobid = int(m.group(1))
            print('jobid:', jobid)
            return jobid
        #write_file(doc, 'sub.out')
        xmldoc = minidom.parseString(doc)
        jobids = xmldoc.getElementsByTagName('jobid')
        if len(jobids) == 0:
            print('No <jobid> tag found:')
            print(doc)
            return None
        if len(jobids) > 1:
            print('Multiple <jobid> tags found:')
            print(doc)
            return None
        jobid = jobids[0]
        if not jobid.hasChildNodes():
            print('<jobid> tag has no child node:')
            print(doc)
            return None
        jobid = jobid.firstChild
        if jobid.nodeType != xml.dom.Node.TEXT_NODE:
            print('job id is not a text node:')
            print(doc)
            return None
        jobid = int(jobid.data)
        if jobid == -1:
            # Error: find error message.
            print('Failed to submit query.  Looking for error message...')
            founderr = False
            msgs = xmldoc.getElementsByTagName('message')
            for msg in msgs:
                if msg.hasChildNodes():
                    c = msg.firstChild
                    if c.nodeType == xml.dom.Node.TEXT_NODE:
                        print('Error message:', c.data)
                        founderr = True
            if not founderr:
                print('Error message not found.  Whole response document:')
                print()
                print(doc)
                print()
        return jobid
    
    def login(self, username, password):
        print('Logging in.')
        data = urllib.parse.urlencode({'userid': username, 'password': password})
        f = urllib.request.urlopen(self.login_url(), data)
        d = f.read()
        # headers = f.info()
        # print 'headers:', headers
        # print 'Got response:'
        # print d
        return None

    def cancel_job(self, jobid):
        data = urllib.parse.urlencode({'id': jobid, 'CancelJob': 'Cancel Job'})
        f = urllib.request.urlopen(self.cancel_url(), data)
        f.read()

    # Returns 'Finished', 'Ready', 'Started', 'Failed', 'Cancelled'
    def get_job_status(self, jobid):
        # print 'Getting job status for', jobid
        url = self.job_details_url() % jobid
        print('Job details URL:', url)
        doc = urllib.request.urlopen(url).read()
        for line in doc.split('\n'):
            for stat in ['Finished', 'Ready', 'Started', 'Failed', 'Cancelled']:
                if ('<td > <p class = "%s">%s</p></td>' %(stat,stat) in line or
                    #'<td > <p class="%s">%s</p></td>' %(stat,stat) in line or
                    '<p class="%s">%s</p>' % (stat,stat) in line or
                    '<td class="center"> <p class = "%s">%s</p></td>' %(stat,stat) in line or
                    '<td class="center"> <p class = "%s">Running</p></td>' %(stat) in line): # Galex "Started"/Running
                    return stat
        return None

    def drop_table(self, dbname):
        url = self.drop_url() % dbname
        try:
            f = urllib.request.urlopen(url)
        except Exception as e:
            print('Failed to drop table', dbname)
            print(e)
            return False
        doc = f.read()
        (vs,ev) = get_viewstate_and_eventvalidation(doc)
        data = urllib.parse.urlencode({'yesButton':'Yes',
                                 '__EVENTVALIDATION':ev,
                                 '__VIEWSTATE':vs})
        print('Dropping table', dbname)
        try:
            f = urllib.request.urlopen(url, data)
        except Exception as e:
            print('Failed to drop table', dbname)
            print(e)
            return False
        d = f.read()
        # print 'Got response:'
        # print d
        # write_file(d, 'res.html')
        return True

    def request_output(self, mydbname):
        url = self.mydb_action_url() % mydbname
        try:
            # Need to prime the VIEWSTATE by "clicking" through...
            f = urllib.request.urlopen(self.mydb_url())
            f.read()
            f = urllib.request.urlopen(self.mydb_index_url())
            f.read()
            # request = urllib2.Request(url)
            # request.add_header('User-Agent', 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.11) Gecko/2009060214 Firefox/3.0.11')
            # f = urllib2.urlopen(request)
            # Referer: http://galex.stsci.edu/casjobs/mydbindex.aspx
            f = urllib.request.urlopen(url)
        except urllib.error.HTTPError as e:
            print('HTTPError:', e)
            print('  code', e.code)
            print('  msg', e.msg)
            print('  hdrs', e.hdrs)
            print('  data:', e.fp.read())
            raise e
        doc = f.read().strip()
        # write_file(doc, 'r1.html')
        (vs,ev) = get_viewstate_and_eventvalidation(doc)
        data = {'extractDDL':'FITS', 'Button1':'Go',
                '__VIEWSTATE':vs}
        if ev is not None:
            data['__EVENTVALIDATION'] = ev
        extra = self.params.get('request_output_extra')
        if extra is not None:
            data.update(extra)
            print('requesting FITS output of MyDB table', mydbname)
        print('url', url)
        print('data', urllib.parse.urlencode(data))
        try:
            f = urllib.request.urlopen(url, urllib.parse.urlencode(data))
        except urllib.error.HTTPError as e:
            print('HTTPError:', e)
            print('  code', e.code)
            # print '  reason', e.reason
            raise e
        d = f.read()
        # print 'Got response:'
        # print d
        # write_file(d, 'res.html')
        return

    def get_ready_outputs(self):
        url = self.output_url()
        print('Hitting output URL', url)
        f = urllib.request.urlopen(url)
        doc = f.read()
        write_file(doc, 'ready.html')
        print('Wrote ready downloads to ready.html')
        urls = []
        fns = []

        # split into successful and failed parts
        ifailed = doc.index('Failed Output:')
        if ifailed > -1:
            failed = doc[ifailed:]
            doc = doc[:ifailed]
        else:
            failed = ''
        
        rex = re.compile('<a href="(' + re.escape(self.params['outputbaseurl']) + '(.*))">Download</a>')
        for line in doc.split('\n'):
            m = rex.search(line)
            if not m:
                continue
            urls.append(m.group(1))
            fns.append(m.group(2))

        
        fails = []
        rex = re.compile(r'^\s*<td>(.*?)</td><td>.*?</td><td>(.*?)</td><td>(.*?)</td>', flags=re.MULTILINE | re.DOTALL)
        for m in rex.finditer(failed):
            tab = m.group(1)
            if '<br>' in tab:
                continue
            msg = m.group(3).replace('\r', ' ').replace('\n', ' ')
            fails.append((tab, m.group(2), msg))
            # ( table name, date string, failure message )
        return (urls, fns, fails)

    def sql_to_fits(self, sql, outfn, dbcontext=None, sleeptime=10):
        import random
        dbname = ''.join(chr(ord('A') + random.randrange(26))
                         for x in range(10))
        if sql.lower().startswith('select '):
            sql = 'select into mydb.%s' % dbname + sql[6:]
        print('Submitting query: "%s"' % sql)
        if dbcontext is not None:
            kwargs = dict(dbcontext=dbcontext)
        else:
            kwargs = {}
        jid = self.submit_query(sql, **kwargs)
        print('Submitted job id', jid)
        print('Waiting for job id:', jid)
        while True:
            jobstatus = self.get_job_status(jid)
            print('Job id', jid, 'is', jobstatus)
            if jobstatus in ['Finished', 'Failed', 'Cancelled']:
                break
            print('Sleeping...')
            time.sleep(sleeptime)
        print('Output-downloads-delete')
        dodelete = True
        self.output_and_download([dbname], [outfn], dodelete)
    
    # Requests output of the given list of databases, waits for them to appear,
    # downloads them, and writes them to the given list of local filenames.
    #
    # 'dbs' and 'fns' must be either strings,
    #   or lists of string of the same length.
    #
    # If 'dodelete' is True, the databases will be deleted after download.
    #
    def output_and_download(self, dbs, fns, dodelete=False, sleeptime=10,
                            raiseonfail=True):
        if type(dbs) is str:
            dbs = [dbs]
            assert(type(fns) is str)
            fns = [fns]
        # copy lists
        dbs = dbs[:]
        fns = fns[:]
            
        print('Getting list of available downloads...')
        (preurls,nil,prefails) = self.get_ready_outputs()
        #print 'Preurls:', preurls
        #print 'Prefails:', prefails
        for db in dbs:
            print('Requesting output of', db)
            self.request_output(db)
        while True:
            print('Waiting for output to appear...')
            (durls,dfns,dfails) = self.get_ready_outputs()
            (newurls, newfns) = find_new_outputs(durls, dfns, preurls)
            #print 'URLs:', durls
            #print 'New URLs:', newurls
            print('New outputs available:', newfns)
            newfails = [f for f in dfails if not f in prefails]
            print('New failures:', newfails)
            for (fn,db) in zip(fns,dbs):
                for (dfn,durl) in zip(newfns,newurls):
                    # the filename will contain the db name.
                    if not db in dfn:
                        continue
                    print('Output', dfn, 'looks like it belongs to database', db)
                    print('Downloading to local file', fn)
                    cmd = 'wget -O "%s" "%s"' % (fn, durl)
                    print('  (running: "%s")' % cmd)
                    w = os.system(cmd)
                    if not os.WIFEXITED(w) or os.WEXITSTATUS(w):
                        print('download failed.')
                        return -1
                    dbs.remove(db)
                    fns.remove(fn)
                    if dodelete:
                        print('Deleting database', db)
                        self.drop_table(db)
            for (fn,db) in zip(fns,dbs):
                for tab,date,msg in newfails:
                    if db != tab:
                        continue
                    print('Failure:', tab, 'date', date, 'message', msg)
                    print('looks like it belongs to database', db)
                    dbs.remove(db)
                    fns.remove(fn)
                    if raiseonfail:
                        raise RuntimeError('Error from CasJobs on output of table "%s": "%s"' % (db, msg))
                    
            if not len(dbs):
                   break
            print('Waiting...')
            time.sleep(sleeptime)
        return 0


def get_viewstate_and_eventvalidation(doc):
    rex = re.compile('<input type="hidden" name="__VIEWSTATE" (?:id="__VIEWSTATE" )?value="(.*)" />')
    vs = None
    for line in doc.split('\n'):
        m = rex.search(line)
        if not m:
            continue
        vs = m.group(1)
        break
    rex = re.compile('<input type="hidden" name="__EVENTVALIDATION" id="__EVENTVALIDATION" value="(.*)" />')
    ev = None
    for line in doc.split('\n'):
        m = rex.search(line)
        if not m:
            continue
        ev = m.group(1)
        break
    return (vs,ev)

def query(sql):
    f = casquery(sql)
    header = f.readline().strip()
    if header.startswith('ERROR'):
        raise RuntimeError('SQL error: ' + f.read())
    cols = header.split(',')
    results = []
    for line in f:
        words = line.strip().split(',')
        row = []
        for w in words:
            try:
                ival = int(w)
                row.append(ival)
                continue
            except ValueError:
                pass
            try:
                fval = float(w)
                row.append(fval)
                continue
            except ValueError:
                pass
            row.append(w)
            results.append(row)
    return (cols, results)

def setup_cookies():
    cookie_handler = urllib.request.HTTPCookieProcessor()
    opener = urllib.request.build_opener(cookie_handler)
    # ...and install it globally so it can be used with urlopen.
    urllib.request.install_opener(opener)

def find_new_outputs(durls, dfns, preurls):
    newurls = [u for u in durls if not u in preurls]
    newfns =  [f for (f,u) in zip(dfns,durls) if not u in preurls]
    return (newurls, newfns)

def get_known_servers():
    return     {
        'galex': Cas(
            base_url='http://galex.stsci.edu/casjobs/',
            submiturl='SubmitJob.aspx',
            actionurl='mydbcontent.aspx?tableName=%s&kind=tables',
            defaultdb='GALEXGR4Plus5',
            outputbaseurl='http://mastweb.stsci.edu/CasOutPut/FITS/'
            ),
        # SDSS
        'dr7': Cas(
            base_url='http://casjobs.sdss.org/CasJobs/',
            submiturl='submitjobhelper.aspx',
            actionurl='mydbcontent.aspx?ObjName=%s&ObjType=TABLE&context=MyDB&type=normal',
            defaultdb='DR7',
            request_output_extra={ 'targetDDL':'TargDR7Long' },
            outputbaseurl='http://casjobs.sdss.org/CasJobsOutput2/FITS/'
            ),
        # SDSS
        'dr8': Cas(
            # These will change...
            base_url='http://skyserver.sdss3.org/casjobs/',
# base_url='http://skyservice.pha.jhu.edu/casjobs/',
            submiturl='submitjobhelper.aspx',
            actionurl='mydbcontent.aspx?ObjName=%s&ObjType=TABLE&context=MyDB&type=normal',
            defaultdb='DR8',
            #request_output_extra={ 'targetDDL':'Thumper_DR7' },
            outputbaseurl='http://skyservice.pha.jhu.edu/CasJobsOutput/FITS/'
            ),
        'dr9': Cas(
            # These will change...
            base_url='http://skyserver.sdss3.org/casjobs/',
            submiturl='submitjobhelper.aspx',
            actionurl='mydbcontent.aspx?ObjName=%s&ObjType=TABLE&context=MyDB&type=normal',
            defaultdb='DR9',
            outputbaseurl='http://skyservice.pha.jhu.edu/CasJobsOutput/FITS/'
            ),
            }


if __name__ == '__main__':
    from optparse import OptionParser

    surveys = get_known_servers()

    parser = OptionParser(usage=('%prog <options> <args>'))
    parser.add_option('-s', '--survey', dest='survey', default='dr7',
                      help=('Set the CasJobs instance to use: one of: ' +
                            ', '.join(list(surveys.keys()))))
    parser.add_option('-c', '--context', '--db', dest='dbcontext',
                      help='Database context ("DR7", "Stripe82", etc)')
    opt,args = parser.parse_args()
    cas = surveys[opt.survey]

    if len(args) < 2:
        print('%s <username> <password> [command <args>]' % sys.argv[0])
        print()
        print('commands include:')
        print('   sqltofits (<sql> or <@sql-file>) output.fits')
        print('   delete <database> [...]')
        print('   query  ( <sql> or <@sqlfile> ) [...]')
        print('   querywait  ( <sql> or <@sqlfile> ) [...]')
        print('     -> submit query and wait for it to finish')
        print('   output <database> [...]')
        print('     -> request that a database be output as a FITS table')
        print('   outputdownload <database> <filename> [...]')
        print('     -> request output, wait for it to finish, and download to <filename>')
        print('   outputdownloaddelete <database> <filename> [...]')
        print('     -> request output, wait for it to finish, download to <filename>, and drop table.')
        sys.exit(-1)

    instance = 4

    username = args[0]
    password = args[1]
    cmd = None
    if len(args) > 2:
        cmd = args[2]

    setup_cookies()

    cas.login(username, password)

    if cmd == 'delete':
        if len(args) < 4:
            print('Usage: ... delete <db>')
            sys.exit(-1)
        db = args[3]
        print('Dropping', db)
        cas.drop_table(db)
        sys.exit(0)

    if cmd in ['query', 'querywait']:
        qs = args[3:]
        if len(qs) == 0:
            print('Usage: ... query <sql> or <@file> [...]')
            sys.exit(-1)
        jids = []
        for q in qs:
            if q.startswith('@'):
                q = read_file(q[1:])
            print('Submitting query: "%s"' % q)
            jobid = cas.submit_query(q, dbcontext=opt.dbcontext)
            print('Submitted job id', jobid)
            jids.append(jobid)
        if cmd in ['querywait']:
            # wait for them to finish.
            while True:
                print('Waiting for job ids:', jids)
                for jid in jids:
                    jobstatus = cas.get_job_status(jid)
                    print('Job id', jid, 'is', jobstatus)
                    if jobstatus in ['Finished', 'Failed', 'Cancelled']:
                        jids.remove(jid)
                if not len(jids):
                    break
                print('Sleeping...')
                time.sleep(10)
        sys.exit(0)

    if cmd == 'sqltofits':
        if len(args) != 5:
            print('Usage: ... sqltofits [<sql> or <@file>] output.fits')
            sys.exit(-1)
        q = args[3]
        outfn = args[4]
        if q.startswith('@'):
            q = read_file(q[1:])
        cas.sql_to_fits(q, outfn, dbcontext=opt.dbcontext)
        sys.exit(0)
            
    if cmd == 'output':
        dbs = args[3:]
        if len(dbs) == 0:
            print('Usage: ... output <db> [<db> ...]')
            sys.exit(-1)
        for db in dbs:
            print('Requesting output of db', db)
            cas.request_output(db)
        sys.exit(0)

    if cmd in ['outputdownload', 'outputdownloaddelete']:
        dodelete = (cmd == 'outputdownloaddelete')
        dbfns = args[3:]
        if len(dbfns) == 0 or len(dbfns) % 2 == 1:
            print('Usage: ... outputdownload <db> <filename> [...]')
            sys.exit(-1)
        dbs = dbfns[0::2]
        fns = dbfns[1::2]

        cas.output_and_download(dbs, fns, dodelete)
        sys.exit(0)

    print('Unrecognized command')
    
