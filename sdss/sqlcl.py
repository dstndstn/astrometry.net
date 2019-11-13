#! /usr/bin/env python3
""">> sqlcl << command line query tool by Tamas Budavari <budavari@jhu.edu>
Usage: sqlcl [options] sqlfile(s)

Options:
        -s url       : URL with the ASP interface (default: pha)
        -f fmt       : set output format (html,xml,csv - default: csv)
        -q query   : specify query on the command line
        -l           : skip first line of output with column names
        -v       : verbose mode dumps settings in header
        -h       : show this message"""
        
from __future__ import print_function

formats = ['csv','xml','html']

astro_url='http://cas.sdss.org/astro/en/tools/search/x_sql.asp'
public_url='http://cas.sdss.org/public/en/tools/search/x_sql.asp'

default_url=public_url
default_fmt='csv'

def usage(status, msg=''):
    "Error message and usage"
    print(__doc__)
    if msg:
        print('-- ERROR: %s' % msg)
    sys.exit(status)

def filtercomment(sql):
    "Get rid of comments starting with --"
    import os
    fsql = ''
    for line in sql.split('\n'):
        fsql += line.split('--')[0] + ' ' + os.linesep;
    return fsql

def query(sql,url=default_url,fmt=default_fmt):
    "Run query and return file object"
    import urllib
    fsql = filtercomment(sql)
    params = urllib.urlencode({'cmd': fsql, 'format': fmt})
    return urllib.urlopen(url+'?%s' % params)     

def write_header(ofp,pre,url,qry):
    import    time
    ofp.write('%s SOURCE: %s\n' % (pre,url))
    ofp.write('%s TIME: %s\n' % (pre,time.asctime()))     
    ofp.write('%s QUERY:\n' % pre)
    for l in qry.split('\n'):
        ofp.write('%s    %s\n' % (pre,l))
    
def main(argv):
    "Parse command line and do it..."
    import os, getopt, string
    
    queries = []
    url = os.getenv("SQLCLURL",default_url)
    fmt = default_fmt
    writefirst = 1
    verbose = 0
    
    # Parse command line
    try:
        optlist, args = getopt.getopt(argv[1:],'s:f:q:vlh?')
    except getopt.error as e:
        usage(1,e)
        
    for o,a in optlist:
        if     o=='-s': url = a
        elif o=='-f': fmt = a
        elif o=='-q': queries.append(a)
        elif o=='-l': writefirst = 0
        elif o=='-v': verbose += 1
        else: usage(0)
        
    if fmt not in formats:
        usage(1,'Wrong format!')

    # Enqueue queries in files
    for fname in args:
        try:
            queries.append(open(fname).read())
        except IOError as e:
            usage(1,e)

    # Run all queries sequentially
    for qry in queries:
        ofp = sys.stdout
        if verbose:
            write_header(ofp,'#',url,qry)
        file = query(qry,url,fmt)
        # Output line by line (in case it's big)
        line = file.readline()
        if line.startswith("ERROR"): # SQL Statement Error -> stderr
            ofp = sys.stderr
        if writefirst:
            ofp.write(string.rstrip(line)+os.linesep)
        line = file.readline()
        while line:
            ofp.write(string.rstrip(line)+os.linesep)
            line = file.readline()


if __name__=='__main__':
    import sys
    main(sys.argv)






