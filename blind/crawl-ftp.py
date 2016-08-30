# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
import os
import os.path
import re
import socket
import sys

from ftplib import FTP


class crawler(object):
    dirstack = [ '' ]
    currentdir = None

    ffiles = None
    fdirs = None
    fnomatches = None

    def __init__(self):
        self.ffiles = open('files', 'ab')
        self.fdirs = open('dirs', 'ab')
        self.fnomatches = open('nomatch', 'ab')

    ire = re.compile(r'^(?P<dir>.)' + # 'd' or '-'
                     r'.{9}' + # mode
                     r'\s*\w*' + # space, link count
                     r'\s*\w*' + # space, owner
                     r'\s*\w*' + # space, group
                     r'\s*(?P<size>\d*)' + # filesize
                     r'\s*(?P<month>\w*)' + # space, month
                     r'\s*(?P<day>\w*)' + # space, day
                     r'\s*(?P<time>[\w:]*)' + # space, time
                     r'\s*(?P<name>[\w.~_+-]*)' + # space, name
                     r'$')

    def close(self):
        self.ffiles.close()
        self.fdirs.close()
        self.fnomatches.close()

    def set_dirstack(self, stack):
        self.dirstack = stack

    def write_stack(self):
        f = open('dirstack.tmp', 'wb')
        for d in self.dirstack:
            f.write(d + '\n')
        f.close()
        os.rename('dirstack.tmp', 'dirstack')

    def add_item(self, s):
        print('item', s)
        m = self.ire.match(s)
        if not m:
            print('no match')
            self.fnomatches.write(self.currentdir + ' ' + s + '\n')
            self.fnomatches.flush()
            return

        d = m.group('dir')
        name = m.group('name')
        path = self.currentdir + '/' + name
        
        if d == 'd':
            self.dirstack.append(path)
            self.fdirs.write(path + '\n')
            self.fdirs.flush()
        else:
            self.ffiles.write(path + '\n')
            self.ffiles.flush()



if __name__ == '__main__':
    ftp = None

    crawl = crawler()

    if os.path.exists('dirstack'):
        f = open('dirstack', 'rb')
        stack = f.read().strip().split('\n')
        #stack = []
        #for ln in f:
        #    stack.append(ln)
        print('Dirstack:')
        for d in stack:
            print(d)
        print('(end dirstack)')
        crawl.set_dirstack(stack)

    nrequests = 0
    socket.setdefaulttimeout(10)

    while len(crawl.dirstack):
        if not ftp: # or not (nrequests % 100):
            if ftp:
                print('closing connection.')
                ftp.quit()
            print('opening connection')
            sys.stdout.flush()
            ftp = FTP('galex.stsci.edu')
            ftp.login('anonymous', 'dstn@cs.toronto.edu')
            #ftp.set_debuglevel(2)

        d = crawl.dirstack.pop()
        crawl.currentdir = d
        print('listing "%s"' % d)
        sys.stdout.flush()
        try:
            ftp.dir(d, crawl.add_item)
            crawl.write_stack()
            nrequests += 1
        except Exception as e:
            print('caught exception:', e)
            sys.stdout.flush()
            crawl.dirstack.append(d)
            ftp.close()
            ftp = None

