# this file is called by ./configure
# it checks for mac OS systems if
# -arch i386 and/or MACOSX_DEPLOYMENT_TARGET have to be set
import os
import commands
import sys
from math import log

lmax = log(sys.maxsize, 2)
if lmax < 31:
    pbit = 32
else:
    pbit = 64
print 'Python is %i bits' %pbit

uname = commands.getoutput('uname')
print 'uname:', uname
if uname =='Darwin':
    vers = commands.getoutput('sw_vers -productVersion')
    deploy_tar = commands.getoutput('echo $MACOSX_DEPLOYMENT_TARGET')
    print 'OS X:', vers
    print '$MACOSX_DEPLOYMENT_TARGET:',deploy_tar 

    vers = vers.replace('.', '')
    if (float(vers) > 105) & (pbit ==32):
        print '\nNote that, your OS is Snow Leopard (or newer), but python is 32 bit'
        
# check compiler and linking flares to avoid architure problems
        cflags = commands.getoutput('echo $CFLAGS')
        ldflags = commands.getoutput('echo $LDFLAGS')
        print 'echo $CFLAGS: ',cflags
        print 'echo $LDFLAGS: ',ldflags

        ok=1
                
        if cflags.find('i386') < 0:
            flag ='export CFLAGS="$CFLAGS -arch i386"'
            print 'please consider setting:'
            print flag
            ok=0
            
        if ldflags.find('i386') < 0:
            flag ='export LDFLAGS="$LDFLAGS -arch i386"'
            print 'please consider setting:'
            print flag
            ok=0
 
        if float(deploy_tar.replace('.','')) < 106:
            print '\nto reduce warnings by python, consider setting:'
            print 'export MACOSX_DEPLOYMENT_TARGET=10.6'

if ok:
    print '\nOk.' 
