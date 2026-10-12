#!/usr/bin/env python3
"""Requires a normal build of libastrometry, util archives and CLI tools.
SANITIZE=1 instruments all changed library modules, plus this test driver.
BASELINE=1 instruments original fitting modules for the reported overflow.
"""
import os,pathlib,subprocess,tempfile,shlex
R=pathlib.Path(__file__).resolve().parents[2]
def run(a,**kw):return subprocess.run(list(map(str,a)),cwd=R,check=True,**kw)
cc=shlex.split(os.getenv('CC','cc'));flags=['-O1','-g','-DNDEBUG','-DAN_GIT_URL="sip-order-test"','-DAN_GIT_REVISION="test"','-DAN_GIT_DATE="test"']
if os.getenv('SANITIZE') or os.getenv('BASELINE'):flags+=['-fsanitize=address,undefined','-fno-omit-frame-pointer']
inc=['-I'+str(R/p) for p in ['include','include/astrometry','gsl-an','util','solver']]
libs=[R/p for p in ['solver/libastrometry.a','catalogs/libcatalogs.a','util/libanfiles.a','libkd/libkd.a','util/libanutils.a','qfits-an/libqfits.a','util/libanbase.a','gsl-an/libgsl-an.a']]+shlex.split(subprocess.check_output(['pkg-config','--libs','wcslib'],text=True))+['-lm','-pthread']
with tempfile.TemporaryDirectory(prefix='sip-orders-') as d:
 t=pathlib.Path(d);objects=[]
 for file in ['util/fit-wcs.c','util/sip-utils.c','util/wcs-pv2sip.c','solver/tweak.c','solver/tweak2.c']:
  source=R/file
  if os.getenv('BASELINE'):
   source=t/pathlib.Path(file).name;source.write_bytes(subprocess.check_output(['git','show','d20a0503:'+file],cwd=R))
  obj=t/(pathlib.Path(file).stem+'.o');run(cc+flags+inc+['-c',source,'-o',obj]);objects.append(obj)
 exe=t/'test';run(cc+flags+inc+[R/'tests/sip-orders/test.c']+objects+libs+['-o',exe])
 if os.getenv('BASELINE'):run([exe,'reproduce']);raise SystemExit
 run([exe]);run([exe,'reproduce']);corr=t/'test.corr';run([exe,'fixture',corr])
 # keirthing is not a named Makefile target; link explicitly.
 keir=t/'keirthing';run(cc+flags+inc+[R/'solver/keirthing.c']+objects+libs+['-o',keir])
 for source,name in [('util/fit-wcs-main.c','fit-wcs'),('util/wcs-pv2sip-main.c','wcs-pv2sip')]:
  execli=t/name;run(cc+flags+inc+[R/source]+objects+libs+['-o',execli])
  for order in ['-1','10','12','2147483647','9999999999999999999999999999','abc','9x']:
   args=[execli,'-s',order,'-c',corr,'-o',t/'out.wcs'] if name=='fit-wcs' else [execli,'-o',order,'missing',t/'out.wcs']
   p=subprocess.run(list(map(str,args)),capture_output=True,text=True);assert p.returncode==1 and 'SIP order must be an integer' in p.stderr,(args,p.returncode,p.stderr)
  if name=='fit-wcs':
   for order in [0,1,8,9]:run([execli,'-s',order,'-c',corr,'-o',t/(str(order)+'.wcs')])
   text=(t/'9.wcs').read_bytes().decode('ascii');assert 'AP_ORDER' in text and 'BP_ORDER' in text
 for order in ['-1','10','12','2147483647','abc']:
  p=subprocess.run([str(keir),'-o',order],capture_output=True,text=True);assert p.returncode==1 and 'SIP order must be an integer' in p.stderr
 print('PASS CLI malformed/negative/oversized orders; actual fit-wcs orders 0,1,8,9; forward-9 inverse capped at 9')
