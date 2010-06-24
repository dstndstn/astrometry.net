#! /usr/bin/env python

import matplotlib
matplotlib.use('Agg')

from astrometry.util.pyfits_utils import *
from astrometry.util.file import *

from pylab import *
from optparse import *
import os
from glob import glob
from numpy import *

def get_field(fieldname, m1, m2, nil, preproc=None):
	t1 = []
	t2 = []
	for k,v in m1.items():
		tt1 = v.getcolumn(fieldname)
		if preproc is not None:
			#print 'tt1=', tt1
			tt1 = preproc(tt1)
			#print 'after: tt1=', tt1
		if not k in m2:
			tt2 = nil
		else:
			tt2 = m2[k].getcolumn(fieldname)
			if preproc is not None:
				tt2 = preproc(tt2)
		t1.append(tt1)
		t2.append(tt2)

	for k,v in m2.items():
		if k in m1:
			continue
		tt2 = v.getcolumn(fieldname)
		if preproc is not None:
			tt2 = preproc(tt2)
		t2.append(tt2)
		t1.append(nil)

	return t1,t2


def get_scalar(fieldname, m1, m2, nil):
	t1,t2 = get_field(fieldname, m1, m2, None)
	s1,s2 = [],[]
	for tt1,tt2 in zip(t1,t2):
		if tt1 is None and tt2 is None:
			s1.append(nil)
			s2.append(nil)
			continue
		if tt1 is None:
			s1.append([nil for i in range(len(tt2))])
		else:
			s1.append(tt1)
		if tt2 is None:
			s2.append([nil for i in range(len(tt1))])
		else:
			s2.append(tt2)
	t1 = hstack(s1)
	t2 = hstack(s2)
	return t1,t2


if __name__ == '__main__':
	parser = OptionParser()
	opt,args = parser.parse_args()

	for id1,d1 in enumerate(args):
		allm1 = os.path.join(d1, 'matches.pickle')
		if os.path.exists(allm1):
			continue
		print 'Caching', allm1
		matches = {}
		for fn in glob(d1 + '/*.match'):
			matches[fn.replace(d1+'/', '')] = fits_table(fn)
		pickle_to_file(matches, allm1)
			

	for id1,d1 in enumerate(args):
		allm1 = os.path.join(d1, 'matches.pickle')
		m1 = unpickle_from_file(allm1)
		for id2,d2 in enumerate(args):
			if id2 <= id1:
				continue

			allm2 = os.path.join(d2, 'matches.pickle')
			m2 = unpickle_from_file(allm2)

			# CPU time.
			tz,tinf = 1e-3, 100.
			clf()
			t1,t2 = get_scalar('timeused', m1, m2, tinf)
			#plot(t1, t2, 'r.')
			loglog(t1, t2, 'r.')
			t1 = clip(t1, tz, tinf)
			t2 = clip(t2, tz, tinf)
			xlabel(d1 + ': CPU time (s)')
			ylabel(d2 + ': CPU time (s)')
			fn = 'cputime-%i-%i.png' % (id1, id2)
			print 'saving', fn
			plot([tz,tinf],[tz,tinf], 'k-')
			axis([tz,tinf,tz,tinf])
			savefig(fn)
			
			# N objs
			tinf = 1000.
			clf()
			def ppmax(x):
				return amax(x, axis=1)
			t1,t2 = get_field('fieldobjs', m1, m2, tinf, preproc=ppmax)
			t1 = array(list(flatten(t1)))
			t2 = array(list(flatten(t2)))
			#print 'nobjs: t1=', t1
			#print 't2=', t2
			dN = 50
			I = logical_or((t1 - t2) > dN , (t1 - t2) < -dN)
			for i in flatnonzero(I):
				k1,v1 = (m1.items())[i]
				k2,v2 = (m2.items())[i]
				print 'Nmatch changed:'
				print '  %s: %i' % (k1, t1[i])
				print '  %s: %i' % (k2, t2[i])
				

			plot(t1, t2, 'r.')
			xlabel(d1 + ': N objects examined')
			ylabel(d2 + ': N objects examined')
			fn = 'nobjs-%i-%i.png' % (id1, id2)
			print 'saving', fn
			plot([0,100],[0,100], 'k-')
			axis([0,100,0,100])
			savefig(fn)

			# N matches
			tz,tinf = 0.,300.
			clf()
			t1,t2 = get_scalar('nmatch', m1, m2, tinf)
			print 'm1,m2,t1,t2', len(m1),len(m2), len(t1),len(t2)
			I = (t2 < 50)
			for i in flatnonzero(I):
				k,v = (m2.items())[i]
				print 't2 file %s, nmatches %i' % (k, t2[i])
			t1 = clip(t1, tz, tinf)
			t2 = clip(t2, tz, tinf)
			plot(t1, t2, 'r.')
			xlabel(d1 + ': N objects matched')
			ylabel(d2 + ': N objects matched')
			fn = 'nmatch-%i-%i.png' % (id1, id2)
			print 'saving', fn
			plot([tz,tinf],[tz,tinf], 'k-')
			axis([tz,tinf,tz,tinf])
			savefig(fn)

			# Index #
			tinf = 1000.
			clf()
			t1,t2 = get_scalar('indexid', m1, m2, tinf)
			plot(t1, t2, 'r.')
			xlabel(d1 + ': Index number')
			ylabel(d2 + ': Index number')
			print 't1: ', unique(t1)
			print 't2: ', unique(t2)
			axis([min(t1)-1, max(t1)+1, min(t2)-1, max(t2)+1])
			xticks(range(min(t1), max(t1)))
			yticks(range(min(t2), max(t2)))
			fn = 'indexid-%i-%i.png' % (id1, id2)
			print 'saving', fn
			savefig(fn)

