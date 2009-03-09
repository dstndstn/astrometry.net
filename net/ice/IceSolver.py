#! /usr/bin/env python

import sys
import time
import threading

import Glacier2
import Ice
import IceGrid

import SolverIce

from astrometry.util.file import *
import astrometry.net.settings as settings
from astrometry.net.portal.log import log

# FIXME
configfile = settings.BASEDIR + 'astrometry/net/ice/config.client'

username = 'foo'
password = 'bar'

printlog = False
def logmsg(*msg):
	if printlog:
		print ' '.join([str(m) for m in msg])
	else:
		log('IceSolver:', *msg)

theclient = None
theclientlock = threading.Lock()

class SolverClient(object):
	def __init__(self):
		pass

	def init(self):
		settings = Ice.InitializationData()
		settings.properties = Ice.createProperties(None, settings.properties)
		settings.properties.load(configfile)
		ice = Ice.initialize(settings)
		self.ice = ice
		self.router = None
		self.initrouter()
		#self.getready()

	def getsession(self, router):
		session = None
		try:
			session = router.createSession(username, password) #'test-%i' % int(time.time()), 'test')
		except Glacier2.PermissionDeniedException,ex:
			logmsg('router session permission denied:', ex)
			raise ex
		except Glacier2.CannotCreateSessionException,ex:
			logmsg('router session: cannot create:', ex)
			raise ex
		logmsg('router is', router)
		logmsg('session is', session)
		return session

	def getadapter(self, router):
		#adapter = self.ice.createObjectAdapter('Callback.Client')			
		adapter = self.ice.createObjectAdapterWithRouter('Callback.Client', router)
		return adapter

	def initrouter(self):
		logmsg('initrouter()')
		router = self.ice.getDefaultRouter()
		router = Glacier2.RouterPrx.checkedCast(router)
		if not router:
			logmsg('not a glacier2 router')
			raise 'not a glacier2 router'

		if self.router is not None:
			# Workaround as per:
			# http://www.zeroc.com/forums/help-center/4207-recovery-after-router-failure.html
			# register and then unregister the client to clear
			# the router's cache.
			session = self.getsession(router)
			adapter = self.getadapter(router)
			try:
				router.destroySession()
			except:
				pass
			adapter.destroy()

		session = self.getsession(router)

		self.router = router
		self.session = session

		logmsg('creating adapter...')
		self.adapter = self.getadapter(self.router)
		logmsg('created adapter', self.adapter)
		self.adapter.activate()

	# check that all my stuff is live...
	def getready(self):
		logmsg('Router is a', type(self.router))
		if self.router is None:
			self.initrouter()
		try:
			logmsg('Pinging router...')
			self.router.ice_ping()
			logmsg('Getting category from router...')
			category = self.router.getCategoryForClient()
			logmsg('Router ready to go')
		except Ice.ConnectionLostException,ex:
			logmsg('Got ConnectionLostException on pinging router.	Reopening connection...')
			self.initrouter()

	def server_status(self):
		logmsg('get servers')
		servers = self.find_all_solvers()
		logmsg('servers:', servers)
		rtn = []
		for s in servers:
			logmsg('getting status for', s)
			stat = s.status()
			logmsg('status:', stat)
			rtn.append((s,stat))
		return rtn

	# this will be called from multiple threads.
	def solve(self, jobid, axy, logfunc):
		category = self.router.getCategoryForClient()
		myid = Ice.Identity()
		myid.category = category
		myid.name = Ice.generateUUID()
		#logmsg('my id:', myid)

		self.adapter.add(LoggerI(logfunc), myid)
		logproxy = SolverIce.LoggerPrx.uncheckedCast(self.adapter.createProxy(myid))
		#logmsg('my logger:', logproxy)

		logmsg('get servers')
		servers = self.find_all_solvers()
		logmsg('servers:', servers)

		results = [SolverResult(s,jobid) for s in servers]

		logmsg('making ICE calls...')
		for (s,r) in zip(servers,results):
			s.solve_async(r, jobid, axy, logproxy)

		waiting = [r for r in results]
		outfiles = None
		lastping = time.time()
		pingperiod = 30 # seconds
		while len(waiting):
			time.sleep(5)
			logmsg('Job', jobid, 'waiting for %i servers' % len(waiting))
			for r in waiting:
				if r.isdone() and r.files is not None and r.solved:
					logmsg('Job', jobid, 'solved by server', str(r.server))
					outfiles = r.files
					break
			if outfiles:
				for r in waiting:
					if not r.isdone():
						logmsg('Job', jobid, 'finished: sending cancel to', str(r.server))
						logfunc('Cancelling ' + str(r.server))
						r.server.ice_oneway().cancel(jobid)
				break
			waiting = [r for r in waiting if not r.isdone()]

			tnow = time.time()
			if tnow - lastping > pingperiod:
				logfunc('Sending pings...')
				for r in waiting:
					r.server.ice_oneway().ice_ping()
				lastping = tnow

		logmsg('Received Ice response(s)')

		# grace period to let servers send their last log messages.
		time.sleep(3)
		return outfiles

	def find_all_solvers(self):
		q = self.ice.stringToProxy('SolverIceGrid/Query')
		logmsg('Q is', q)
		q = IceGrid.QueryPrx.checkedCast(q)
		logmsg('Q is', q)
		solvers = q.findAllObjectsByType('::SolverIce::Solver')
		logmsg('Found %i solvers' % len(solvers))
		for s in solvers:
			logmsg('  ', s)
		servers = []
		for s in solvers:
			logmsg('Resolving ', s)
			server = SolverIce.SolverPrx.checkedCast(s)
			servers.append(server)
		return servers




class LoggerI(SolverIce.Logger):
	def __init__(self, logfunc):
		self.logfunc = logfunc
	def logmessage(self, msg, current=None):
		#logmsg('logger callback')
		self.logfunc(msg)

class SolverResult(object):
	def __init__(self, server, jobid):
		self.tardata = None
		self.files = None
		self.errmsg = None
		self.failed = False
		self.solved = False
		self.server = server
		self.jobid = jobid
		self.got_response = False
	def ice_response(self, files, solved, errmsg):
		#logmsg('async response from server', self.server, ', job', self.jobid, ': ', (solved and 'solved' or 'did not solve'))
		logmsg('async response from server', self.server, ', job', self.jobid, ': ')
		logmsg('solved:', solved)
		logmsg('errmsg:', errmsg)
		logmsg('len files: %i' % len(files))
		for f in files:
			#logmsg('  type:', str(type(f)))
			#logmsg('  type f.data:', str(type(f.data)))
			logmsg('  f.name', f.name)
			logmsg('  len f.data:', len(f.data))
		#logmsg('files:', files)
		#self.tardata = tardata
		self.solved = solved
		self.errmsg = errmsg
		self.files = files
		self.got_response = True

	def ice_exception(self, ex):
		logmsg('async exception from server', self.server, ', job', self.jobid, ': ', ex)
		self.failed = True

	def isdone(self):
		return self.failed or self.got_response

def solve(jobid, axy, logfunc):
	global theclient
	
	theclientlock.acquire()
	if not theclient:
		theclient = SolverClient()
		theclient.init()
	else:
		theclient.getready()
	theclientlock.release()

	files = theclient.solve(jobid, axy, logfunc)
	return files

def status():
	global theclient
	
	theclientlock.acquire()
	if not theclient:
		theclient = SolverClient()
		theclient.init()
	else:
		theclient.getready()
	theclientlock.release()

	stats = theclient.server_status()
	

if __name__ == '__main__':
	printlog = True
	if len(sys.argv) == 2 and sys.argv[1] == 'status':
		#while True:
		#	status()
		status()
		time.sleep(30)
		status()
		sys.exit(0)

	if len(sys.argv) != 3:
		print 'Usage: %s <jobid> <job.axy>\n' % sys.argv[0]
		print '	  or: %s status\n' % sys.argv[0]
		sys.exit(-1)

	jobid = sys.argv[1]
	axyfn = sys.argv[2]
	axydata = read_file(axyfn)
	print 'jobid is ', jobid
	print 'axyfile is %i bytes' % len(axydata)

	def logfunc(msg):
		logmsg(msg)

	files = solve(jobid, axydata, logfunc)

	if files is None:
		print 'Field unsolved.'
	else:
		print 'got %i files:' % len(files)
		for f in files:
			logmsg('  %i bytes:' % len(f.data), fn)
			write_file(f.data, fn)

