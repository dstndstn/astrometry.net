from astrometry.net1.portal.log import log as logmsg
from astrometry.net1.portal.job import Tag, Job
from astrometry.util import healpix

def get_neighbouring_healpixes(nside, hp):
	hps = [ (nside, hp) ]
	# neighbours at this scale.
	neigh = healpix.get_neighbours(hp, nside)
	for n in neigh:
		hps.append((nside, n))
	# the next bigger scale.
	hps.append((nside-1, n/4))
	# the next smaller scale, plus neighbours.
	# (use a set because some of the neighbours are in common)
	allneigh = set()
	for i in range(4):
		n = healpix.get_neighbours(hp*4 + i, nside+1)
		allneigh.update(n)
	for i in allneigh:
		hps.append((nside+1, i))
	return hps

def get_tags_nearby(job):
	tags = job.tags.all().filter(machineTag=True, text__startswith='hp:')
	if tags.count() != 1:
		logmsg('get_tags_nearby: no such tag')
		return None
	tag = tags[0]
	parts = tag.text.split(':')
	if len(parts) != 3:
		logmsg('get_tags_nearby: bad tag')
		return None
	nside = int(parts[1])
	hp = int(parts[2])
	#logmsg('nside %i, hp %i' % (nside, hp))
	hps = get_neighbouring_healpixes(nside, hp)
	tagtxts = ['hp:%i:%i' % (nside,hp) for (nside,hp) in hps]
	tags = Tag.objects.all().filter(machineTag=True, text__in=tagtxts)
	return tags

def add_tags_to_job(job):
	# Find the field size:
	wcs = job.get_tan_wcs()
	radiusdeg = wcs.get_field_radius()
	nside = healpix.get_closest_pow2_nside(radiusdeg)
	#logmsg('Field has radius %g deg.' % radiusdeg)
	#logmsg('Closest power-of-2 healpix Nside is %i.' % nside)
	(ra,dec) = wcs.get_field_center()
	#logmsg('Field center: (%g, %g)' % (ra,dec))
	logmsg('nside', nside)
	hp = healpix.radectohealpix(ra, dec, nside)
	logmsg('ok')
	#logmsg('Healpix: %i' % hp)
	tag = Tag(job=job,
			  user=job.get_user(),
			  machineTag=True,
			  text='hp:%i:%i' % (nside, hp),
			  addedtime=Job.timenow())
	tag.save()
