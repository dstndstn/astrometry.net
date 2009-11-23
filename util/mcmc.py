from numpy import *

def mcmc(data, params, proposal, lnposterior, step_info,
		 prior_info, nlinks, beta=1.,
		 record=None, record_info=None, keepchain=False,
		 verbose=True):

	OUTPUTPERIOD = 100
	lnp = lnposterior(data, params, prior_info)
	if verbose:
		print 'starting lnp=', lnp
	oldlnp = lnp
	oldparams = params
	bestparams = oldparams
	bestlnp = oldlnp
	if verbose:
		print 'doing', nlinks, 'links of MCMC...'
	naccept = 0

	chain = []

	for link in range(nlinks):
		(newparams, step_info) = proposal(oldparams, step_info)
		lnp = lnposterior(data, newparams, prior_info)
		randnum = random.uniform()
		accept = (beta * (lnp - oldlnp)) > log(randnum)
		if record is not None:
			record_info = record(oldparams, oldlnp, newparams, lnp, randnum,
								 accept, link, nlinks, record_info)
		if accept:
			# keep new parameters
			if keepchain:
				chain.append((lnp, newparams))

			oldparams = newparams
			oldlnp = lnp
			naccept += 1
			if lnp > bestlnp:
				bestlnp = lnp
				bestparams = newparams
		else:
			# keep old parameters
			if keepchain:
				chain.append((oldlnp, oldparams))

			pass
		if verbose and ((link % OUTPUTPERIOD == 0) or link == nlinks-1):
			print link, float(naccept)/(link+1), bestlnp, bestparams
	return (bestparams, bestlnp, chain, step_info, naccept)


def cycle_proposals(oldparams, stepinfo):
	(stepnum, sigmas, lastip) = stepinfo
	NZ = array([i for i,s in enumerate(sigmas) if s != 0])
	np = len(NZ)
	ip = NZ[stepnum % np]
	params = oldparams[:]
	params[ip] += random.normal() * sigmas[ip]
	return (params, (stepnum+1, sigmas, ip))

# roll-off a quadratic error function to linear at the given 'linsig'.
# the curve and its slope are continuous.
def quadratic_to_linear(sig2, linsig):
	lin = (sig2 > linsig**2)
	sig2[lin] = -linsig**2 + 2.*linsig * sqrt(sig2[lin])
	return sig2

