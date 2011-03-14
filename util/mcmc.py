from numpy import *

def metropolis(data, model, nlinks, beta=1., keepchain=True, startlink=0):
	'''
	The "model" object must implement:
	
	p = model.get_params()
	-- this must return an *independent copy* of the parameters.

	model.set_params(p)

	p = model.propose_params()

	model.tally(accept, linknumber)   accept: boolean

	lnp = model.lnposterior(data)


	The "data" object is an opaque object that just gets passed back to you

	Returns: (bestlnp, bestparams, chain)

	Where chain is a list of all the MCMC steps:
	  [ (lnp, params), (lnp, params), ... ]

	'''
	oldparams = model.get_params()
	oldlnp = model.lnposterior(data)

	bestparams = oldparams
	bestlnp = oldlnp

	chain = []
	for link in range(startlink, nlinks):

		newparams = model.propose_params()
		model.set_params(newparams)
		newlnp = model.lnposterior(data)

		randnum = random.uniform()
		accept = (beta * (newlnp - oldlnp)) > log(randnum)

		model.tally(accept, link)

		if accept:
			# keep new parameters
			if keepchain:
				chain.append((newlnp, newparams))
			oldparams = newparams
			oldlnp = newlnp
			#naccept += 1
			if newlnp > bestlnp:
				bestlnp = newlnp
				bestparams = newparams
		else:
			# keep old parameters
			if keepchain:
				chain.append((oldlnp, oldparams))
			model.set_params(oldparams)

	return (bestlnp, bestparams, chain)

'''
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

'''
