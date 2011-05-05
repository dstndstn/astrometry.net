import numpy as np

def usnob_apply_cuts(X):
	# USNO-B sources (not Tycho-2)
	I = (X.num_detections >= 2)
	# no diffraction spikes
	I = np.logical_and(I, np.logical_not(X.flags[:,0]))
	#X = X[I]
	#print '%i pass USNO-B diffraction spike cut' % len(X)

	if hasattr(X, 'an_diffraction_spike'):
		f = X.an_diffraction_spike
		print 'f shape', f.shape
		#u = unique(f)
		#print 'unique flags:', u
		for j in range(8):
			print 'flag', j, 'vals', unique(f[:,j])
		I *= (np.logical_not(f[:,7]) * np.logical_not(f[:,6]))
		#X = X[good]
		#print '%i pass AN diffraction spike cut' % len(X)
	return I

def usnob_compute_average_mags(X):
	# Compute average R and B mags.
	epoch1 = (X.field_1 > 0)
	epoch2 = (X.field_3 > 0)
	nmag = np.where(epoch1, 1, 0) + np.where(epoch2, 1, 0)
	summag = np.where(epoch1, X.magnitude_1, 0) + np.where(epoch2, X.magnitude_3, 0)
	X.r_mag = np.where(nmag == 0, 0, summag / nmag)
	# B
	epoch1 = (X.field_0 > 0)
	epoch2 = (X.field_2 > 0)
	nmag = np.where(epoch1, 1, 0) + np.where(epoch2, 1, 0)
	summag = np.where(epoch1, X.magnitude_0, 0) + np.where(epoch2, X.magnitude_2, 0)
	X.b_mag = np.where(nmag == 0, 0, summag / nmag)
