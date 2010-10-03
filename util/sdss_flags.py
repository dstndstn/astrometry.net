flag1dict = dict(
		CANONICAL_CENTER          =        1, 
		BRIGHT                    =        2, 
		EDGE                      =        4, 
		BLENDED                   =        8, 
		CHILD                     =       16, 
		PEAKCENTER                =       32, 
		NODEBLEND                 =       64, 
		NOPROFILE                 =      128, 
		NOPETRO                   =      256, 
		MANYPETRO                 =      512, 
		NOPETRO_BIG               =     1024, 
		DEBLEND_TOO_MANY_PEAKS    =     2048, 
		CR                        =     4096, 
		MANYR50                   =     8192, 
		MANYR90                   =    16384, 
		BAD_RADIAL                =    32768, 
		INCOMPLETE_PROFILE        =    65536, 
		INTERP                    =   131072, 
		SATUR                     =   262144, 
		NOTCHECKED                =   524288, 
		SUBTRACTED                =  1048576, 
		NOSTOKES                  =  2097152, 
		BADSKY                    =  4194304, 
		PETROFAINT                =  8388608, 
		TOO_LARGE            =      16777216, 
		DEBLENDED_AS_PSF     =      33554432, 
		DEBLEND_PRUNED       =      67108864, 
		ELLIPFAINT           =     134217728, 
		BINNED1              =     268435456, 
		BINNED2              =     536870912, 
		BINNED4              =    1073741824, 
		MOVED                =    2147483648L,
		)

flag2dict = dict(
	DEBLENDED_AS_MOVING       =        1, 
	NODEBLEND_MOVING          =        2,
	TOO_FEW_DETECTIONS        =        4, 
	BAD_MOVING_FIT            =        8, 
	STATIONARY                =       16, 
	PEAKS_TOO_CLOSE           =       32, 
	BINNED_CENTER             =       64, 
	LOCAL_EDGE                =      128, 
	BAD_COUNTS_ERROR          =      256, 
	BAD_MOVING_FIT_CHILD      =      512, 
	DEBLEND_UNASSIGNED_FLUX   =     1024, 
	SATUR_CENTER              =     2048, 
	INTERP_CENTER             =     4096, 
	DEBLENDED_AT_EDGE         =     8192, 
	DEBLEND_NOPEAK            =    16384, 
	PSF_FLUX_INTERP           =    32768, 
	TOO_FEW_GOOD_DETECTIONS   =    65536, 
	CENTER_OFF_AIMAGE         =   131072, 
	DEBLEND_DEGENERATE        =   262144, 
	BRIGHTEST_GALAXY_CHILD    =   524288, 
	CANONICAL_BAND            =  1048576, 
	AMOMENT_UNWEIGHTED        =  2097152, 
	AMOMENT_SHIFT             =  4194304, 
	AMOMENT_MAXITER           =  8388608, 
	MAYBE_CR                  = 16777216, 
	MAYBE_EGHOST              = 33554432, 
	NOTCHECKED_CENTER         = 67108864, 
	HAS_SATUR_DN       =       134217728, 
	SPARE4             =       268435456, 
	SPARE3             =       536870912, 
	SPARE2             =      1073741824, 
	SPARE1             =      2147483648L,
)

def fpobjc_decode_flag1(name):
	return flag1dict[name]

def fpobjc_decode_flag2(name):
	return flag2dict[name]

def fpobjc_flag1_to_name(flags, sep=', '):
	names = []
	for k,v in flag1dict.items():
		if flags & v:
			names.append(k)
	return sep.join(names)
