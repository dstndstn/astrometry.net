import matplotlib
#matplotlib.use('Agg')
#matplotlib.use('cairo')

# Check out: http://matplotlib.sourceforge.net/users/customizing.html
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font', family='computer modern roman')
rc('text', usetex=True)
#rc('text.latex', preamble=r'\usepackage{color}')
rc('lines', antialiased=True)
rc('legend', fontsize='medium', numpoints=1) # default: large
rc('axes', titlesize='medium')
rc('image', interpolation='nearest', cmap='gray', origin='lower',
   resample=False)

def savefig(fn, **kwargs):
	from pylab import savefig as sf
	sf(fn, **kwargs)
	print 'Wrote figure', fn
