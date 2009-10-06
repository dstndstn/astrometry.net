import matplotlib
matplotlib.use('Agg')

# Check out: http://matplotlib.sourceforge.net/users/customizing.html
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font', family='computer modern roman')
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{color}')
rc('lines', antialiased=True)
rc('legend', fontsize='medium', numpoints=1) # default: large
rc('axes', titlesize='medium')
rc('image', interpolation='nearest', cmap='gray', origin='lower',
   resample=False)
