from plotstuff_c import *

# Could consider using swig's "addmethods" mechanism to create this "class" rep.

class Plotstuff(object):
	def __init__(self):
		p = plotstuff_new()
		print 'plotstuff.__init__, pargs=', p
		self.pargs = p
		#self.pargs = plotstuff_new()

	def __del__(self):
		print 'plotstuff.__del__, pargs=', self.pargs
		plotstuff_free(self.pargs)

	def __getattr__(self, name):
		return self.pargs.__getattr__(name)

	def __setattr__(self, name, val):
		if name == 'pargs':
			print 'plotstuff.py: setting pargs to', val
			self.__dict__[name] = val
		elif name == 'size':
			print 'plotstuff.py: setting plot size of', self.pargs, 'to %i,%i' % (val[0], val[1])
			plotstuff_set_size(self.pargs, val[0], val[1])
		elif name == 'color':
			self.set_color(val)
		elif name == 'marker' and type(val) is str:
			plotstuff_set_marker(self.pargs, val)
		else:
			self.pargs.__setattr__(name, val)

	def plot(self, layer):
		plotstuff_plot_layer(self.pargs, layer)

	def set_wcs_box(self, ra, dec, width):
		plotstuff_set_wcs_box(self.pargs, ra, dec, width)

	def set_color(self, color):
		x = plotstuff_set_color(self.pargs, color)
		print 'set_color returned:', x

	def plot_grid(self, rastep, decstep, ralabelstep=None, declabelstep=None):
		grid = plot_grid_get(self.pargs)
		grid.rastep = rastep
		grid.decstep = decstep
		grid.ralabelstep = ralabelstep
		grid.declabelstep = declabelstep
		self.plot('grid')
		
	def write(self, filename):
		self.outfn = filename
		plotstuff_output(self.pargs)
