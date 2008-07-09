import sys

from time import clock

from array import array

import png

import wsgiref.handlers
from google.appengine.ext import webapp

# nope, doesn't work.
#from numpy import *

def heatmap(pix):
	if pix <= 96.0:
		r = pix * 255.0 / 96.0
		g = b = 0
	elif pix <= 192.0:
		r = 255
		g = (pix - 96.0) * 255.0 / 96.0
		b = 0
	else:
		r = g = 255
		b = (pix - 192.0) * 255.0 / 63.0
	return [int(r), int(g), int(b)]


class MainPage(webapp.RequestHandler):
	def get(self):
		res = self.response
		req = self.request

		res.headers['Content-Type'] = 'image/png'
		W = int(req.get('w', 256))
		H = int(req.get('h', 256))

		layers = req.get('layers', '').split(',')
		if 'grid' in layers:
			pixels = array('B')
			off = [0,0,0,0]
			on  = [128,128,0,255]
			for j in range(H):
				for i in range(W):
					if j == H/2 or i == W/2:
						pixels.extend(on)
					else:
						pixels.extend(off)
			writer = png.Writer(width=W, height=H, has_alpha=True)
			writer.write_array(res.out, pixels)
			return
		
		cx = float(req.get('seedx', -0.726895347709114071439))
		cy = float(req.get('seedy', 0.188887129043845954792))

		ax = float(req.get('ax', 0))
		ay = float(req.get('ay', 0))
		zoom = float(req.get('b', 1))

		nmax = 2.**(zoom+1)
		xmin = ax / nmax
		ymin = ay / nmax
		xmax = (ax+1) / nmax
		ymax = (ay+1) / nmax

		# rescale to the [-1, 1] box.
		xmin = xmin * 2.0 - 1.0
		xmax = xmax * 2.0 - 1.0
		ymin = ymin * 2.0 - 1.0
		ymax = ymax * 2.0 - 1.0

		xstep = (xmax - xmin) / float(W)
		ystep = (ymax - ymin) / float(H)

		cmap = []
		for i in range(256):
			cmap.append(heatmap(i))

		y0 = [ymin + ystep * j for j in range(H)]
		x0 = [xmin + ystep * i for i in range(W)]

		t = [0]*10

		t[0] = clock()
		pixels = array('B')
		for j in range(H):
			for i in range(W):
				x = x0[i]
				y = y0[j]
				for k in range(254):
					xn = x*x - y*y + cx
					yn = 2.0 * x * y + cy
					x = xn
					y = yn
					if x*x + y*y > 2.0:
						break
				pixels.extend(cmap[k])
		t[1] = clock()

		for i in range(9):
			if t[i+1] == 0:
				break
			print >> sys.stderr, 't%i to %i: %g' % (i, i+1, t[i+1]-t[i])

		writer = png.Writer(width=W, height=H)
		writer.write_array(res.out, pixels)



# yummy boilerplate...
def main():
	application = webapp.WSGIApplication([('/tile', MainPage)], debug=True)
	wsgiref.handlers.CGIHandler().run(application)
if __name__ == "__main__":
	main()

