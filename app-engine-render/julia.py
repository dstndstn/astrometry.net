import sys
from logging import debug
from time import clock

from array import array

import png

import wsgiref.handlers
from google.appengine.ext import webapp

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

		cx = float(req.get('seedx', -0.726895347709114071439))
		cy = float(req.get('seedy', 0.188887129043845954792))

		ax = float(req.get('ax', 0))
		ay = float(req.get('ay', 0))
		zoom = float(req.get('b', 1))

		xmin = ax / (2.**zoom)
		ymin = ay / (2.**zoom)
		xmax = (ax+1) / (2.**zoom)
		ymax = (ay+1) / (2.**zoom)

		# rescale to the [-1, 1] box.
		xmin = xmin * 2.0 - 1.0
		xmax = xmax * 2.0 - 1.0
		ymin = ymin * 2.0 - 1.0
		ymax = ymax * 2.0 - 1.0

		xstep = (xmax - xmin) / float(W)
		ystep = (ymax - ymin) / float(H)

		t = [0]*10
		t[0] = clock()

		cmap = []
		for i in range(256):
			cmap.append(heatmap(i))

		t[1] = clock()

		xy = [(xmin + xstep * i, ymin + ystep * j, j*W+i) for j in range(H) for i in range(W)]

		t[2] = clock()

		pp = [0]*len(xy)

		t[3] = clock()

		for k in range(254):
			ii = [i for (x,y,i) in xy if x*x+y*y >= 2]
			for i in ii:
				pp[i] = k
			xy = [(x*x - y*y + cx, 2.0 * x * y + cy, i) for (x,y,i) in xy if x*x+y*y < 2]
		t[4] = clock()

		for (x,y,i) in xy:
			pp[i] = 255

		t[5] = clock()

		pixels = array('B')
		for k in pp:
			pixels.extend(cmap[k])
		t[6] = clock()

		#print >> sys.stderr, 'min in pixels:', min(pixels), 'max in pixels:', max(pixels)

		writer = png.Writer(width=W, height=H, compression=1)
		writer.write_array(res.out, pixels)
		t[7] = clock()

		for i in range(7):
			print >> sys.stderr, 't%i to %i: %g' % (i, i+1, t[i+1]-t[i])



# yummy boilerplate...
def main():
	application = webapp.WSGIApplication([('/tile', MainPage)], debug=True)
	wsgiref.handlers.CGIHandler().run(application)
if __name__ == "__main__":
	main()

