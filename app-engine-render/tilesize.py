import sys

from array import array

import png

import wsgiref.handlers
from google.appengine.ext import webapp

class MainPage(webapp.RequestHandler):
	def get(self):
		res = self.response
		req = self.request
		res.headers['Content-Type'] = 'image/png'
		W = int(req.get('w', 256))
		H = int(req.get('h', 256))

		ax = float(req.get('ax', 0))
		ay = float(req.get('ay', 0))
		zoom = float(req.get('b', 1))

		nmax = 2.**(zoom+1)
		xmin = ax / nmax
		ymin = ay / nmax
		xmax = (ax+1) / nmax
		ymax = (ay+1) / nmax

		xstep = (xmax - xmin) / float(W)
		ystep = (ymax - ymin) / float(H)

		writer = png.Writer(width=W, height=H)
		pixels = array('B')
		#for y in range(H-1,-1,-1):
		for y in range(H):
			for x in range(W):
				r = int(256 * (xmin + xstep * x))
				g = int(256 * (ymin + ystep * y))
				b = 128
				pixels.append(r)
				pixels.append(g)
				pixels.append(b)
		writer.write_array(res.out, pixels)



# yummy boilerplate...
def main():
	application = webapp.WSGIApplication([('/tileSizeTile', MainPage)], debug=True)
	wsgiref.handlers.CGIHandler().run(application)
if __name__ == "__main__":
	main()

