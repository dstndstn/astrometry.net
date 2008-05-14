import sys

from array import array

import png

import wsgiref.handlers
from google.appengine.ext import webapp

class MainPage(webapp.RequestHandler):
	def get(self):
		res = self.response
		res.headers['Content-Type'] = 'image/png'
		W = 256
		H = 256
		writer = png.Writer(width=W, height=H, has_alpha=True, compression=1)
		pixels = array('B')
		for x in range(H):
			for y in range(W):
				r = x
				g = y
				b = 64
				a = 128
				pixels.append(r)
				pixels.append(g)
				pixels.append(b)
				pixels.append(a)
		#writer.write_array(sys.stdout, pixels)
		writer.write_array(res.out, pixels)

		#res.out.write('Hello, webapp World!')



# yummy boilerplate...
def main():
	application = webapp.WSGIApplication([('/', MainPage)], debug=True)
	wsgiref.handlers.CGIHandler().run(application)
if __name__ == "__main__":
	main()

