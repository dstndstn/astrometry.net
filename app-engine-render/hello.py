import sys

from array import array

import png

print 'Content-Type: image/png'
print ''
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
writer.write_array(sys.stdout, pixels)
