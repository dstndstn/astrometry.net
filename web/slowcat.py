#! /usr/bin/env python

import sys
import time

if __name__ == '__main__':

	readblock = 1024
	writeblock = 1024
	period = 0.1

	for arg in sys.argv[1:]:
		fin = open(arg, 'rb')
		while True:
			data = fin.read(readblock)
			if not len(data):
				break
			while len(data):
				out = data[:writeblock]
				data = data[writeblock:]
				sys.stdout.write(out)
				sys.stdout.flush()
				time.sleep(period)

