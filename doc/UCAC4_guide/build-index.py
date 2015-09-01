# Script for automatically running astrometry.net index building
# Last revision: Dec 20, 2014
# Author: Denis Vida, denis.vida@gmail.com

# This file is part of the Astrometry.net suite.
# Copyright 2014 Denis Vida.
# The Astrometry.net suite is free software; you can redistribute
# it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2, or
# (at your option) any later version.
# The Astrometry.net suite is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with the Astrometry.net suite ; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA

import time
import os

#Determine the range of scales (default: from 5 to 7):
scale_range = [5, 7]

#Base file name
base_file_name = 'cut-'
extenstion = '.fits'

#Determine the file naming range, e.g. the output names will be cut-00.fits, cut-01.fits, ..., cut-11.fits
file_range = [0, 11]

#Run name (number to identify the indexing run, default is current date + scale)
run_name = time.strftime('%Y%m%d')


print 'Running index generation with following parameters:'
print '\t Scale range: ', scale_range
print '\t Base file name: ', base_file_name + '??' + extenstion
print '\t File range: ', file_range
print '\t Run name: ', run_name

for file_no in range(file_range[0], file_range[1]+1):
	for scale in range(scale_range[0], scale_range[1]+1):
		file_no = str(file_no).zfill(2)
		scale = str(scale)

		base_command = 'build-astrometry-index -i '+base_file_name+file_no+extenstion+' -o index-ucac4-'+scale+'-'+file_no+'.fits -P '+scale+' -S MAG -H '+file_no+' -s 1 -I '+run_name+scale.zfill(2)
		
		print base_command

		#Run command
		os.system(base_command)
