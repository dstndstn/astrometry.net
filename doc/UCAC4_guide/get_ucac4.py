# Script for automatically downloading UCAC4 catalogue
# Last revision: Jan 08, 2015
# Author: Denis Vida, denis.vida@gmail.com

# This file is part of the Astrometry.net suite.
# Copyright 2014 Denis Vida.
# The Astrometry.net suite is free software; you can redistribute
# it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, version 2.
# The Astrometry.net suite is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with the Astrometry.net suite ; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA

# Start dectination (in degrees, e.g. +50.0)
start_dec = +90.0

# End declination (in degrees, e.g. -30.0)
end_dec = -90.0

import urllib
import os
from shutil import copyfileobj
import bz2

# Retry failed downloads (how many times?)
No_retries = 5 

# The URL prefix to the FTP repository
prefix = 'http://cdsarc.u-strasbg.fr/viz-bin/ftp-index?/ftp/cats/aliases/U/UCAC4/UCAC4/u4b/'

def Convert_Dec_to_Znum(start_dec, end_dec):
	""" Convert declination range to UCAC4 file number range. """
	z_file_constatnt = 180.0/900

	if start_dec < end_dec:
		temp = start_dec
		start_dec = end_dec
		end_dec = start_dec

	if start_dec > +90.0:
		start_dec = +90.0

	if end_dec < -90.0:
		end_dec = -90.0

	start_z = int(-(start_dec-90)/z_file_constatnt) + 1
	end_z = int(-(end_dec-90)/z_file_constatnt)

	return start_z, end_z

def Download_File(name):
	""" Download UCAC4 file. """

	url_name = prefix+name
	ucac_file = urllib.URLopener()
	ucac_file.retrieve(url_name, name)
	
	inp = open(name, 'rb')
	bz2_file = bz2.BZ2File(name+'.bz2', 'wb', compresslevel=1) 
	copyfileobj(inp, bz2_file)
	inp.close()
	bz2_file.close()

	os.remove(name)

	return 0

start_z, end_z = Convert_Dec_to_Znum(start_dec, end_dec)

# Downloading the catalogue
fail_list = []
successes = 0
for i in range(start_z, end_z+1):
	name = 'z'+str(i).zfill(3)

	print 'Downloading: '+name
	try:
		Download_File(name)

		successes += 1
	except:
		fail_list.append(name)
		print 'ERROR downloading file: ', name

# Retry failed downloads
for i in range(No_retries):
	for name in fail_list:
		print 'Retrying:', name
		try:
			Download_File(name)
			successes += 1
			fail_list.pop(fail_list.index(name))
		except:
			print 'Will retry', name, 'again...'


if len(fail_list) == 0:
	print 'SUCCESS! All files downloaded successfully!'
elif successes > 0:
	print 'WARNING! PARTIAL SUCCESS:'
	print successes, 'files downloaded successfully,', len(fail_list), 'failed!'
	print 'These files were NOT downloaded:', fail_list
else:
	print 'ERROR! ALL FILES FAILED TO DOWNLOAD!'
	print 'Check your internet connection or try downloading later...'