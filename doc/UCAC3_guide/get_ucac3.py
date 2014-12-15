#Script for automatically downloading UCAC3 catalogue
#Last revision: Dec 15, 2014
#Author: Denis Vida, denis.vida@gmail.com

# This file is part of the Astrometry.net suite.
# Copyright 2011 Dustin Lang.

# The Astrometry.net suite is free software; you can redistribute
# it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, version 2.

# The Astrometry.net suite is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with the Astrometry.net suite ; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA


import urllib
import os

prefix = 'http://cdsarc.u-strasbg.fr/ftp/cats/aliases/U/UCAC3/UCAC3/'

for i in range(1, 361):
	name = 'z'+str(i).zfill(3)+'.bz2'
	url_name = prefix+name

	print 'Downloading: '+url_name
	ucac_file = urllib.URLopener()
	ucac_file.retrieve(url_name, name)
