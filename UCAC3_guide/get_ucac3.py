#Script for automatically downloading UCAC3 catalogue
#Last revision: Dec 15, 2014
#Author: Denis Vida, denis.vida@gmail.com
import urllib
import os

prefix = 'http://cdsarc.u-strasbg.fr/ftp/cats/aliases/U/UCAC3/UCAC3/'

for i in range(1, 361):
	name = 'z'+str(i).zfill(3)+'.bz2'
	url_name = prefix+name

	print 'Downloading: '+url_name
	ucac_file = urllib.URLopener()
	ucac_file.retrieve(url_name, name)
