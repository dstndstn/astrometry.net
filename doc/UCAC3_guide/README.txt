# Last revision: Dec 15, 2014
# Author: Denis Vida, denis.vida@gmail.com
# NOTICE: This guide was developed on CentOS 6.6, but the author sees no reason why it wouldn't work on any other distribution, provided that you have Python and other prerequisites installed.

# This file is part of the Astrometry.net suite.
# Copyright 2014 Denis Vida.

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

-------------------------------------------------------------------------------------
PREPARATION:

0) Install astrometry.net and make sure you can run astrometry.net commands from the shell. Test by running: hpsplit

0.1) Go to astrometry.net folder and run the following commands:
# cd catalogs
# make ucac3tofits 
# mkdir /tmp/UCAC3
# cp ucac3tofits /tmp/UCAC3
# cd /tmp/UCAC3
This will compile the ucac3tofits program, make a working directory in the /tmp and copy the executable to your working directory (where you will be downloading UCAC3 and running the scripts)

0.2) copy get_ucac3.py and build-index.py from this directory into the working directory
# cp /path/to/this/directory/*.py /tmp/UCAC3

-------------------------------------------------------------------------------------
MAIN GUIDE:

1) Download UCAC3 by running (this will take some time):
# python get_ucac3.py

1.1) OPTIONAL: Delete sky areas you don't need (read the UCAC3 table_zones file which list which declination is covered in which file): http://cdsarc.u-strasbg.fr/ftp/cats/aliases/U/UCAC3/UCAC3/table_zones

2) Run this command to convert UCAC3 format to fits tables:
# ./ucac3tofits -N 1 z???.bz2 

3) Run this command to split the fits tables into 12 healpix tiles with 1 deg overlap (-m 1):
# hpsplit -o split-%02i.fits -n 1 -m 1 ucac3_???.fits    

4) To trim out unnecessary FITS table columns, run the following commands:
fitscopy split-00.fits"[col RA;DEC;MAG]" cut-00.fits
fitscopy split-01.fits"[col RA;DEC;MAG]" cut-01.fits
fitscopy split-02.fits"[col RA;DEC;MAG]" cut-02.fits
fitscopy split-03.fits"[col RA;DEC;MAG]" cut-03.fits
fitscopy split-04.fits"[col RA;DEC;MAG]" cut-04.fits
fitscopy split-05.fits"[col RA;DEC;MAG]" cut-05.fits
fitscopy split-06.fits"[col RA;DEC;MAG]" cut-06.fits
fitscopy split-07.fits"[col RA;DEC;MAG]" cut-07.fits
fitscopy split-08.fits"[col RA;DEC;MAG]" cut-08.fits
fitscopy split-09.fits"[col RA;DEC;MAG]" cut-09.fits
fitscopy split-10.fits"[col RA;DEC;MAG]" cut-10.fits
fitscopy split-11.fits"[col RA;DEC;MAG]" cut-11.fits

5) Determine the required scales of index files (refer to the astrometry.net documentation for determining your scales). I am using 1.5°x1.5° FOV images, so I use scales 5, 6 and 7.

6) Edit the build-index.py according to your scale requirements (make other edits if needed). Editing is easy, just change the scale_range variable to whatever you need. After editing and saving, run the script (this will take some time):
# python build-index.py

If everything goes well, you should have 12*[number of different scales] files (in my case I have 36 files, as I have 3 different scales).

7) After building the index files, you need to add an entry to /usr/local/astrometry/etc/astrometry.cfg to use the index files with astrometry.net. Open that file and add:
add_path /tmp/UCAC3
autoindex

8) You can delete all files from the /tmp/UCAC3 folder except those beginning with index-ucac3*

9) Enjoy using your UCAC3 index files!

OPTIONAL - TEST IMAGE
1) I have provided a test image (La Sagra Sky Survey) which will work for scales 5 to 7 and declination range from +24° to +20°. Copy the centu1.jpg image to the directory of your choice and run:
# solve-field centu1.jpg --overwrite --downsample 2 --tweak-order 4

The solved image should have a center in (RA H:M:S, Dec D:M:S) = (07:38:20.142, +22:18:57.895). Congratulations, you made it!

