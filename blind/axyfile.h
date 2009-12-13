/*
 This file is part of the Astrometry.net suite.
 Copyright 2009 Dustin Lang.

 The Astrometry.net suite is free software; you can redistribute
 it and/or modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation, version 2.

 The Astrometry.net suite is distributed in the hope that it will be
 useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the GNU
 General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with the Astrometry.net suite ; if not, write to the Free Software
 Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA	 02110-1301 USA
 */

#ifndef AXYFILE_H
#define AXYFILE_H

#include "bl.h"

/*
 Shared between 'augment-xylist' and 'backend': description of a field
 to be solved.

 Uses FITS header cards:

 IMAGEW
 IMAGEH

 ANPOSERR -- positional error; sp->verify_pix
 ANCTOL   -- code tolerance; sp->codetol
 ANDISTR  -- distractor ratio; sp->distractor_ratio

 ANSOLVED -- solved output filename; bp
 ANSOLVIN -- solved input filename; bp
 ANMATCH  -- match file; bp
 ANRDLS   -- RA,Dec of index stars
 ANSCAMP  -- Scamp catalog
 ANWCS    -- WCS FITS header
 ANCORR   -- correlation between sources and index
 ANCANCEL -- cancel file.

 ANTAG#   (string) Tag-along columns to copy from index into index-rdls
 ANTAGALL (bool)   Tag-along all columns

 ANXCOL   -- X column name
 ANYCOL   -- Y column name

 ANTLIM   -- time limit (seconds)
 ANCLIM   -- CPU time limit (seconds)
 ANODDSPR -- odds ratio to print
 ANODDSKP -- odds ratio to keep
 ANODDSSL -- odds ratio to solve
 ANODDSBL -- odds ratio to bail
 ANODDSST -- odds ratio to stop looking further
 
 ANAPPDEF  (bool)   include default image scales
 ANPARITY  (string) "NEG"/"POS"
 ANTWEAK   (bool)   tweak?
 ANTWEAKO  (int)    tweak polynomial order
 ANQSFMIN  (float)  minimum quad size fraction (of image size)
 ANQSFMAX  (float)  maximum quad size fraction (of image size)

 ANCRPIXC  (bool)   set CRPIX to image center
 ANCRPIX1  (float)  set CRPIX to...
 ANCRPIX2  (float)  set CRPIX to...

 ANERA     (float)  estimated RA (deg)
 ANEDEC    (float)  estimated DEC (deg)
 ANERAD    (float)  estimated field radius (deg)

 ANAPPL#   (float)  #=1,2,... image scale, lower bound, arcsec/pixel
 ANAPPU#   (float)  #=1,2,... image scale, upper bound, arcsec/pixel
 ANDPL#    (int)    #=1,2,... depth (image obj #), lower bound, >= 1
 ANDPU#    (int)    #=1,2,... depth (image obj #), upper bound, >= min
 ANFDL#    (int)    #=1,2,... field (FITS extension), lower, >= 1
 ANFDU#    (int)    #=1,2,... field (FITS extension), upper, >= min
 ANFD#     (int)    #=1,2,... field (FITS extension), single field, >= 1.

 ANW#PIX1  (float)  #=1,2,... WCS to verify.
 ANW#PIX2  (float)  (crpix)
 ANW#VAL1  (float)  (crval)
 ANW#VAL2  (float)
 ANW#CD11  (float)  (cd matrix)
 ANW#CD12  (float)
 ANW#CD21  (float)
 ANW#CD22  (float)
 ANW#SAO   (int)    SIP order, forward
 ANW#A##   (float)  ## = (i,j)  SIP coefficients
 ANW#B##   (float)  ## = (i,j)  
 ANW#SAPO  (int)    SIP order, inverse
 ANW#AP##  (float)  ## = (i,j)  SIP coefficients
 ANW#BP##  (float)  ## = (i,j)  

 ANRUN     (bool)   Go.

 **/

struct axyfile {
	/*
	 // contains ranges of depths as pairs of ints.
	 il* depths;
	 bool include_default_scales;
	 double ra_center;
	 double dec_center;
	 double search_radius;
	 bool use_radec_center;
	 blind_t bp;
	 */
};
typedef struct axyfile axyfile_t;



#endif
