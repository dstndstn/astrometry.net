/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef UCAC4_H
#define UCAC4_H

#include <stdint.h>

#define UCAC4_RECORD_SIZE 78

struct ucac4_entry {
    // (in brackets are the name, format, and units in the UCAC4 data files)
    // [degrees] (ra, I4: mas)
    double ra;
    // [degrees] (spd, I4: mas)
    double dec;


    // [degrees] (sigra, I1, mas)
    // error in RA*cos(Dec)
    float sigma_ra;
    // [degrees] (sigdc, I1, mas)
    float sigma_dec;
    /* The range of values sigra and sigdc is 1 to 255 which is represented
     as a signed 1-byte integer (range -127 to 127); thus add 128 to the
     integer number found in the data file.  There is no 0 mas value;
     data less than 1 mas have been set to 1 mas.  Original data larger
     than 255 mas have been set to 255.  
     */   

    // fit model mag
    // [mag] (magm, I2: millimag)
    // value 20.000 is used to flag errors
    float mag;

    // aperture mag
    // [mag] (maga, I2: millimag)
    // value 20.000 is used to flag errors
    float apmag;

    // [mag] (sigmag, I1: 1/100 mag)
    float mag_err;

    // (objt, I1)
    /*
     0 = good, clean star (from MPOS), no known problem
     1 = largest flag of any image = near overexposed star (from MPOS)
     2 = largest flag of any image = possible streak object (from MPOS)
     3 = high proper motion (HPM) star, match with external PM file (MPOS)
     4 = actually use external HPM data instead of UCAC4 observ.data
     (accuracy of positions varies between catalogs)
     5 = poor proper motion solution, report only CCD epoch position
     6 = substitute poor astrometric results by FK6/Hip/Tycho-2 data
     7 = added supplement star (no CCD data) from FK6/Hip/Tycho-2 data,
     and 2 stars added from high proper motion surveys
     8 = high proper motion solution in UCAC4, star not matched with PPMXL
     9 = high proper motion solution in UCAC4, discrepant PM to PPMXL
     (see discussion of flags 8,9 in redcution section 2e above)
     */
    int8_t objtype;

    // (cdf, I1)
    /*
     The cdf flag is a combined double star flag used to indicate 
     the type/quality of double star fit.  It is a combination of 2 flags,
     cdf = 10 * dsf + dst  with the following meaning:

     dsf = double star flag = overall classification
     0 = single star
     1 = component #1 of "good" double star
     2 = component #2 of "good" double star
     3 = blended image

     dst = double star type, from pixel data image profile fits,
     largest value of all images used for this star
     0 = no double star, not sufficient #pixels or elongation
     to even call double star fit subroutine
     1 = elongated image but no more than 1 peak detected
     2 = 2 separate peaks detected -> try double star fit
     3 = secondary peak found on each side of primary
     4 = case 1 after successful double fit (small separ. blended image)
     5 = case 2 after successful double fit (most likely real double)
     6 = case 3 after successful double fit (brighter secondary picked)
     */
    int8_t doublestar;

    // (na1: I1)
    // total # of CCD images of this star
    uint8_t navail;

    // (nu1: I1)
    // # of CCD images used for this star
    uint8_t nused;

    // (cu1: I1)
    // total numb. catalogs (epochs) used for proper motion
    uint8_t nmatch;

    // Central epoch for mean RA/Dec
    // [yr] (cepra/cepdc, I2, 0.01 yr - 1900.00)
    float epoch_ra;
    float epoch_dec;

    // Proper motion in RA*cos(Dec), Dec at central epoch
    // [arcsec/yr] (pmrac/pmdc, I2, 0.1 mas/yr)
    float pm_rac;
    float pm_dec;

    // [arcsec/pr] (sigpmr/sigpmd, I2, 0.1 mas/yr)
    float sigma_pm_ra;
    float sigma_pm_dec;

    // 2MASS pts_key star identifier
    // (pts_key, I4)
    uint32_t twomass_id;

    // 2MASS J mag
    // (j_m, I2, millimag)
    float jmag;

    // 2MASS H mag
    // (h_m, I2, millimag)
    float hmag;

    // 2MASS K_s mag
    // (k_m, I2, millimag)
    float kmag;

    // e2mpho I*1 * 3         2MASS error photom. (1/100 mag)
    float jmag_err;
    float hmag_err;
    float kmag_err;

    // icqflg I*1 * 3         2MASS cc_flg*10 + phot.qual.flag
    /*
     (cc_flg*10 + ph_qual) consisting of the contamination flag (0 to 5) 
     and the photometric quality flag (0 to 8).  

     0 =  cc_flg  2MASS 0, no artifacts or contamination
     1 =  cc_flg  2MASS p, source may be contaminated by a latent image
     2 =  cc_flg  2MASS c, photometric confusion
     3 =  cc_flg  2MASS d, diffraction spike confusion
     4 =  cc_flg  2MASS s, electronic stripe
     5 =  cc_flg  2MASS b, bandmerge confusion

     0 =  no ph_qual flag
     1 =  ph_qual 2MASS X, no valid brightness estimate
     2 =  ph_qual 2MASS U, upper limit on magnitude
     3 =  ph_qual 2MASS F, no reliable estimate of the photometric error
     4 =  ph_qual 2MASS E, goodness-of-fit quality of profile-fit poor
     5 =  ph_qual 2MASS A, valid measurement, [jhk]snr>10 AND [jhk]cmsig<0.10857
     6 =  ph_qual 2MASS B, valid measurement, [jhk]snr> 7 AND [jhk]cmsig<0.15510
     7 =  ph_qual 2MASS C, valid measurement, [jhk]snr> 5 AND [jhk]cmsig<0.21714
     8 =  ph_qual 2MASS D, valid measurement, no [jhk]snr OR [jhk]cmsig req.
	 
     For example icqflg = 05 is decoded to be cc_flg=0, and ph_qual=5, meaning
     no artifacts or contamination from cc_flg and 2MASS qual flag = "A" .
     */
    uint8_t twomass_jflags;
    uint8_t twomass_hflags;
    uint8_t twomass_kflags;

    //  APASS magnitudes B,V,g,r,i
    // (apsm, 5 * I2, millimag)
    float Bmag;
    float Vmag;
    float gmag;
    float rmag;
    float imag;
    // APASS magnitudes error 
    // (apase, 5 * I2, 1/100 millimag)
    float Bmag_err;
    float Vmag_err;
    float gmag_err;
    float rmag_err;
    float imag_err;

    // Yale SPM g-flag*10 c-flag
    // (gcflg, I1)
    /*
     The g-flag from the Yale San Juan first epoch Southern
     Proper Motion data (YSJ1, SPM) has the following meaning:

     0 = no info
     1 = matched with 2MASS extended source list
     2 = LEDA  galaxy
     3 = known QSO

     The c-flag from the Yale San Juan first epoch Southern
     Proper Motion data (YSJ1, SPM) indicates which input catalog
     has been used to identify stars for pipeline processing:

     1 = Hipparcos
     2 = Tycho2
     3 = UCAC2
     4 = 2MASS psc
     5 = 2MASS xsc (extended sources, largely (but not all!) galaxies)
     6 = LEDA  (confirmed galaxies, Paturel et al. 2005)
     7 = QSO   (Veron-Cetty & Veron 2006)
     */
    uint8_t yale_gc_flag;
    
    // Catalog flags
    // (icf, I4)
    /*
     That 4-byte integer has the value:
     icf = icf(1)*10^8 + icf(2)*10^7 + ...  + icf(8)*10 + icf(9)

     The FK6-Hipparcos-Tycho-source-flag has the following meaning:
     (= icf(1))
     0 = not a Hip. or Tycho star
     1 = Hipparcos 1997 version main catalog (not in UCAC4 data files)
     2 = Hipparcos double star annex
     3 = Tycho-2
     4 = Tycho annex 1
     5 = Tycho annex 2
     6 = FK6 position and proper motion (instead of Hipparcos data)
     7 = Hippparcos 2007 solution position and proper motion
     8 = FK6      only PM substit. (not in UCAC4 data)
     9 = Hipparcos 2007, only proper motion substituted

     The catflg match flag is provided for major catalogs used
     in the computation of the proper motions.  Each match is analyzed
     for multiple matches of entries of the 1st catalog to 2nd catalog 
     entries, and the other way around.  Matches are also classified
     by separation and difference in magnitude to arrive at a confidence
     level group.  The flag has the following meaning: 

     0 = star not matched with this catalog
     1 = unique-unique match,  not involving a double star
     2 =  ... same, but involving a flagged double star
     3 = multiple match but unique in high confidence level group, no double
     4 =  ... same, but involving a flagged double star
     5 = closest match, not involving a double, likely o.k. 
     6 =  ... same, but involving a flagged double star
     7 = maybe o.k. smallest sep. match in both directions, no double
     8 =  ... same, but involving a flagged double star
     */
    uint32_t catalog_flags;

    // LEDA galaxy match flag
    // (leda, I1)
    /*
     This flag is either 0 (no match) or contains the log of
     the apparent total diameter for I-band (object size) information
     copied from the LEDA catalog (galaxies).  A size value of less
     than 1 has been rounded up to 1.
     */
    uint8_t leda_flag;

    // 2MASS extend.source flag
    // (x2m, I1)
    /*
     This flag is either 0 (no match) or contains the length of
     the semi-major axis of the fiducial ellipse at the K-band 
     (object size) information copied from the 2MASS extended source
     catalog.  If the size is larger than 127, the flag was set to 127.
     */
    uint8_t twomass_extsource_flag;

    // unique star identification number
    // (rnm, I4)
    /*
     This unique star identification number is between 200001
     and  321640 for Hipparcos stars, and between 1 and 9430 for non-
     Hipparcos stars supplemented to the UCAC4 catalog (no CCD observ.).
     For all other stars this unique star identification number is the
     internal mean-position-file (MPOS) number + 1 million.
     For both the Hipparcos and the supplement stars there is an entry
     on the u4supl.dat file providing more information, including the
     original Hipparcos star number.  Note, there are several thousand
     cases where different UCAC4 stars link to the same Hipparcos star
     number due to resolved binary stars with each component being a 
     separate star entry in UCAC4.
     */
    uint32_t mpos;

    // zone number of UCAC2 
    // (zn2, I2)
    uint16_t ucac2_zone;
    // running record number along UCAC2 zone
    // (rn2, I4)
    uint32_t ucac2_number;
};

typedef struct ucac4_entry ucac4_entry;

int ucac4_parse_entry(ucac4_entry* entry, const void* encoded);

#endif
