/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <math.h>
#include <assert.h>

#include "os-features.h"
#include "mathutil.h"
#include "fitsioutils.h"
#include "ioutils.h"
#include "errors.h"
#include "log.h"
#include "an-bool.h"
#include "sip.h"
#include "sip_qfits.h"
#include "sip-utils.h"
#include "starutil.h"
#include "starxy.h"
#include "tweak.h"

#include "anqfits.h"
#include "qfits_rw.h"

#include "fit-wcs.h"

/*
 Scamp's copy of wcslib has  "raw_to_pv" in "proj.c".

 wcslib-4.4.4 has no such thing...


 tanrev()
 called by celrev(), by wcsrev()

 wcsrev()
 - calls linrev()          (pixcrd -> imgcrd)
 - calls celrev(, x=imgcrd,y=imgcrd)
 -   calls tanrev()        (x,y -> phi,theta)
 -     calls raw_to_pv()   (x,y -> xp, yp)
 -   calls sphrev()        (phi,theta -> lng,lat)

 Got imcrd = (-0.0154408,-0.00816145)
 = sip.c (x,y) before -deg2rd

 // lng = 0, lat = 1


 CFHTLS via CVO headers:

 PV1_0   =   6.888383659426E-03 / xi =   PV1_0                       
 PV1_1   =    1.01653233841     /      + PV1_1 * x                   
 PV1_2   =   7.672948165520E-03 /      + PV1_2 * y                   
 PV1_3   =    0.00000000000     /      + PV1_3 * sqrt(x**2 + y**2)   
 PV1_4   =  -1.613906528201E-03 /      + PV1_4 * x**2                
 PV1_5   =  -1.170568723090E-03 /      + PV1_5 * x*y                 
 PV1_6   =  -6.175903023930E-04 /      + PV1_6 * y**2                
 PV1_7   =  -2.467136651059E-02 /      + PV1_7 * x**3                
 PV1_8   =  -1.806292484275E-03 /      + PV1_8 * x**2 * y            
 PV1_9   =  -2.439766180834E-02 /      + PV1_9 * x * y**2            
 PV1_10  =  -4.872349869816E-04 /      + PV1_10* y**3                
 PV2_0   =   5.963468826495E-03 / eta =  PV2_0                       
 PV2_1   =    1.01450676752     /      + PV2_1 * y                   
 PV2_2   =   7.677145522017E-03 /      + PV2_2 * x                   
 PV2_3   =    0.00000000000     /      + PV2_3 * sqrt(x**2 + y**2)   
 PV2_4   =  -1.353520666662E-03 /      + PV2_4 * y**2                
 PV2_5   =  -1.247925715556E-03 /      + PV2_5 * y*x                 
 PV2_6   =  -5.742047327244E-04 /      + PV2_6 * x**2                
 PV2_7   =  -2.435753005264E-02 /      + PV2_7 * y**3                
 PV2_8   =  -1.842813673530E-03 /      + PV2_8 * y**2 * x            
 PV2_9   =  -2.444782516561E-02 /      + PV2_9 * y * x**2            
 PV2_10  =  -4.717653697970E-04 /      + PV2_10* x**3                


 a = prj->p+100;
 b = prj->p;
 xp = *(a++);
 xp += *(a++)*x;
 */

sip_t* wcs_pv2sip_header(qfits_header* hdr,
                         double* xy, int Nxy,
               
                         double stepsize,
                         double xlo, double xhi,
                         double ylo, double yhi,

                         int imageW, int imageH,
                         int order,
                         anbool forcetan,
                         int doshift) {
    double* radec = NULL;
    tan_t tanwcs;
    double x,y, px,py;
    double* rddist = NULL;
    int i, j;
    int nx, ny;
    double xstep, ystep;
    sip_t* sip = NULL;

    /**
     From http://iraf.noao.edu/projects/mosaic/tpv.html

     p = PV1_

     xi' = p0 +
     p1 * xi + p2 * eta + p3 * r +
     p4 * xi^2 + p5 * xi * eta + p6 * eta^2 +
     p7 * xi^3 + p8 * xi^2 * eta + p9 * xi * eta^2 +
     p10 * eta^3 + p11 * r^3 + 
     p12 * xi^4 + p13 * xi^3 * eta + p14 * xi^2 * eta^2 +
     p15 * xi * eta^3 + p16 * eta^4 +
     p17 * xi^5 + p18 * xi^4 * eta + p19 * xi^3 * eta^2 +
     p20 * xi^2 * eta^3 + p21 * xi * eta^4 + p22 * eta^5 + p23 * r^5 +
     p24 * xi^6 + p25 * xi^5 * eta + p26 * xi^4 * eta^2 +
     p27 * xi^3 * eta^3 + p28 * xi^2 * eta^4 + p29 * xi * eta^5 +
     p30 * eta^6
     p31 * xi^7 + p32 * xi^6 * eta + p33 * xi^5 * eta^2 +
     p34 * xi^4 * eta^3 + p35 * xi^3 * eta^4 + p36 * xi^2 * eta^5 +
     p37 * xi * eta^6 + p38 * eta^7 + p39 * r^7

     p = PV2_
     eta' = p0 +
     p1 * eta + p2 * xi + p3 * r +
     p4 * eta^2 + p5 * eta * xi + p6 * xi^2 +
     p7 * eta^3 + p8 * eta^2 * xi + p9 * eta * xi^2 + p10 * xi^3 +
     p11 * r^3 +
     p12 * eta^4 + p13 * eta^3 * xi + p14 * eta^2 * xi^2 +
     p15 * eta * xi^3 + p16 * xi^4 +
     p17 * eta^5 + p18 * eta^4 * xi + p19 * eta^3 * xi^2 +
     p20 * eta^2 * xi^3 + p21 * eta * xi^4 + p22 * xi^5 +
     p23 * r^5 +
     p24 * eta^6 + p25 * eta^5 * xi + p26 * eta^4 * xi^2 +
     p27 * eta^3 * xi^3 + p28 * eta^2 * xi^4 + p29 * eta * xi^5 +
     p30 * xi^6
     p31 * eta^7 + p32 * eta^6 * xi + p33 * eta^5 * xi^2 +
     p34 * eta^4 * xi^3 + p35 * eta^3 * xi^4 + p36 * eta^2 * xi^5 +
     p37 * eta * xi^6 + p38 * xi^7 + p39 * r^7

     Note the "cross-over" -- the xi' powers are in terms of xi,eta
     while the eta' powers are in terms of eta,xi.
     */

    //           1  x  y  r x2 xy y2 x3 x2y xy2 y3 r3 x4 x3y x2y2 xy3 y4
    //          x5 x4y x3y2 x2y3 xy4 y5 r5 x6 x5y x4y2, x3y3 x2y4 xy5 y6
    //          x7 x6y x5y2 x4y3 x3y4 x2y5 xy6 y7 r7
    int xp[] = {
        0,
        1, 0, 0,
        2, 1, 0,
        3, 2, 1, 0, 0,
        4, 3, 2, 1, 0,
        5, 4, 3, 2, 1, 0, 0,
        6, 5, 4, 3, 2, 1, 0,
        7, 6, 5, 4, 3, 2, 1, 0, 0};
    int yp[] = {
        0,
        0, 1, 0,
        0, 1, 2,
        0, 1, 2, 3, 0,
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4, 5, 0,
        0, 1, 2, 3, 4, 5, 6,
        0, 1, 2, 3, 4, 5, 6, 7, 0};
    int rp[] = {
        0,
        0, 0, 1,
        0, 0, 0,
        0, 0, 0, 0, 3,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 5,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 7};
    double xpows[8];
    double ypows[8];
    double rpows[8];
    double pv1[40];
    double pv2[40];
    double r;

    char* ct;
    ct = fits_get_dupstring(hdr, "CTYPE1");
    if ((ct && streq(ct, "RA---TPV")) || forcetan) {
        // http://iraf.noao.edu/projects/ccdmosaic/tpv.html
        logmsg("Replacing CTYPE1 = %s header with RA---TAN\n", ct);
        fits_update_value(hdr, "CTYPE1", "RA---TAN");
    }
    ct = fits_get_dupstring(hdr, "CTYPE2");
    if ((ct && streq(ct, "DEC--TPV")) || forcetan) {
        logmsg("Replacing CTYPE2 = %s header with DEC--TAN\n", ct);
        fits_update_value(hdr, "CTYPE2", "DEC--TAN");
    }

    tan_read_header(hdr, &tanwcs);

    if (log_get_level() >= LOG_VERB) {
        printf("Read TAN header:\n");
        tan_print(&tanwcs);
    }

    if (imageW && (imageW != tanwcs.imagew)) {
        logmsg("Overriding image width %f with user-specified %i\n",
               tanwcs.imagew, imageW);
        tanwcs.imagew = imageW;
    }
    if (imageH && (imageH != tanwcs.imageh)) {
        logmsg("Overriding image height %f with user-specified %i\n",
               tanwcs.imageh, imageH);
        tanwcs.imageh = imageH;
    }

    for (i=0; i<sizeof(pv1)/sizeof(double); i++) {
        char key[10];
        double defaultval;

        if (i == 1) {
            defaultval = 1.0;
        } else {
            defaultval = 0.0;
        }
        sprintf(key, "PV1_%i", i);
        pv1[i] = qfits_header_getdouble(hdr, key, defaultval);
        sprintf(key, "PV2_%i", i);
        pv2[i] = qfits_header_getdouble(hdr, key, defaultval);
    }


    // choose grid for evaluating TAN-PV WCS
    if (xlo == 0 && xhi == 0) {
        xlo = 1.;
        xhi = tanwcs.imagew;
    }
    if (ylo == 0 && yhi == 0) {
        ylo = 1.;
        yhi = tanwcs.imageh;
    }

    assert(xhi >= xlo);
    assert(yhi >= ylo);

    if (stepsize == 0)
        stepsize = 100.;
    nx = MAX(order+2, round((xhi - xlo)/stepsize));
    ny = MAX(order+2, round((yhi - ylo)/stepsize));
    xstep = (xhi - xlo) / (double)(nx - 1);
    ystep = (yhi - ylo) / (double)(ny - 1);

    logverb("Stepping from x = %g to %g, steps of %g\n", xlo, xhi, xstep);
    logverb("Stepping from y = %g to %g, steps of %g\n", ylo, yhi, ystep);

    Nxy = nx * ny;

    if (xy == NULL) {
        int k = 0;
        xy = malloc(Nxy * 2 * sizeof(double));
        for (i=0; i<ny; i++) {
            y = ylo + i*ystep;
            for (j=0; j<nx; j++) {
                x = xlo + j*xstep;
                //if (i == 0)
                //printf("x=%f\n", x);
                xy[k] = x;
                k++;
                xy[k] = y;
                k++;
            }
            //printf("y=%f\n", y);
        }
        assert(k == (Nxy*2));
    }

    // distorted RA,Dec
    rddist = malloc(2 * Nxy * sizeof(double));

    for (j=0; j<Nxy; j++) {
        double ix = xy[2*j+0];
        double iy = xy[2*j+1];

        tan_pixelxy2iwc(&tanwcs, ix, iy, &x, &y);
        // "x,y" here are most commonly known as "xi, eta".
        r = sqrt(x*x + y*y);
        // compute powers of x,y
        xpows[0] = ypows[0] = rpows[0] = 1.0;
        for (i=1; i<sizeof(xpows)/sizeof(double); i++) {
            xpows[i] = xpows[i-1]*x;
            ypows[i] = ypows[i-1]*y;
            rpows[i] = rpows[i-1]*r;
        }
        px = py = 0;
        for (i=0; i<sizeof(xp)/sizeof(int); i++) {
            px += pv1[i] * xpows[xp[i]] * ypows[yp[i]] * rpows[rp[i]];
            // here's the "cross-over" mentioned above
            py += pv2[i] * ypows[xp[i]] * xpows[yp[i]] * rpows[rp[i]];
        }

        // Note that the PV terms *include* a linear term, so no need
        // to re-add x,y to px,py.
        tan_iwc2radec(&tanwcs, px, py,
                      rddist + 2*j, rddist + 2*j + 1);
    }

    sip = malloc(sizeof(sip_t));
    assert(sip);
    {
        double* starxyz;
        Unused int rtn;
        starxyz = malloc(3 * Nxy * sizeof(double));
        for (i=0; i<Nxy; i++)
            radecdegarr2xyzarr(rddist + i*2, starxyz + i*3);
        memset(sip, 0, sizeof(sip_t));
        rtn = fit_sip_coefficients(starxyz, xy, NULL, Nxy,
                                   &tanwcs, order, order, sip);
        assert(rtn == 0);
        if (log_get_level() >= LOG_VERB) {
            printf("Fit SIP:\n");
            sip_print(sip);
        }

        if (doshift) {
            // Push the A_0_0 and B_0_0 terms -- pixel offsets -- into the CRVAL,
            // indirectly, by evaluating the SIP at the CRPIX and setting CRVAL to that!
            double ra, dec;
            sip_t sip2;
            double DPIX = 50;
            double xi, eta, xi2, eta2;
            double cd11, cd21, cd12, cd22;

            x = sip->wcstan.crpix[0];
            y = sip->wcstan.crpix[1];
            sip_pixelxy2radec(sip, x, y, &ra, &dec);
            logverb("CRPIX -> SIP -> RA,DEC (%.6f, %.6f)\n", ra, dec);
            logverb("              vs CRVAL (%.6f, %.6f)\n", sip->wcstan.crval[0], sip->wcstan.crval[1]);
            sip->wcstan.crval[0] = ra;
            sip->wcstan.crval[1] = dec;
            rtn = fit_sip_coefficients(starxyz, xy, NULL, Nxy,
                                       &(sip->wcstan), order, order, sip);
            assert(rtn == 0);
            if (log_get_level() >= LOG_VERB) {
                printf("Fit SIP after moving CRVAL:\n");
                sip_print(sip);
            }

            // Push the A_1_0, A_0_1, B_1_0, B_0_1 terms into the CD matrix,
            // indirectly, by zeroing out the other SIP terms and then evaluating at
            // CRPIX +- 50 pixels!
            sip_copy(&sip2, sip);
            // zero out terms order 2 and above
            for (int i=0; i<SIP_MAXORDER; i++) {
                for (int j=0; j<SIP_MAXORDER; j++) {
                    if (i + j < 2)
                        continue;
                    sip2.a[i][j] = 0.;
                    sip2.b[i][j] = 0.;
                }
            }
            // Evaluate CRPIX +- 50 pixels in X, in IWC -- which are what the CD matrix
            // rotates into
            x = sip->wcstan.crpix[0] - DPIX;
            y = sip->wcstan.crpix[1];
            sip_pixelxy2iwc(&sip2, x, y, &xi, &eta);
            x = sip->wcstan.crpix[0] + DPIX;
            sip_pixelxy2iwc(&sip2, x, y, &xi2, &eta2);
            cd11 = (xi2  - xi ) / (2.*DPIX);
            cd21 = (eta2 - eta) / (2.*DPIX);
            logverb("Changing X: d_xi %g, d_eta %g\n", cd11, cd21);

            x = sip->wcstan.crpix[0];
            y = sip->wcstan.crpix[1] - DPIX;
            sip_pixelxy2iwc(&sip2, x, y, &xi, &eta);
            y = sip->wcstan.crpix[1] + DPIX;
            sip_pixelxy2iwc(&sip2, x, y, &xi2, &eta2);
            cd12 = (xi2  - xi ) / (2.*DPIX);
            cd22 = (eta2 - eta) / (2.*DPIX);
            logverb("Changing Y: d_xi %g, d_eta %g\n", cd12, cd22);
            logverb("Updating CD from [%g, %g, %g, %g]\n",
                    sip->wcstan.cd[0][0], sip->wcstan.cd[0][1],
                    sip->wcstan.cd[1][0], sip->wcstan.cd[1][1]);
            logverb("              to [%g, %g, %g, %g]\n", cd11, cd12, cd21, cd22);

            sip->wcstan.cd[0][0] = cd11;
            sip->wcstan.cd[0][1] = cd12;
            sip->wcstan.cd[1][0] = cd21;
            sip->wcstan.cd[1][1] = cd22;
            rtn = fit_sip_coefficients(starxyz, xy, NULL, Nxy,
                                       &(sip->wcstan), order, order, sip);
            assert(rtn == 0);
            if (log_get_level() >= LOG_VERB) {
                printf("Fit SIP after changing CD:\n");
                sip_print(sip);
            }

            if (log_get_level() >= LOG_VERB) {
                double dist = 0.0;
                for (int i=0; i<Nxy; i++) {
                    double xyz[3];
                    sip_pixelxy2xyzarr(sip, xy[2*i+0], xy[2*i+1], xyz);
                    dist += sqrt(square(xyz[0] - starxyz[3*i+0]) +
                                 square(xyz[1] - starxyz[3*i+1]) +
                                 square(xyz[2] - starxyz[3*i+2]));
                }
                logverb("Fitting SIP: error is %g arcsec\n", rad2arcsec(dist / Nxy));
            }

            // Set the low-order terms to exactly zero, assuming that the above steps made
            // them close enough!
            sip->a[0][0] = 0.;
            sip->a[0][1] = 0.;
            sip->a[1][0] = 0.;
            sip->b[0][0] = 0.;
            sip->b[0][1] = 0.;
            sip->b[1][0] = 0.;

            if (log_get_level() >= LOG_VERB) {
                double dist = 0.0;
                for (int i=0; i<Nxy; i++) {
                    double xyz[3];
                    sip_pixelxy2xyzarr(sip, xy[2*i+0], xy[2*i+1], xyz);
                    dist += sqrt(square(xyz[0] - starxyz[3*i+0]) +
                                 square(xyz[1] - starxyz[3*i+1]) +
                                 square(xyz[2] - starxyz[3*i+2]));
                }
                logverb("Zeroing out A,B terms below order 2: error is %g arcsec\n", rad2arcsec(dist / Nxy));
            }
        }

        // FIXME? -- use xlo,xhi,ylo,yhi here??  Not clear.
        sip_compute_inverse_polynomials(sip, 0, 0, 0, 0, 0, 0);

        if (log_get_level() >= LOG_VERB) {
            printf("Fit SIP inverse polynomials:\n");
            sip_print(sip);
        }
        free(starxyz);
    }

    free(rddist);
    free(radec);
    return sip;
}



/**
 Evaluates the given TAN-TPV WCS header on a grid of points,
 fitting a SIP distortion solution to it.

 The grid can be specified by either:

 double* xy, int Nxy

 double stepsize=100, double xlo=0, double xhi=0, double ylo=0, double yhi=0

 xlo and xhi, if both 0, default to 1. and the WCS width
 ylo and yhi, if both 0, default to 1. and the WCS height

 The number of steps is chosen to be the closest step size to split the range
 xlo to xhi into an integer number of steps.

 imageW and imageH, if non-zero, override the image width read from the WCS,
 and ALSO the WCS width/height mentioned above.

 */
int wcs_pv2sip(const char* wcsinfn, int ext,
               const char* wcsoutfn,
               anbool scamp_head_file,

               double* xy, int Nxy,
               
               double stepsize,
               double xlo, double xhi,
               double ylo, double yhi,

               int imageW, int imageH,
               int order,
               anbool forcetan,
               int doshift) {

    qfits_header* hdr = NULL;
    sip_t* sip = NULL;
    int rtn = -1;

    if (scamp_head_file) {
        size_t sz = 0;
        char* txt;
        char* prefix;
        int np;
        int nt;
        unsigned char* txthdr;
        sl* lines;
        int i;
        txt = file_get_contents(wcsinfn, &sz, TRUE);
        if (!txt) {
            ERROR("Failed to read file %s", wcsinfn);
            goto bailout;
        }
        lines = sl_split(NULL, txt, "\n");
        prefix =
            "SIMPLE  =                    T / Standard FITS file                             "
            "BITPIX  =                    8 / ASCII or bytes array                           "
            "NAXIS   =                    0 / Minimal header                                 "
            "EXTEND  =                    T / There may be FITS ext                          "
            "WCSAXES =                    2 /                                                ";
        np = strlen(prefix);
        nt = np + FITS_LINESZ * sl_size(lines);
        txthdr = malloc(nt);
        memset(txthdr, ' ', np + FITS_LINESZ * sl_size(lines));
        memcpy(txthdr, prefix, np);
        for (i=0; i<sl_size(lines); i++)
            memcpy(txthdr + np + i*FITS_LINESZ, sl_get(lines, i), strlen(sl_get(lines, i)));
        sl_free2(lines);
        hdr = qfits_header_read_hdr_string(txthdr, nt);
        free(txthdr);
        free(txt);
    } else {
        hdr = anqfits_get_header2(wcsinfn, ext);
    }
    if (!hdr) {
        ERROR("Failed to read header: file %s, ext %i\n", wcsinfn, ext);
        goto bailout;
    }

    sip = wcs_pv2sip_header(hdr, xy, Nxy, stepsize, xlo, xhi, ylo, yhi,
                            imageW, imageH, order, forcetan, doshift);
    if (!sip) {
        goto bailout;
    }
    sip_write_to_file(sip, wcsoutfn);

    rtn = 0;

 bailout:
    qfits_header_destroy(hdr);
    return rtn;
}

/*
 xp = *(a++);     //a0
 xp += *(a++)*x; //a1
 xp += *(a++)*y; //a2
 xp += *(a++)*r; //a3
 xp += *(a++)*(x2=x*x); //a4
 xp += *(a++)*(xy=x*y); //a5
 xp += *(a++)*y2; //a6
 xp += *(a++)*(x3=x*x2); //a7
 xp += *(a++)*x2*y; //a8
 xp += *(a++)*x*y2; //a9
 xp += *(a++)*y3; //a10
 xp += *(a++)*(r3=r*r*r); //a11
 xp += *(a++)*(x4=x2*x2); //a12
 xp += *(a++)*x3*y; //a13
 xp += *(a++)*x2*y2; //a14
 xp += *(a++)*x*y3; //a15
 xp += *(a++)*y4; //a16
 xp += *(a++)*(x5=x4*x); //a17
 xp += *(a++)*x4*y; //a18
 xp += *(a++)*x3*y2; //a19
 xp += *(a++)*x2*y3; //a20
 xp += *(a++)*x*y4; //a21
 xp += *(a++)*y5; //a22
 xp += *(a++)*(r5=r3*r*r); //a23
 xp += *(a++)*(x6=x5*x); //a24
 xp += *(a++)*x5*y; //a25
 xp += *(a++)*x4*y2; //a26
 xp += *(a++)*x3*y3; //a27
 xp += *(a++)*x2*y4; //a28
 xp += *(a++)*x*y5; //a29
 xp += *(a++)*y6; //a30
 xp += *(a++)*(x7=x6*x); //a31
 xp += *(a++)*x6*y; //a32
 xp += *(a++)*x5*y2; //a33
 xp += *(a++)*x4*y3; //a34
 xp += *(a++)*x3*y4; //a35
 xp += *(a++)*x2*y5; //a36
 xp += *(a++)*x*y6; //a37
 xp += *(a++)*y7; //a38
 xp += *a*(r7=r5*r*r); //a39
 */


