/*
  This file is part of the Astrometry.net suite.
  Copyright 2007 Dustin Lang.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation, version 2.

  The Astrometry.net suite is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the Astrometry.net suite ; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
*/
#include <math.h>
#include <sys/param.h>

#include "sip-utils.h"

static double fmod_pos(double a, double b) {
    double fm = fmod(a, b);
    if (fm < 0.0)
        fm += b;
    return fm;
}

static double shift(double ra) {
    return fmod_pos(ra + 180.0, 360.0);
}

static double unshift(double ra) {
    return fmod_pos(ra - 180.0, 360.0);
}

void get_radec_bounds(sip_t* wcs, int stepsize,
                      double* pramin, double* pramax,
                      double* pdecmin, double* pdecmax) {
    double ramin, ramax, decmin, decmax;
    int i, side;
    // Walk the perimeter of the image in steps of stepsize pixels
    // to find the RA,Dec min/max.
    int W = wcs->wcstan.imagew;
    int H = wcs->wcstan.imageh;
    {
        int offsetx[] = { stepsize, W, W, 0 };
        int offsety[] = { 0, 0, H, H };
        int stepx[] = { +stepsize, 0, -stepsize, 0 };
        int stepy[] = { 0, +stepsize, 0, -stepsize };
        int Nsteps[] = { (W/stepsize)-1, H/stepsize, W/stepsize, H/stepsize };
        double lastra;
        bool wrap = FALSE;

        /*
         We handle RA wrap-around in a hackish way here: if we detect wrap-around,
         we just shift the RA values by 180 degrees so that MIN() and MAX() still
         work, then shift the resulting min and max values back by 180 at the end.
         */

        sip_pixelxy2radec(wcs, 0, 0, &lastra, &decmin);
        ramin = ramax = lastra;
        decmax = decmin;

        for (side=0; side<4; side++) {
            for (i=0; i<Nsteps[side]; i++) {
                double ra, dec;
                int x, y;
                x = offsetx[side] + i * stepx[side];
                y = offsety[side] + i * stepy[side];
                sip_pixelxy2radec(wcs, x, y, &ra, &dec);

                decmin = MIN(decmin, dec);
                decmax = MAX(decmax, dec);

                // Did we just walk over the RA wrap-around line?
                if (!wrap &&
                    (((lastra < 90) && (ra > 270)) ||
                     ((lastra > 270) && (ra < 90)))) {
                    wrap = TRUE;
                    ramin = shift(ramin);
                    ramax = shift(ramax);
                }

                if (wrap)
                    ra = shift(ra);

                ramin = MIN(ramin, ra);
                ramax = MAX(ramax, ra);

                lastra = ra;
            }
        }
        if (wrap) {
            ramin = unshift(ramin);
            ramax = unshift(ramax);
            if (ramin > ramax)
                ramax += 360.0;
        }
    }
    if (pramin) *pramin = ramin;
    if (pramax) *pramax = ramax;
    if (pdecmin) *pdecmin = decmin;
    if (pdecmax) *pdecmax = decmax;
}


