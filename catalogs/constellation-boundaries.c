/*
 IAU Constellation boundaries, in J2000, from
 http://vizier.cfa.harvard.edu/viz-bin/VizieR-3?-source=VI/49/bound_20
 */
/*
#
#   VizieR Astronomical Server vizier.cfa.harvard.edu
#    Date: 2014-07-16T14:54:10 [V1.99+ (14-Oct-2013)]
#
#Coosys	J2000:	eq_FK5 J2000
#INFO	votable-version=1.99+ (14-Oct-2013)	
#INFO	-ref=VIZ53c69102128c	
#INFO	-out.max=unlimited	
#INFO	queryParameters=12	
#-oc.form=dec
#-out.max=unlimited
#-nav=cat:VI/49&tab:{VI/49/bound_20}&key:source=VI/49/bound_20&HTTPPRM:&
#-c.eq=J2000
#-c.r=  2
#-c.u=arcmin
#-c.geom=r
#-source=VI/49/bound_20
#-order=I
#-out=RAJ2000
#-out=DEJ2000
#-out=cst
#

#RESOURCE=yCat_6049
#Name: VI/49
#Title: Constellation Boundary Data (Davenhall+ 1989)
#Table	VI_49_bound_20:
#Name: VI/49/bound_20
#Title: Boundaries for J2000
#Column	RAJ2000	(F10.6)	Right ascension in decimal hours (J2000)	[ucd=pos.eq.ra;meta.main]
#Column	DEJ2000	(F11.7)	Declination in degrees (J2000)	[ucd=pos.eq.dec;meta.main]
#Column	cst	(A4)	Constellation abbreviation	[ucd=meta.id.part]
RAJ2000	DEJ2000	cst
deg	deg	
----------	-----------	----
 */

#include "constellation-boundaries.h"
#include "an-bool.h"
#include "bl.h"
#include "starutil.h"

typedef struct {
    double ra;
    double dec;
    int con;
} boundarypoint_t;

static boundarypoint_t boundaries[] = {
#include "constellation-boundaries-data.c"
};

/**
 Returns the "enum constellations" number of the constellation
 containing the given RA,Dec point, or -1 if none such is found.
 */
int constellation_containing(double ra, double dec) {
    int i;
    int N = sizeof(boundaries) / sizeof(boundarypoint_t);
    dl* poly = dl_new(256);
    double xyz[3];
    radecdeg2xyzarr(ra, dec, xyz);

    //printf("%i boundary points, %i constellations (%i to %i)\n", N,
    //CON_FINAL, CON_AND, CON_VUL);
    for (i=0; i<CON_FINAL; i++) {
        int j;
        anbool con_ok;
        con_ok = TRUE;
        dl_remove_all(poly);
        // Find start and end of this constellation, and project
        // boundary points about the target RA,Dec.
        for (j=0; j<N; j++) {
            double xyzc[3];
            double px, py;
            if (boundaries[j].con != i)
                continue;
            radecdeg2xyzarr(boundaries[j].ra, boundaries[j].dec, xyzc);
            if (!star_coords(xyzc, xyz, TRUE, &px, &py)) {
                // the constellation is too far away (on other side of
                // the sky)
                con_ok = FALSE;
                break;
            }
            dl_append(poly, px);
            dl_append(poly, py);
        }
        if (!con_ok)
            continue;
        // Now we have projected all the boundary points of this
        // constellation about the query point, which is (0,0) by
        // definition.  Does the boundary polygon contain (0,0)?
        if (point_in_polygon(0., 0., poly))
            return i;
    }

    return -1;
}


