/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <math.h>
#include <stdio.h>
#include <assert.h>

#include "os-features.h"
#include "cutest.h"
#include "starutil.h"
#include "healpix.h"
#include "healpix-utils.h"
#include "bl.h"

static double square(double x) {
    return x*x;
}

void test_hp_rangesearch(CuTest* ct) {
    double ra = 120.12998471831297;
    double dec = 37.72245880730694;
    double rad = 0.2242483527520963;
    int nside = 8;
    il* hps = il_new(4);
    int i;
    healpix_rangesearch_radec(ra, dec, rad, nside, hps);
    for (i=0; i<il_size(hps); i++) {
        int hp = il_get(hps, i);
        printf("Healpix %i\n", hp);
    }
    CuAssertIntEquals(ct, il_size(hps), 2);
    CuAssertIntEquals(ct, il_get(hps,0), 84);
    CuAssertIntEquals(ct, il_get(hps,1), 85);
}

void test_side_length(CuTest* ct) {
    double hp;
    double len = healpix_side_length_arcmin(1);
    CuAssertDblEquals(ct, 3517.9, len, 0.1);
    hp = healpix_nside_for_side_length_arcmin(len);
    CuAssertDblEquals(ct, 1.0, hp, 0.001);

    len = healpix_side_length_arcmin(2);
    CuAssertDblEquals(ct, 1758.969, len, 0.001);
    hp = healpix_nside_for_side_length_arcmin(len);
    CuAssertDblEquals(ct, 2.0, hp, 0.001);
}

static void add_plot_xyz_point(double* xyz) {
    double ra,dec;
    xyzarr2radecdeg(xyz, &ra, &dec);
    fprintf(stderr, "xp.append(%g)\n", ra);
    fprintf(stderr, "yp.append(%g)\n", dec);
}

static void add_plot_point(int hp, int nside, double dx, double dy) {
    double xyz[3];
    healpix_to_xyzarr(hp, nside, dx, dy, xyz);
    add_plot_xyz_point(xyz);
}

void plot_point(int hp, int nside, double dx, double dy, char* style) {
    fprintf(stderr, "xp=[]\n");
    fprintf(stderr, "yp=[]\n");
    add_plot_point(hp, nside, dx, dy);
    fprintf(stderr, "plot(xp, yp, '%s')\n", style);
}

void plot_xyz_point(double* xyz, char* style) {
    fprintf(stderr, "xp=[]\n");
    fprintf(stderr, "yp=[]\n");
    add_plot_xyz_point(xyz);
    fprintf(stderr, "plot(xp, yp, '%s')\n", style);
}

static void plot_hp_boundary(int hp, int nside, double start, double step, char* style) {
    double dx, dy;
    fprintf(stderr, "xp=[]\n");
    fprintf(stderr, "yp=[]\n");
    dy = 0.0;
    for (dx=start; dx<=1.0; dx+=step)
        add_plot_point(hp, nside, dx, dy);
    dx = 1.0;
    for (dy=start; dy<=1.0; dy+=step)
        add_plot_point(hp, nside, dx, dy);
    dy = 1.0;
    for (dx=1.0-start; dx>=0.0; dx-=step)
        add_plot_point(hp, nside, dx, dy);
    dx = 0.0;
    for (dy=1.0-start; dy>=0.0; dy-=step)
        add_plot_point(hp, nside, dx, dy);
    dy = 0.0;
    add_plot_point(hp, nside, dx, dy);
    fprintf(stderr, "xp,yp = wrapxy(xp,yp)\nplot(xp, yp, '%s')\n", style);
}

static void hpmap(int nside, const char* fn) {
#if 0
    int nhp;
#endif
    double xyz[3];
    double range;
    int hps[9];
    int i;
    int hp;
    double dx, dy;

    // pick a point on the edge.
    //hp = 8;
    hp = 9;
    dx = 0.95;
    dy = 0.0;
    range = 0.1;
    /*
     hp = 6;
     dx = 0.05;
     dy = 0.95;
     range = 0.1;
     */
    healpix_to_xyzarr(hp, nside, dx, dy, xyz);

    for (i=0; i<12*nside*nside; i++) {
        plot_hp_boundary(i, nside, 0.005, 0.01, "b-");
    }

#if 0
    nhp = healpix_get_neighbours_within_range(xyz, range, hps, nside);
    assert(nhp >= 1);
    assert(nhp <= 9);
#else
    (void)healpix_get_neighbours_within_range(xyz, range, hps, nside);
#endif

    /*
     for (i=0; i<nhp; i++) {
     printf("in range: %i\n", hps[i]);
     plot_hp_boundary(hps[i], nside, 0.005, 0.01, "b-");
     }
     plot_hp_boundary(hp, nside, 0.005, 0.01, "k-");
     plot_point(hp, nside, dx, dy, "r.");
     */

    for (i=0; i<12*nside*nside; i++) {
        fprintf(stderr, "xp=[]; yp=[]\n");
        add_plot_point(i, nside, 0.5, 0.5);
        //fprintf(stderr, "text(xp[0], yp[0], '%i', color='b', horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='w', edgecolor='w'))\n", i);
        fprintf(stderr, "text(xp[0], yp[0], '%i', color='b', horizontalalignment='center', verticalalignment='center')\n", i);
    }

    fprintf(stderr, "axis((360, 0, -90, 90))\n");
    fprintf(stderr, "xlabel('RA (deg)')\n");
    fprintf(stderr, "ylabel('Dec (deg)')\n");
    fprintf(stderr, "title('healpixels: nside=%i')\n", nside);
    fprintf(stderr, "savefig('%s')\n", fn);
    fprintf(stderr, "clf()\n");
}

void testX_make_map(CuTest* ct) {
    fprintf(stderr, "%s", "from pylab import *\n");
    fprintf(stderr, "clf()\n"
            "def wrapxy(x,y):\n"
            "    lastx = x[0]\n"
            "    lasty = y[0]\n"
            "    outx = [lastx]\n"
            "    outy = [lasty]\n"
            "    for xx,yy in zip(x[1:],y[1:]):\n"
            "        if (xx-lastx)**2 + (yy - lasty)**2 > 1.:\n"
            "            if xx < 180:\n"
            "                xx += 360\n"
            "            else:\n"
            "                xx -= 360\n"
            "        outx.append(xx)\n"
            "        outy.append(yy)\n"
            "        lastx = xx\n"
            "        lasty = yy\n"
            "    return (array(outx),array(outy))\n"
            );

    hpmap(1, "hp.png");
    hpmap(2, "hp2.png");
}


int tst_xyztohpf(CuTest* ct,
		 int hp, int nside, double dx, double dy) {
    double x,y,z;
    double outdx, outdy;
    int outhp;
    double outx,outy,outz;
    double dist;
    healpix_to_xyz(hp, nside, dx, dy, &x, &y, &z);
    outhp = xyztohealpixf(x, y, z, nside, &outdx, &outdy);
    healpix_to_xyz(outhp, nside, outdx, outdy, &outx, &outy, &outz);
    dist = sqrt(MAX(0, square(x-outx) + square(y-outy) + square(z-outz)));
    printf("true/computed:\n"
           "hp: %i / %i\n"
           "dx: %.20g / %.20g\n"
           "dy: %.20g / %.20g\n"
           "x:  %g / %g\n"
           "y:  %g / %g\n"
           "z:  %g / %g\n"
           "dist: %g\n\n",
           hp, outhp, dx, outdx, dy, outdy,
           x, outx, y, outy, z, outz, dist);

    if (dist > 1e-6) {
        double a, b;
        double outa, outb;
        a = xy2ra(x,y) / (2.0 * M_PI);
        b = z2dec(z) / (M_PI);
        outa = xy2ra(outx, outy) / (2.0 * M_PI);
        outb = z2dec(outz) / (M_PI);
        fprintf(stderr,
                "plot([%g, %g],[%g, %g],'r.-')\n", a, outa, b, outb);
        fprintf(stderr, 
                "text(%g, %g, \"(%g,%g)\")\n",
                a, b, dx, dy);
    }

    CuAssertIntEquals(ct, 1, (dist < 1e-6)?1:0);
    return (dist > 1e-6);
}

void tEst_xyztohpf(CuTest* ct) {
    double dx, dy;
    int hp;
    int nside;
    double step = 0.1;
    double a, b;
    nside = 1;

    fprintf(stderr, "%s", "from pylab import plot,text,savefig,clf\n");
    fprintf(stderr, "clf()\n");

    /*
     Plot the grid of healpixes with dx,dy=step steps.
     */
    step = 0.25;
    //for (hp=0; hp<12*nside*nside; hp++) {
    for (hp=0; hp<1*nside*nside; hp++) {
        double x,y,z;
        for (dx=0.0; dx<=1.05; dx+=step) {
            fprintf(stderr, "xp=[]\n");
            fprintf(stderr, "yp=[]\n");
            for (dy=0.0; dy<=1.05; dy+=step) {
                healpix_to_xyz(hp, nside, dx, dy, &x, &y, &z);
                a = xy2ra(x,y) / (2.0 * M_PI);
                b = z2dec(z) / (M_PI);
                fprintf(stderr, "xp.append(%g)\n", a);
                fprintf(stderr, "yp.append(%g)\n", b);
            }
            fprintf(stderr, "plot(xp, yp, 'k-')\n");
        }
        for (dy=0.0; dy<=1.05; dy+=step) {
            fprintf(stderr, "xp=[]\n");
            fprintf(stderr, "yp=[]\n");
            for (dx=0.0; dx<=1.0; dx+=step) {
                healpix_to_xyz(hp, nside, dx, dy, &x, &y, &z);
                a = xy2ra(x,y) / (2.0 * M_PI);
                b = z2dec(z) / (M_PI);
                fprintf(stderr, "xp.append(%g)\n", a);
                fprintf(stderr, "yp.append(%g)\n", b);
            }
            fprintf(stderr, "plot(xp, yp, 'k-')\n");
        }
    }

    step = 0.5;
    /*
     Plot places where the conversion screws up.
     */
    for (hp=0; hp<12*nside*nside; hp++) {
        for (dx=0.0; dx<=1.01; dx+=step) {
            for (dy=0.0; dy<=1.01; dy+=step) {
                tst_xyztohpf(ct, hp, nside, dx, dy);
            }
        }
    }
    fprintf(stderr, "savefig('plot.png')\n");

}

static void tst_neighbours(CuTest* ct, int pix, int* true_neigh, int true_nn,
                           int Nside) {
    int neigh[8];
    int nn;
    int i;
    for (i=0; i<8; i++)
        neigh[i] = -1;
    nn = healpix_get_neighbours(pix, neigh, Nside);
    /*
     printf("true(%i) : [ ", pix);
     for (i=0; i<true_nn; i++)
     printf("%u, ", true_neigh[i]);
     printf("]\n");
     printf("got (%i) : [ ", pix);
     for (i=0; i<nn; i++)
     printf("%u, ", neigh[i]);
     printf("]\n");
     */
    CuAssertIntEquals(ct, nn, true_nn);

    for (i=0; i<true_nn; i++)
        CuAssertIntEquals(ct, true_neigh[i], neigh[i]);
}

static void tst_nested(CuTest* ct, int pix, int* true_neigh, int true_nn,
                       int Nside) {
    int i;
    int truexy[8];
    int xypix;

    /*
     printf("nested true(%i) : [ ", pix);
     for (i=0; i<true_nn; i++)
     printf("%u ", true_neigh[i]);
     printf("]\n");
     */

    CuAssert(ct, "true_nn <= 8", true_nn <= 8);
    for (i=0; i<true_nn; i++) {
        truexy[i] = healpix_nested_to_xy(true_neigh[i], Nside);
        CuAssertIntEquals(ct, true_neigh[i], healpix_xy_to_nested(truexy[i], Nside));
    }
    xypix = healpix_nested_to_xy(pix, Nside);
    CuAssertIntEquals(ct, pix, healpix_xy_to_nested(xypix, Nside));

    tst_neighbours(ct, xypix, truexy, true_nn, Nside);
}

void print_node(double z, double phi, int Nside) {
    double ra, dec;
    int hp;
    int nn;
    int neigh[8];
    int k;

    double scale = 10.0;

    ra = phi;
    dec = asin(z);
    while (ra < 0.0)
        ra += 2.0 * M_PI;
    while (ra > 2.0 * M_PI)
        ra -= 2.0 * M_PI;

    // find its healpix.
    hp = radectohealpix(ra, dec, Nside);
    // find its neighbourhood.
    nn = healpix_get_neighbours(hp, neigh, Nside);
    fprintf(stderr, "  N%i [ label=\"%i\", pos=\"%g,%g!\" ];\n", hp, hp,
            scale * ra/M_PI, scale * z);
    for (k=0; k<nn; k++) {
        fprintf(stderr, "  N%i -- N%i\n", hp, neigh[k]);
    }
}

void test_healpix_distance_to_radec(CuTest *ct) {
    double d;
    double rd[2];

    d = healpix_distance_to_radec(4, 1, 0, 0, NULL);
    CuAssertDblEquals(ct, 0, d, 0);
    d = healpix_distance_to_radec(4, 1, 45, 0, NULL);
    CuAssertDblEquals(ct, 0, d, 0);
    d = healpix_distance_to_radec(4, 1, 45+1, 0, NULL);
    CuAssertDblEquals(ct, 1, d, 1e-9);
    d = healpix_distance_to_radec(4, 1, 45+1, 0+1, NULL);
    CuAssertDblEquals(ct, 1.414, d, 1e-3);

    d = healpix_distance_to_radec(4, 1, 45+10, 0, NULL);
    CuAssertDblEquals(ct, 10, d, 1e-9);

    // top corner
    d = healpix_distance_to_radec(4, 1, 0, rad2deg(asin(2.0/3.0)), NULL);
    CuAssertDblEquals(ct, 0, d, 1e-9);

    d = healpix_distance_to_radec(4, 1, 0, 1 + rad2deg(asin(2.0/3.0)), NULL);
    CuAssertDblEquals(ct, 1, d, 1e-9);

    d = healpix_distance_to_radec(4, 1, -45-10, -10, NULL);
    CuAssertDblEquals(ct, 14.106044, d, 1e-6);

    d = healpix_distance_to_radec(10, 1, 225, 5, NULL);
    CuAssertDblEquals(ct, 5, d, 1e-6);

    d = healpix_distance_to_radec(44, 2, 300, -50, NULL);
    CuAssertDblEquals(ct, 3.007643, d, 1e-6);

    d = healpix_distance_to_radec(45, 2, 310, -50, NULL);
    CuAssertDblEquals(ct, 1.873942, d, 1e-6);

    // south-polar hp, north pole.
    d = healpix_distance_to_radec(36, 2, 180, 90, NULL);
    // The hp corner is -41.8 deg; add 90.
    CuAssertDblEquals(ct, 131.810, d, 1e-3);

    // just south of equator to nearly across the sphere
    d = healpix_distance_to_radec(35, 2, 225, 20, NULL);
    // this one actually has the midpoint further than A and B.
    CuAssertDblEquals(ct, 158.189685, d, 1e-6);

    /*
     xyz[0] = -100.0;
     ra = dec = -1.0;
     d = healpix_distance_to_xyz(4, 1, 0, 0, xyz);
     xyzarr2radecdeg(xyz, &ra, &dec);
     CuAssertDblEquals(ct, 0, ra, 0);
     CuAssertDblEquals(ct, 0, dec, 0);
     */
    rd[0] = rd[1] = -1.0;
    d = healpix_distance_to_radec(4, 1, 0, 0, rd);
    CuAssertDblEquals(ct, 0, rd[0], 0);
    CuAssertDblEquals(ct, 0, rd[1], 0);


    /*
     xyz[0] = -100.0;
     ra = dec = -1.0;
     d = healpix_distance_to_xyz(4, 1, 45, 0, xyz);
     xyzarr2radecdeg(xyz, &ra, &dec);
     CuAssertDblEquals(ct, 45, ra, 0);
     CuAssertDblEquals(ct, 0, dec, 0);
     */

    rd[0] = rd[1] = -1.0;
    d = healpix_distance_to_radec(4, 1, 45+1, 0, rd);
    CuAssertDblEquals(ct, 45, rd[0], 1e-12);
    CuAssertDblEquals(ct, 0,  rd[1], 1e-8);

    d = healpix_distance_to_radec(4, 1, 45+1, 0+1, rd);
    CuAssertDblEquals(ct, 45, rd[0], 1e-12);
    CuAssertDblEquals(ct, 0,  rd[1], 0);
    // really??

    d = healpix_distance_to_radec(4, 1, 20, 25, rd);
    CuAssertDblEquals(ct, d, 2.297298, 1e-6);
    CuAssertDblEquals(ct, 18.200995, rd[0], 1e-6);
    CuAssertDblEquals(ct, 23.392159, rd[1], 1e-6);

}

void test_healpix_neighbours(CuTest *ct) {
    int n0[] = { 1,3,2,71,69,143,90,91 };
    int n5[] = { 26,27,7,6,4,94,95 };
    int n13[] = { 30,31,15,14,12,6,7,27 };
    int n15[] = { 31,47,63,61,14,12,13,30 };
    int n30[] = { 31,15,13,7,27,25,28,29 };
    int n101[] = { 32,34,103,102,100,174,175,122 };
    int n127[] = { 58,37,36,126,124,125,56 };
    int n64[] = { 65,67,66,183,181,138,139 };
    int n133[] = { 80,82,135,134,132,152,154 };
    int n148[] = { 149,151,150,147,145,162,168,170 };
    int n160[] = { 161,163,162,145,144,128,176,178 };
    int n24[] = { 25,27,26,95,93,87,18,19 };
    int n42[] = { 43,23,21,111,109,40,41 };
    int n59[] = { 62,45,39,37,58,56,57,60 };
    int n191[] = { 74,48,117,116,190,188,189,72 };
    int n190[] = { 191,117,116,113,187,185,188,189 };
    int n186[] = { 187,113,112,165,164,184,185 };
    int n184[] = { 185,187,186,165,164,161,178,179 };

    // These were taken (IIRC) from the Healpix paper, so the healpix
    // numbers are all in the NESTED scheme.

    tst_nested(ct, 0,   n0,   sizeof(n0)  /sizeof(int), 4);
    tst_nested(ct, 5,   n5,   sizeof(n5)  /sizeof(int), 4);
    tst_nested(ct, 13,  n13,  sizeof(n13) /sizeof(int), 4);
    tst_nested(ct, 15,  n15,  sizeof(n15) /sizeof(int), 4);
    tst_nested(ct, 30,  n30,  sizeof(n30) /sizeof(int), 4);
    tst_nested(ct, 101, n101, sizeof(n101)/sizeof(int), 4);
    tst_nested(ct, 127, n127, sizeof(n127)/sizeof(int), 4);
    tst_nested(ct, 64,  n64,  sizeof(n64) /sizeof(int), 4);
    tst_nested(ct, 133, n133, sizeof(n133)/sizeof(int), 4);
    tst_nested(ct, 148, n148, sizeof(n148)/sizeof(int), 4);
    tst_nested(ct, 160, n160, sizeof(n160)/sizeof(int), 4);
    tst_nested(ct, 24,  n24,  sizeof(n24) /sizeof(int), 4);
    tst_nested(ct, 42,  n42,  sizeof(n42) /sizeof(int), 4);
    tst_nested(ct, 59,  n59,  sizeof(n59) /sizeof(int), 4);
    tst_nested(ct, 191, n191, sizeof(n191)/sizeof(int), 4);
    tst_nested(ct, 190, n190, sizeof(n190)/sizeof(int), 4);
    tst_nested(ct, 186, n186, sizeof(n186)/sizeof(int), 4);
    tst_nested(ct, 184, n184, sizeof(n184)/sizeof(int), 4);
}

/*
 void pnprime_to_xy(int, int*, int*, int);
 int xy_to_pnprime(int, int, int);

 void tst_healpix_pnprime_to_xy(CuTest *ct) {
 int px,py;
 pnprime_to_xy(6, &px, &py, 3);
 CuAssertIntEquals(ct, px, 2);
 CuAssertIntEquals(ct, py, 0);
 pnprime_to_xy(8, &px, &py, 3);
 CuAssertIntEquals(ct, px, 2);
 CuAssertIntEquals(ct, py, 2);
 pnprime_to_xy(0, &px, &py, 3);
 CuAssertIntEquals(ct, px, 0);
 CuAssertIntEquals(ct, py, 0);
 pnprime_to_xy(2, &px, &py, 3);
 CuAssertIntEquals(ct, px, 0);
 CuAssertIntEquals(ct, py, 2);
 pnprime_to_xy(4, &px, &py, 3);
 CuAssertIntEquals(ct, px, 1);
 CuAssertIntEquals(ct, py, 1);
 }

 void tst_healpix_xy_to_pnprime(CuTest *ct) {
 CuAssertIntEquals(ct, xy_to_pnprime(0,0,3), 0);
 CuAssertIntEquals(ct, xy_to_pnprime(1,0,3), 3);
 CuAssertIntEquals(ct, xy_to_pnprime(2,0,3), 6);
 CuAssertIntEquals(ct, xy_to_pnprime(0,1,3), 1);
 CuAssertIntEquals(ct, xy_to_pnprime(1,1,3), 4);
 CuAssertIntEquals(ct, xy_to_pnprime(2,1,3), 7);
 CuAssertIntEquals(ct, xy_to_pnprime(0,2,3), 2);
 CuAssertIntEquals(ct, xy_to_pnprime(1,2,3), 5);
 CuAssertIntEquals(ct, xy_to_pnprime(2,2,3), 8);
 }
 */
void print_test_healpix_output(int Nside) {

    int i, j;
    double z;
    double phi;
    fprintf(stderr, "graph Nside4 {\n");

    // north polar
    for (i=1; i<=Nside; i++) {
        for (j=1; j<=(4*i); j++) {
            // find the center of the pixel in ring i
            // and longitude j.
            z = 1.0 - square((double)i / (double)Nside)/3.0;
            phi = M_PI / (2.0 * i) * ((double)j - 0.5);
            fprintf(stderr, "  // North polar, i=%i, j=%i.  z=%g, phi=%g\n", i, j, z, phi);
            print_node(z, phi, Nside);
        }
    }
    // south polar
    for (i=1; i<=Nside; i++) {
        for (j=1; j<=(4*i); j++) {
            z = 1.0 - square((double)i / (double)Nside)/3.0;
            z *= -1.0;
            phi = M_PI / (2.0 * i) * ((double)j - 0.5);
            fprintf(stderr, "  // South polar, i=%i, j=%i.  z=%g, phi=%g\n", i, j, z, phi);
            print_node(z, phi, Nside);
        }
    }
    // north equatorial
    for (i=Nside+1; i<=2*Nside; i++) {
        for (j=1; j<=(4*Nside); j++) {
            int s;
            z = 4.0/3.0 - 2.0 * i / (3.0 * Nside);
            s = (i - Nside + 1) % 2;
            s = (s + 2) % 2;
            phi = M_PI / (2.0 * Nside) * ((double)j - (double)s / 2.0);
            fprintf(stderr, "  // North equatorial, i=%i, j=%i.  z=%g, phi=%g, s=%i\n", i, j, z, phi, s);
            print_node(z, phi, Nside);
        }
    }
    // south equatorial
    for (i=Nside+1; i<2*Nside; i++) {
        for (j=1; j<=(4*Nside); j++) {
            int s;
            z = 4.0/3.0 - 2.0 * i / (3.0 * Nside);
            z *= -1.0;
            s = (i - Nside + 1) % 2;
            s = (s + 2) % 2;
            phi = M_PI / (2.0 * Nside) * ((double)j - s / 2.0);
            fprintf(stderr, "  // South equatorial, i=%i, j=%i.  z=%g, phi=%g, s=%i\n", i, j, z, phi, s);
            print_node(z, phi, Nside);
        }
    }

    fprintf(stderr, "  node [ shape=point ]\n");
    fprintf(stderr, "  C0 [ pos=\"0,-10!\" ];\n");
    fprintf(stderr, "  C1 [ pos=\"20,-10!\" ];\n");
    fprintf(stderr, "  C2 [ pos=\"20,10!\" ];\n");
    fprintf(stderr, "  C3 [ pos=\"0,10!\" ];\n");
    fprintf(stderr, "  C0 -- C1 -- C2 -- C3 -- C0\n");
    fprintf(stderr, "}\n");
}

void print_healpix_grid(int Nside) {
    int i;
    int j;
    int N = 500;

    fprintf(stderr, "x%i=[", Nside);
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
            fprintf(stderr, "%i ", radectohealpix(i*2*M_PI/N, M_PI*(j-N/2)/N, Nside));
        }
        fprintf(stderr, ";");
    }
    fprintf(stderr, "];\n\n");
    fflush(stderr);
}

void print_healpix_borders(int Nside) {
    int i;
    int j;
    int N = 1;

    fprintf(stderr, "x%i=[", Nside);
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
            fprintf(stderr, "%i ", radectohealpix(i*2*M_PI/N, M_PI*(j-N/2)/N, Nside));
        }
        fprintf(stderr, ";");
    }
    fprintf(stderr, "];\n\n");
    fflush(stderr);
}

void test_big_nside(CuTest* ct) {
    double ra1, dec1, ra2, dec2;
    int Nside = 2097152;
    int64_t hp;
    double dx, dy;

    // just a random val...
    ra1 = 43.7;
    dec1 = -38.4;

    hp = radecdegtohealpixl(ra1, dec1, Nside);
    healpixl_to_radecdeg(hp, Nside, 0, 0, &ra2, &dec2);

    CuAssertDblEquals(ct, ra1, ra2, arcsec2deg(0.1));
    CuAssertDblEquals(ct, dec1, dec2, arcsec2deg(0.1));

    // another random val...
    ra1 = 0.0003;
    dec1 = 75.3;

    dx = dy = -1.0;
    hp = radecdegtohealpixlf(ra1, dec1, Nside, &dx, &dy);
    CuAssert(ct, "dx", dx >= 0.0);
    CuAssert(ct, "dx", dx <= 1.0);
    CuAssert(ct, "dy", dy >= 0.0);
    CuAssert(ct, "dy", dy <= 1.0);
    healpixl_to_radecdeg(hp, Nside, dx, dy, &ra2, &dec2);

    CuAssertDblEquals(ct, ra1, ra2, arcsec2deg(1e-10));
    CuAssertDblEquals(ct, dec1, dec2, arcsec2deg(1e-10));

    printf("RA,Dec difference: %g, %g arcsec\n", deg2arcsec(ra2-ra1), deg2arcsec(dec2-dec1));
}

void test_distortion_at_pole(CuTest* ct) {
    // not really a test of the code, more of healpix itself...
    double ra1, dec1, ra2, dec2, ra3, dec3, ra4, dec4;
    int Nside = 2097152;
    int64_t hp;
    double d1, d2, d3, d4, d5, d6;
    double testras [] = {  0.0, 45.0,  0.0, 0.0 };
    double testdecs[] = { 90.0, 50.0, 40.0, 0.0 };
    char* testnames[] = { "north pole", "mid-polar", "mid-equatorial",
                          "equator" };
    double ra, dec;
    int i;

    for (i=0; i<sizeof(testras)/sizeof(double); i++) {
        ra = testras[i];
        dec = testdecs[i];

        hp = radecdegtohealpixl(ra, dec, Nside);

        healpixl_to_radecdeg(hp, Nside, 0, 0, &ra1, &dec1);
        healpixl_to_radecdeg(hp, Nside, 0, 1, &ra2, &dec2);
        healpixl_to_radecdeg(hp, Nside, 1, 1, &ra3, &dec3);
        healpixl_to_radecdeg(hp, Nside, 1, 0, &ra4, &dec4);

        // sides
        d1 = arcsec_between_radecdeg(ra1, dec1, ra2, dec2);
        d2 = arcsec_between_radecdeg(ra2, dec2, ra3, dec3);
        d3 = arcsec_between_radecdeg(ra3, dec3, ra4, dec4);
        d4 = arcsec_between_radecdeg(ra4, dec4, ra1, dec1);
        // diagonals
        d5 = arcsec_between_radecdeg(ra1, dec1, ra3, dec3);
        d6 = arcsec_between_radecdeg(ra2, dec2, ra4, dec4);
		
        printf("%-15s (%4.1f, %4.1f): %-5.3f, %-5.3f, %-5.3f, %-5.3f / %-5.3f, %-5.3f\n", testnames[i], ra, dec, d1, d2, d3, d4, d5, d6);
    }
}


#if defined(TEST_HEALPIX_MAIN)
int main(int argc, char** args) {

    /* Run all tests */
    CuString *output = CuStringNew();
    CuSuite* suite = CuSuiteNew();

    /* Add new tests here */
    SUITE_ADD_TEST(suite, test_healpix_neighbours);
    SUITE_ADD_TEST(suite, test_healpix_pnprime_to_xy);
    SUITE_ADD_TEST(suite, test_healpix_xy_to_pnprime);
    SUITE_ADD_TEST(suite, test_healpix_distance_to_radec);

    /* Run the suite, collect results and display */
    CuSuiteRun(suite);
    CuSuiteSummary(suite, output);
    CuSuiteDetails(suite, output);
    printf("%s\n", output->buffer);

    /*
     print_healpix_grid(1);
     print_healpix_grid(2);
     print_healpix_grid(3);
     print_healpix_grid(4);
     print_healpix_grid(5);
     */

    //print_test_healpix_output();
	
    /*
     int rastep, decstep;
     int Nra = 100;
     int Ndec = 100;
     double ra, dec;
     int healpix;
     printf("radechealpix=zeros(%i,3);\n", Nra*Ndec);
     for (rastep=0; rastep<Nra; rastep++) {
     ra = ((double)rastep / (double)(Nra-1)) * 2.0 * M_PI;
     for (decstep=0; decstep<Ndec; decstep++) {
     dec = (((double)decstep / (double)(Ndec-1)) * M_PI) - M_PI/2.0;
     healpix = radectohealpix(ra, dec);
     printf("radechealpix(%i,:)=[%g,%g,%i];\n", 
     (rastep*Ndec) + decstep + 1, ra, dec, healpix);
     }
     }
     printf("ra=radechealpix(:,1);\n");
     printf("dec=radechealpix(:,2);\n");
     printf("healpix=radechealpix(:,3);\n");
     return 0;
     */
    return 0;
}
#endif
