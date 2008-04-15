/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

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
#include <stdio.h>

#include "cutest.h"
#include "starutil.h"
#include "healpix.h"

static double square(x) {
    return x*x;
}

static void tst_neighbours(CuTest* ct, uint pix, uint* true_neigh, uint true_nn,
                           int Nside) {
	uint neigh[8];
	uint nn;
	uint i;
	for (i=0; i<8; i++)
		neigh[i] = -1;
	nn = healpix_get_neighbours(pix, neigh, Nside);
	printf("true(%i) : [ ", pix);
	for (i=0; i<true_nn; i++)
		printf("%u, ", true_neigh[i]);
	printf("]\n");
	printf("got (%i) : [ ", pix);
	for (i=0; i<nn; i++)
		printf("%u, ", neigh[i]);
	printf("]\n");

	CuAssertIntEquals(ct, nn, true_nn);

	for (i=0; i<true_nn; i++)
		CuAssertIntEquals(ct, true_neigh[i], neigh[i]);
}

static void tst_nested(CuTest* ct, uint pix, uint* true_neigh, uint true_nn,
                       int Nside) {
    int i;
    uint truexy[8];
    uint xypix;

	printf("nested true(%i) : [ ", pix);
	for (i=0; i<true_nn; i++)
		printf("%u ", true_neigh[i]);
	printf("]\n");

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
	uint nn;
	uint neigh[8];
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

void test_healpix_neighbours(CuTest *ct) {
	uint n0[] = { 1,3,2,71,69,143,90,91 };
	uint n5[] = { 26,27,7,6,4,94,95 };
	uint n13[] = { 30,31,15,14,12,6,7,27 };
	uint n15[] = { 31,47,63,61,14,12,13,30 };
	uint n30[] = { 31,15,13,7,27,25,28,29 };
	uint n101[] = { 32,34,103,102,100,174,175,122 };
	uint n127[] = { 58,37,36,126,124,125,56 };
	uint n64[] = { 65,67,66,183,181,138,139 };
	uint n133[] = { 80,82,135,134,132,152,154 };
	uint n148[] = { 149,151,150,147,145,162,168,170 };
	uint n160[] = { 161,163,162,145,144,128,176,178 };
	uint n24[] = { 25,27,26,95,93,87,18,19 };
	uint n42[] = { 43,23,21,111,109,40,41 };
	uint n59[] = { 62,45,39,37,58,56,57,60 };
	uint n191[] = { 74,48,117,116,190,188,189,72 };
	uint n190[] = { 191,117,116,113,187,185,188,189 };
	uint n186[] = { 187,113,112,165,164,184,185 };
	uint n184[] = { 185,187,186,165,164,161,178,179 };

    // These were taken (IIRC) from the Healpix paper, so the healpix
    // numbers are all in the NESTED scheme.

	tst_nested(ct, 0,   n0,   sizeof(n0)  /sizeof(uint), 4);
	tst_nested(ct, 5,   n5,   sizeof(n5)  /sizeof(uint), 4);
	tst_nested(ct, 13,  n13,  sizeof(n13) /sizeof(uint), 4);
	tst_nested(ct, 15,  n15,  sizeof(n15) /sizeof(uint), 4);
	tst_nested(ct, 30,  n30,  sizeof(n30) /sizeof(uint), 4);
	tst_nested(ct, 101, n101, sizeof(n101)/sizeof(uint), 4);
	tst_nested(ct, 127, n127, sizeof(n127)/sizeof(uint), 4);
	tst_nested(ct, 64,  n64,  sizeof(n64) /sizeof(uint), 4);
	tst_nested(ct, 133, n133, sizeof(n133)/sizeof(uint), 4);
	tst_nested(ct, 148, n148, sizeof(n148)/sizeof(uint), 4);
	tst_nested(ct, 160, n160, sizeof(n160)/sizeof(uint), 4);
	tst_nested(ct, 24,  n24,  sizeof(n24) /sizeof(uint), 4);
	tst_nested(ct, 42,  n42,  sizeof(n42) /sizeof(uint), 4);
	tst_nested(ct, 59,  n59,  sizeof(n59) /sizeof(uint), 4);
	tst_nested(ct, 191, n191, sizeof(n191)/sizeof(uint), 4);
	tst_nested(ct, 190, n190, sizeof(n190)/sizeof(uint), 4);
	tst_nested(ct, 186, n186, sizeof(n186)/sizeof(uint), 4);
	tst_nested(ct, 184, n184, sizeof(n184)/sizeof(uint), 4);
}

/*
void pnprime_to_xy(uint, uint*, uint*, uint);
uint xy_to_pnprime(uint, uint, uint);

void tst_healpix_pnprime_to_xy(CuTest *ct) {
	uint px,py;
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
			fprintf(stderr, "%i ", radectohealpix(i*2*PIl/N, PIl*(j-N/2)/N, Nside));
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
			fprintf(stderr, "%i ", radectohealpix(i*2*PIl/N, PIl*(j-N/2)/N, Nside));
		}
		fprintf(stderr, ";");
	}
	fprintf(stderr, "];\n\n");
	fflush(stderr);
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
