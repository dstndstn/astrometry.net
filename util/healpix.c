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
#include <assert.h>
#include <stdio.h>
#include <sys/param.h>

#include "healpix.h"
#include "mathutil.h"
#include "starutil.h"
#include "keywords.h"

// I've had troubles with rounding functions being declared properly
// in other contexts...  Declare it here so the compiler complains if
// something is wrong.
double round(double x);

Const static Inline double mysquare(double d) {
	return d*d;
}

Const int healpix_xy_to_nested(int hp, int Nside) {
	int bighp,x,y;
	int index;
	int i;

	healpix_decompose_xy(hp, &bighp, &x, &y, Nside);
	if (!is_power_of_two(Nside)) {
		fprintf(stderr, "healpix_xy_to_nested: Nside must be a power of two.\n");
		return -1;
	}

	// We construct the index called p_n' in the healpix paper, whose bits
	// are taken from the bits of x and y:
	//    x = ... b4 b2 b0
	//    y = ... b5 b3 b1
	// We go through the bits of x,y, building up "index":
	index = 0;
	for (i=0; i<(8*sizeof(int)/2); i++) {
		index |= (((y & 1) << 1) | (x & 1)) << (i*2);
		y >>= 1;
		x >>= 1;
		if (!x && !y) break;
	}

	return index + bighp * Nside * Nside;
}

Const int healpix_nested_to_xy(int hp, int Nside) {
	int bighp, x, y;
	int index;
	int i;
	if (!is_power_of_two(Nside)) {
		fprintf(stderr, "healpix_xy_to_nested: Nside must be a power of two.\n");
		return -1;
	}
	bighp = hp / (Nside*Nside);
	index = hp % (Nside*Nside);
	x = y = 0;
	for (i=0; i<(8*sizeof(int)/2); i++) {
		x |= (index & 0x1) << i;
		index >>= 1;
		y |= (index & 0x1) << i;
		index >>= 1;
		if (!index) break;
	}
	return healpix_compose_xy(bighp, x, y, Nside);
}

Const int healpix_compose_ring(int ring, int longind, int Nside) {
	if (ring <= Nside)
		// north polar
		return ring * (ring-1) * 2 + longind;
	if (ring < 3*Nside)
		// equatorial
		return Nside*(Nside-1)*2 + Nside*4*(ring-Nside) + longind;
	{
		int ri;
		ri = 4*Nside - ring;
		return 12*Nside*Nside-1 - ( ri*(ri-1)*2 + (ri*4 - 1 - longind) );
	}
}

void healpix_decompose_ring(int hp, int Nside, int* p_ring, int* p_longind) {
	// FIXME: this could be written in closed form...
	int longind;
	int ring;
	int offset = 0;
	for (ring=1; ring<=Nside; ring++) {
		if (offset + ring*4 > hp) {
			longind = hp - offset;
			goto gotit;
		}
		offset += ring*4;
	}
	for (; ring<(3*Nside); ring++) {
		if (offset + Nside*4 > hp) {
			longind = hp - offset;
			goto gotit;
		}
		offset += Nside*4;
	}
	for (; ring<(4*Nside); ring++) {
		if (offset + (Nside*4 - ring)*4 > hp) {
			longind = hp - offset;
			goto gotit;
		}
		offset += (Nside*4 - ring)*4;
	}
	fprintf(stderr, "healpix_decompose_ring: shouldn't get here!\n");
	if (p_ring) *p_ring = -1;
	if (p_longind) *p_longind = -1;
	return;
 gotit:
	if (p_ring)
		*p_ring = ring;
	if (p_longind)
		*p_longind = longind;
}

Const int healpix_ring_to_xy(int ring, int Nside) {
	int bighp, x, y;
	int ringind, longind;
	healpix_decompose_ring(ring, Nside, &ringind, &longind);
	if (ringind <= Nside) {
		int ind;
		int v;
		int F1;
		int frow;
		bighp = longind / ringind;
		ind = longind - bighp * ringind;
		y = (Nside - 1 - ind);
		frow = bighp / 4;
		F1 = frow + 2;
		v = F1*Nside - ringind - 1;
		x = v - y;
		return healpix_compose_xy(bighp, x, y, Nside);
	} else if (ringind < 3*Nside) {
		int panel;
		int ind;
		int bottomleft;
		int topleft;
		int frow, F1, F2, s, v, h;
		int bighp = -1;
		int x, y;
		int hp;
		int R = 0;

		panel = longind / Nside;
		ind = longind % Nside;
		bottomleft = ind < (ringind - Nside + 1) / 2;
		topleft = ind < (3*Nside - ringind + 1)/2;

		if (!bottomleft && topleft) {
			// top row.
			bighp = panel;
		} else if (bottomleft && !topleft) {
			// bottom row.
			bighp = 8 + panel;
		} else if (bottomleft && topleft) {
			// left side.
			bighp = 4 + panel;
		} else if (!bottomleft && !topleft) {
			// right side.
			bighp = 4 + (panel + 1) % 4;
			if (bighp == 4) {
				longind -= (4*Nside - 1);
				// Gah!  Wacky hack - it seems that since
				// "longind" is negative in this case, the
				// rounding behaves differently, so we end up
				// computing the wrong "h" and have to correct
				// for it.
				R = 1;
			}
		}

		frow = bighp / 4;
		F1 = frow + 2;
		F2 = 2*(bighp % 4) - (frow % 2) + 1;
		s = (ringind - Nside) % 2;
		v = F1*Nside - ringind - 1;
		h = 2*longind - s - F2*Nside;
		if (R)
			h--;
		x = (v + h) / 2;
		y = (v - h) / 2;
		//fprintf(stderr, "bighp=%i, frow=%i, F1=%i, F2=%i, s=%i, v=%i, h=%i, x=%i, y=%i.\n", bighp, frow, F1, F2, s, v, h, x, y);

		if ((v != (x+y)) || (h != (x-y))) {
			h++;
			x = (v + h) / 2;
			y = (v - h) / 2;
			//fprintf(stderr, "tweak h=%i, x=%i, y=%i\n", h, x, y);

			if ((v != (x+y)) || (h != (x-y))) {
				//fprintf(stderr, "still not right.\n");
			}
		}
		hp = healpix_compose_xy(bighp, x, y, Nside);
		//fprintf(stderr, "hp %i\n", hp);
		return hp;
	} else {
		int ind;
		int v;
		int F1;
		int frow;
		int ri;
		ri = 4*Nside - ringind;
		bighp = 8 + longind / ri;
		ind = longind - (bighp%4) * ri;
		y = (ri-1) - ind;
		frow = bighp / 4;
		F1 = frow + 2;
		v = F1*Nside - ringind - 1;
		x = v - y;
		return healpix_compose_xy(bighp, x, y, Nside);
	}
}

Const int healpix_xy_to_ring(int hp, int Nside) {
	int bighp,x,y;
	int frow;
	int F1;
	int v;
	int ring;
	int index;

	healpix_decompose_xy(hp, &bighp, &x, &y, Nside);
	frow = bighp / 4;
	F1 = frow + 2;
	v = x + y;
	// "ring" starts from 1 at the north pole and goes to 4Nside-1 at
	// the south pole; the pixels in each ring have the same latitude.
	ring = F1*Nside - v - 1;
	/*
	 ring:
	 [1, Nside] : n pole
	 (Nside, 2Nside] : n equatorial
	 (2Nside+1, 3Nside) : s equat
	 [3Nside, 4Nside-1] : s pole
	 */
	// this probably can't happen...
	if ((ring < 1) || (ring >= 4*Nside)) {
		fprintf(stderr, "Invalid ring index: %i\n", ring);
		return -1;
	}
	if (ring <= Nside) {
		// north polar.
		// left-to-right coordinate within this healpix
		index = (Nside - 1 - y);
		// offset from the other big healpixes
		index += ((bighp % 4) * ring);
		// offset from the other rings
		index += ring*(ring-1)*2;
	} else if (ring >= 3*Nside) {
		// south polar.
		// Here I first flip everything so that we label the pixels
		// at zero starting in the southeast corner, increasing to the
		// west and north, then subtract that from the total number of
		// healpixels.
		int ri = 4*Nside - ring;
		// index within this healpix
		index = (ri-1) - x;
		// big healpixes
		index += ((3-(bighp % 4)) * ri);
		// other rings
		index += ri*(ri-1)*2;
		// flip!
		index = 12*Nside*Nside - 1 - index;
	} else {
		// equatorial.
		int s, F2, h;
		s = (ring - Nside) % 2;
		F2 = 2*((int)bighp % 4) - (frow % 2) + 1;
		h = x - y;
		index = (F2 * (int)Nside + h + s) / 2;
		// offset from the north polar region:
		index += Nside*(Nside-1)*2;
		// offset within the equatorial region:
		index += Nside * 4 * (ring - Nside);
		// handle healpix #4 wrap-around
		if ((bighp == 4) && (y > x))
			index += (4 * Nside - 1);
		//fprintf(stderr, "frow=%i, F1=%i, v=%i, ringind=%i, s=%i, F2=%i, h=%i, longind=%i.\n", frow, F1, v, ring, s, F2, h, (F2*(int)Nside+h+s)/2);
	}
	return index;
}

Const double healpix_side_length_arcmin(int Nside) {
	return sqrt((4.0 * M_PI * mysquare(180.0 * 60.0 / M_PI)) /
				(double)(12 * Nside * Nside));
}

static Inline void swap(int* i1, int* i2) {
	int tmp;
	tmp = *i1;
	*i1 = *i2;
	*i2 = tmp;
}

static Inline void swap_double(double* i1, double* i2) {
	double tmp;
	tmp = *i1;
	*i1 = *i2;
	*i2 = tmp;
}

static Inline bool ispolar(int healpix)
{
	// the north polar healpixes are 0,1,2,3
	// the south polar healpixes are 8,9,10,11
	return (healpix <= 3) || (healpix >= 8);
}

static Inline bool isequatorial(int healpix)
{
	// the north polar healpixes are 0,1,2,3
	// the south polar healpixes are 8,9,10,11
	return (healpix >= 4) && (healpix <= 7);
}

static Inline bool isnorthpolar(int healpix)
{
	return (healpix <= 3);
}

static Inline bool issouthpolar(int healpix)
{
	return (healpix >= 8);
}

static int compose_xy(int x, int y, int Nside) {
	return (x * Nside) + y;
}

int healpix_compose_xy(int bighp, int x, int y, int Nside) {
	return (bighp * Nside * Nside) + compose_xy(x, y, Nside);
}

void healpix_decompose_xy(int finehp, int* pbighp, int* px, int* py, int Nside) {
	int hp;
	if (pbighp) {
		int bighp   = finehp / (Nside * Nside);
		*pbighp = bighp;
	}
	hp = finehp % (Nside * Nside);
	if (px)
		*px = hp / Nside;
	if (py)
		*py = hp % Nside;
}

/**
 Given a large-scale healpix number, computes its neighbour in the
 direction (dx,dy).  Returns -1 if there is no such neighbour.
 */
static int healpix_get_neighbour(int hp, int dx, int dy)
{
	if (isnorthpolar(hp)) {
		if ((dx ==  1) && (dy ==  0)) return (hp + 1) % 4;
		if ((dx ==  0) && (dy ==  1)) return (hp + 3) % 4;
		if ((dx ==  1) && (dy ==  1)) return (hp + 2) % 4;
		if ((dx == -1) && (dy ==  0)) return (hp + 4);
		if ((dx ==  0) && (dy == -1)) return 4 + ((hp + 1) % 4);
		if ((dx == -1) && (dy == -1)) return hp + 8;
		return -1;
	} else if (issouthpolar(hp)) {
		if ((dx ==  1) && (dy ==  0)) return 4 + ((hp + 1) % 4);
		if ((dx ==  0) && (dy ==  1)) return hp - 4;
		if ((dx == -1) && (dy ==  0)) return 8 + ((hp + 3) % 4);
		if ((dx ==  0) && (dy == -1)) return 8 + ((hp + 1) % 4);
		if ((dx == -1) && (dy == -1)) return 8 + ((hp + 2) % 4);
		if ((dx ==  1) && (dy ==  1)) return hp - 8;
		return -1;
	} else {
		if ((dx ==  1) && (dy ==  0)) return hp - 4;
		if ((dx ==  0) && (dy ==  1)) return (hp + 3) % 4;
		if ((dx == -1) && (dy ==  0)) return 8 + ((hp + 3) % 4);
		if ((dx ==  0) && (dy == -1)) return hp + 4;
		if ((dx ==  1) && (dy == -1)) return 4 + ((hp + 1) % 4);
		if ((dx == -1) && (dy ==  1)) return 4 + ((hp - 1) % 4);
		return -1;
	}
	return -1;
}

int healpix_get_neighbours(int pix, int* neighbour, int Nside)
{
	int base;
	int x, y;
	int nn = 0;
	int nbase;
	int Ns2 = Nside * Nside;
	int nx, ny;

	healpix_decompose_xy(pix, &base, &x, &y, Nside);

	// ( + , 0 )
	nx = (x + 1) % Nside;
	ny = y;
	if (x == (Nside - 1)) {
		nbase = healpix_get_neighbour(base, 1, 0);
		if (isnorthpolar(base)) {
			nx = x;
			swap(&nx, &ny);
		}
	} else
		nbase = base;

	//printf("(+ 0): nbase=%i, nx=%i, ny=%i, pix=%i\n", nbase, nx, ny, nbase*Ns2+xy_to_pnprime(nx,ny,Nside));
	neighbour[nn] = nbase * Ns2 + compose_xy(nx, ny, Nside);
	nn++;


	// ( + , + )
	nx = (x + 1) % Nside;
	ny = (y + 1) % Nside;
	if ((x == Nside - 1) && (y == Nside - 1)) {
		if (ispolar(base))
			nbase = healpix_get_neighbour(base, 1, 1);
		else
			nbase = -1;
	} else if (x == (Nside - 1))
		nbase = healpix_get_neighbour(base, 1, 0);
	else if (y == (Nside - 1))
		nbase = healpix_get_neighbour(base, 0, 1);
	else
		nbase = base;

	if (isnorthpolar(base)) {
		if (x == (Nside - 1))
			nx = Nside - 1;
		if (y == (Nside - 1))
			ny = Nside - 1;
		if ((x == (Nside - 1)) || (y == (Nside - 1)))
			swap(&nx, &ny);
	}

	//printf("(+ +): nbase=%i, nx=%i, ny=%i, pix=%i\n", nbase, nx, ny, nbase*Ns2+xy_to_pnprime(nx,ny,Nside));

	if (nbase != -1) {
		neighbour[nn] = nbase * Ns2 + compose_xy(nx, ny, Nside);
		nn++;
	}



	// ( 0 , + )
	nx = x;
	ny = (y + 1) % Nside;
	if (y == (Nside - 1)) {
		nbase = healpix_get_neighbour(base, 0, 1);
		if (isnorthpolar(base)) {
			ny = y;
			swap(&nx, &ny);
		}
	} else
		nbase = base;
    
	//printf("(0 +): nbase=%i, nx=%i, ny=%i, pix=%i\n", nbase, nx, ny, nbase*Ns2+xy_to_pnprime(nx,ny,Nside));

	neighbour[nn] = nbase * Ns2 + compose_xy(nx, ny, Nside);
	nn++;



	// ( - , + )
	nx = (x + Nside - 1) % Nside;
	ny = (y + 1) % Nside;
	if ((x == 0) && (y == (Nside - 1))) {
		if (isequatorial(base))
			nbase = healpix_get_neighbour(base, -1, 1);
		else
			nbase = -1;
	} else if (x == 0) {
		nbase = healpix_get_neighbour(base, -1, 0);
		if (issouthpolar(base)) {
			nx = 0;
			swap(&nx, &ny);
		}
	} else if (y == (Nside - 1)) {
		nbase = healpix_get_neighbour(base, 0, 1);
		if (isnorthpolar(base)) {
			ny = y;
			swap(&nx, &ny);
		}
	} else
		nbase = base;

	//printf("(- +): nbase=%i, nx=%i, ny=%i, pix=%i\n", nbase, nx, ny, nbase*Ns2+xy_to_pnprime(nx,ny,Nside));

	if (nbase != -1) {
		neighbour[nn] = nbase * Ns2 + compose_xy(nx, ny, Nside);
		nn++;
	}


	// ( - , 0 )
	nx = (x + Nside - 1) % Nside;
	ny = y;
	if (x == 0) {
		nbase = healpix_get_neighbour(base, -1, 0);
		if (issouthpolar(base)) {
			nx = 0;
			swap(&nx, &ny);
		}
	} else
		nbase = base;

	//printf("(- 0): nbase=%i, nx=%i, ny=%i, pix=%i\n", nbase, nx, ny, nbase*Ns2+xy_to_pnprime(nx,ny,Nside));

	neighbour[nn] = nbase * Ns2 + compose_xy(nx, ny, Nside);
	nn++;



	// ( - , - )
	nx = (x + Nside - 1) % Nside;
	ny = (y + Nside - 1) % Nside;
	if ((x == 0) && (y == 0)) {
		if (ispolar(base))
			nbase = healpix_get_neighbour(base, -1, -1);
		else
			nbase = -1;
	} else if (x == 0)
		nbase = healpix_get_neighbour(base, -1, 0);
	else if (y == 0)
		nbase = healpix_get_neighbour(base, 0, -1);
	else
		nbase = base;

	if (issouthpolar(base)) {
		if (x == 0)
			nx = 0;
		if (y == 0)
			ny = 0;
		if ((x == 0) || (y == 0))
			swap(&nx, &ny);
	}

	//printf("(- -): nbase=%i, nx=%i, ny=%i, pix=%i\n", nbase, nx, ny, nbase*Ns2+xy_to_pnprime(nx,ny,Nside));

	if (nbase != -1) {
		neighbour[nn] = nbase * Ns2 + compose_xy(nx, ny, Nside);
		nn++;
	}


	// ( 0 , - )
	ny = (y + Nside - 1) % Nside;
	nx = x;
	if (y == 0) {
		nbase = healpix_get_neighbour(base, 0, -1);
		if (issouthpolar(base)) {
			ny = y;
			swap(&nx, &ny);
		}
	} else
		nbase = base;

	//printf("(0 -): nbase=%i, nx=%i, ny=%i, pix=%i\n", nbase, nx, ny, nbase*Ns2+xy_to_pnprime(nx,ny,Nside));

	neighbour[nn] = nbase * Ns2 + compose_xy(nx, ny, Nside);
	nn++;


	// ( + , - )
	nx = (x + 1) % Nside;
	ny = (y + Nside - 1) % Nside;
	if ((x == (Nside - 1)) && (y == 0)) {
		if (isequatorial(base)) {
			nbase = healpix_get_neighbour(base, 1, -1);
		} else
			nbase = -1;

	} else if (x == (Nside - 1)) {
		nbase = healpix_get_neighbour(base, 1, 0);
		if (isnorthpolar(base)) {
			nx = x;
			swap(&nx, &ny);
		}
	} else if (y == 0) {
		nbase = healpix_get_neighbour(base, 0, -1);
		if (issouthpolar(base)) {
			ny = y;
			swap(&nx, &ny);
		}
	} else
		nbase = base;

	//printf("(+ -): nbase=%i, nx=%i, ny=%i, pix=%i\n", nbase, nx, ny, nbase*Ns2+xy_to_pnprime(nx,ny,Nside));

	if (nbase != -1) {
		neighbour[nn] = nbase * Ns2 + compose_xy(nx, ny, Nside);
		nn++;
	}

	return nn;
}

int xyztohealpix(double x, double y, double z, int Nside) {
    return xyztohealpixf(x, y, z, Nside, NULL, NULL);
}

int xyztohealpixf(double x, double y, double z, int Nside,
				  double* p_dx, double* p_dy) {
	double phi;
	double twothirds = 2.0 / 3.0;
	double pi = M_PI;
	double twopi = 2.0 * M_PI;
	double halfpi = 0.5 * M_PI;
    double dx, dy;
    int basehp;
    int hp;
    int pnprime;
	double sector;
	int offset;
	double phi_t;

	double EPS = 1e-8;

	/* Convert our point into cylindrical coordinates for middle ring */
	phi = atan2(y, x);
	if (phi < 0.0)
		phi += twopi;
	phi_t = fmod(phi, halfpi);
	assert (phi_t >= 0.0);

	// North or south polar cap.
	if ((z >= twothirds) || (z <= -twothirds)) {
		double zfactor;
		bool north;
		int x, y;
		int column;
		double root;
		double xx, yy, kx, ky;

		// Which pole?
		if (z >= twothirds) {
			north = TRUE;
			zfactor = 1.0;
		} else {
			north = FALSE;
			zfactor = -1.0;
		}

        // solve eqn 20: k = Ns - xx (in the northern hemi)
		root = (1.0 - z*zfactor) * 3.0 * mysquare(Nside * (2.0 * phi_t - pi) / pi);
		kx = (root <= 0.0) ? 0.0 : sqrt(root);

        // solve eqn 19 for k = Ns - yy
		root = (1.0 - z*zfactor) * 3.0 * mysquare(Nside * 2.0 * phi_t / pi);
		ky = (root <= 0.0) ? 0.0 : sqrt(root);

		if (north) {
			xx = Nside - kx;
			yy = Nside - ky;
		} else {
			xx = ky;
			yy = kx;
		}

		// xx, yy should be in [0, Nside].
		x = MIN(Nside-1, floor(xx));
		assert(x >= 0);
		assert(x < Nside);

		y = MIN(Nside-1, floor(yy));
		assert(y >= 0);
		assert(y < Nside);

		dx = xx - x;
		dy = yy - y;

		pnprime = compose_xy(x, y, Nside);
		assert(pnprime < Nside*Nside);

		sector = (phi - phi_t) / (halfpi);
		offset = (int)round(sector);
		assert(fabs(sector - offset) < EPS);
		offset = ((offset % 4) + 4) % 4;
		assert(offset >= 0);
		assert(offset <= 3);
		column = offset;

		if (north)
			basehp = column;
		else
			basehp = 8 + column;

	} else {
		// could be polar or equatorial.
		double sector;
		int offset;
		double u1, u2;
		double zunits, phiunits;
		int x, y;
        double xx, yy;

		// project into the unit square z=[-2/3, 2/3], phi=[0, pi/2]
		zunits = (z + twothirds) / (4.0 / 3.0);
		phiunits = phi_t / halfpi;
		// convert into diagonal units
		// (add 1 to u2 so that they both cover the range [0,2].
		u1 = zunits + phiunits;
		u2 = zunits - phiunits + 1.0;
		assert(u1 >= 0.);
		assert(u1 <= 2.);
		assert(u2 >= 0.);
		assert(u2 <= 2.);
		// x is the northeast direction, y is the northwest.
        xx = u1 * Nside;
        yy = u2 * Nside;

		// now compute which big healpix it's in.
		// (note that we subtract off the modded portion used to
		// compute the position within the healpix, so this should be
		// very close to one of the boundaries.)
		sector = (phi - phi_t) / (halfpi);
		offset = (int)round(sector);
		assert(fabs(sector - offset) < EPS);
		offset = ((offset % 4) + 4) % 4;
		assert(offset >= 0);
		assert(offset <= 3);

		// we're looking at a square in z,phi space with an X dividing it.
		// we want to know which section we're in.
		// xx ranges from 0 in the bottom-left to 2Nside in the top-right.
		// yy ranges from 0 in the bottom-right to 2Nside in the top-left.
		// (of the phi,z unit box)
		if (xx >= Nside) {
			xx -= Nside;
			if (yy >= Nside) {
				// north polar.
				yy -= Nside;
				basehp = offset;
			} else {
				// right equatorial.
				basehp = ((offset + 1) % 4) + 4;
			}
		} else {
			if (yy >= Nside) {
				// left equatorial.
				yy -= Nside;
				basehp = offset + 4;
			} else {
				// south polar.
				basehp = 8 + offset;
			}
		}

		assert(xx >= -EPS);
		assert(xx < (Nside+EPS));
		x = MAX(0, MIN(Nside-1, floor(xx)));
		assert(x >= 0);
		assert(x < Nside);
		dx = xx - x;

		assert(yy >= -EPS);
		assert(yy < (Nside+EPS));
		y = MAX(0, MIN(Nside-1, floor(yy)));
		assert(y >= 0);
		assert(y < Nside);
		dy = yy - y;

		pnprime = compose_xy(x, y, Nside);
		assert(pnprime < Nside*Nside);
	}
    hp = basehp * Nside * Nside + pnprime;

    if (p_dx) *p_dx = dx;
    if (p_dy) *p_dy = dy;

    return hp;
}

int radectohealpix(double ra, double dec, int Nside) {
    return xyztohealpix(radec2x(ra,dec), radec2y(ra,dec), radec2z(ra,dec), Nside);
}

int radectohealpixf(double ra, double dec, int Nside, double* dx, double* dy) {
    return xyztohealpixf(radec2x(ra,dec), radec2y(ra,dec), radec2z(ra,dec),
                         Nside, dx, dy);
}
 
Const int radecdegtohealpix(double ra, double dec, int Nside) {
	return radectohealpix(deg2rad(ra), deg2rad(dec), Nside);
}

int radecdegtohealpixf(double ra, double dec, int Nside, double* dx, double* dy) {
	return radectohealpixf(deg2rad(ra), deg2rad(dec), Nside, dx, dy);
}

int xyzarrtohealpix(double* xyz, int Nside) {
	return xyztohealpix(xyz[0], xyz[1], xyz[2], Nside);
}

int xyzarrtohealpixf(double* xyz,int Nside, double* p_dx, double* p_dy) {
    return xyztohealpixf(xyz[0], xyz[1], xyz[2], Nside, p_dx, p_dy);
}

void healpix_to_xyz(int hp, int Nside,
					double dx, double dy, 
					double* rx, double *ry, double *rz) {
	int chp;
	bool equatorial = TRUE;
	double zfactor = 1.0;
	int xp, yp;
	double x, y, z;
	double pi = M_PI, phi;
	double rad;

	healpix_decompose_xy(hp, &chp, &xp, &yp, Nside);

	// this is x,y position in the healpix reference frame
	x = xp + dx;
	y = yp + dy;

	if (isnorthpolar(chp)) {
		if ((x + y) > Nside) {
			equatorial = FALSE;
			zfactor = 1.0;
		}
	}
	if (issouthpolar(chp)) {
		if ((x + y) < Nside) {
			equatorial = FALSE;
			zfactor = -1.0;
		}
	}

	if (equatorial) {
		double zoff=0;
		double phioff=0;
		x /= (double)Nside;
		y /= (double)Nside;

		if (chp <= 3) {
			// north
			phioff = 1.0;
		} else if (chp <= 7) {
			// equator
			zoff = -1.0;
			chp -= 4;
		} else if (chp <= 11) {
			// south
			phioff = 1.0;
			zoff = -2.0;
			chp -= 8;
		} else {
			// should never get here
			assert(0);
		}

		z = 2.0/3.0*(x + y + zoff);
		phi = pi/4*(x - y + phioff + 2*chp);

	} else {
		/*
		 Rearrange eqns (19) and (20) to find phi_t in terms of x,y.

		 y = Ns - k in eq(19)
		 x - Ns - k in eq(20)

		 (Ns - y)^2 / (Ns - x)^2 = (2 phi_t)^2 / (2 phi_t - pi)^2

		 Recall than y<=Ns, x<=Ns and 0<=phi_t<pi/2, so we can choose the
		 root we want by taking square roots:

		 (Ns - y) (pi - 2 phi_t) = 2 phi_t (Ns - x)
		 (Ns - y) pi = 2 phi_t (Ns - x + Ns - y)
		 phi_t = pi (Ns-y) / (2 (Ns - x) + (Ns - y))
		 */
		double phi_t;

		if (zfactor == -1.0) {
			swap_double(&x, &y);
			x = (Nside - x);
			y = (Nside - y);
		}

		if (y == Nside && x == Nside)
			phi_t = 0.0;
		else
			phi_t = pi * (Nside-y) / (2.0 * ((Nside-x) + (Nside-y)));

		if (phi_t < pi/4.) {
			z = 1.0 - mysquare(pi * (Nside - x) / ((2.0 * phi_t - pi) * Nside)) / 3.0;
		} else {
			z = 1.0 - mysquare(pi * (Nside - y) / (2.0 * phi_t * Nside)) / 3.0;
		}
		assert(0.0 <= fabs(z) && fabs(z) <= 1.0);
		z *= zfactor;
		assert(0.0 <= fabs(z) && fabs(z) <= 1.0);

		// The big healpix determines the phi offset
		if (issouthpolar(chp))
			phi = pi/2.0* (chp-8) + phi_t;
		else
			phi = pi/2.0 * chp + phi_t;
	}

	if (phi < 0.0)
		phi += 2*pi;

	rad = sqrt(1.0 - z*z);
	*rx = rad * cos(phi);
	*ry = rad * sin(phi);
	*rz = z;
}

void healpix_to_xyzarr(int hp, int Nside,
					   double dx, double dy,
					   double* xyz) {
	healpix_to_xyz(hp, Nside, dx, dy, xyz, xyz+1, xyz+2);
}

void healpix_to_radec(int hp, int Nside,
					  double dx, double dy,
					  double* ra, double* dec) {
	double xyz[3];
	healpix_to_xyzarr(hp, Nside, dx, dy, xyz);
	xyzarr2radec(xyz, ra, dec);
}

void healpix_to_radecdeg(int hp, int Nside,
                         double dx, double dy,
                         double* ra, double* dec) {
	double xyz[3];
	healpix_to_xyzarr(hp, Nside, dx, dy, xyz);
	xyzarr2radecdeg(xyz, ra, dec);
}

void healpix_to_radecarr(int hp, int Nside,
						 double dx, double dy,
						 double* radec) {
	double xyz[3];
	healpix_to_xyzarr(hp, Nside, dx, dy, xyz);
	xyzarr2radec(xyz, radec, radec+1);
}

void healpix_to_radecdegarr(int hp, int Nside,
                            double dx, double dy,
                            double* radec) {
	double xyz[3];
	healpix_to_xyzarr(hp, Nside, dx, dy, xyz);
	xyzarr2radecdeg(xyz, radec, radec+1);
}

/*
 int healpix_get_neighbours_within_range(int hp, double dx, double dy,
 int* neighbour, int Nside) {
 }
 */

/*
 static int add_hp(int** healpixes, int* nhp, int hp) {}
 */

struct neighbour_dirn {
    double x, y;
    double dx, dy;
};

int healpix_get_neighbours_within_range(double* xyz, double range, int* out_healpixes,
										//int maxhp,
										int Nside) {
	int hp;
	int i,j;
	double fx, fy;
	int nhp = 0;

    // HACK -- temp array to avoid cleverly avoiding duplicates
    int healpixes[100];

	hp = xyzarrtohealpixf(xyz, Nside, &fx, &fy);
	healpixes[nhp] = hp;
	nhp++;

    {
        struct neighbour_dirn dirs[] = {
            // edges
            { fx, 0,  0, -1 },
            { fx, 1,  0,  1 },
            { 0 , fy,-1,  0 },
            { 1 , fy, 1,  0 },
            // bottom corner
            { 0, 0, -1,  1 },
            { 0, 0, -1,  0 },
            { 0, 0, -1, -1 },
            { 0, 0,  0, -1 },
            { 0, 0,  1, -1 },
            // right corner
            { 1, 0,  1,  1 },
            { 1, 0,  1,  0 },
            { 1, 0,  1, -1 },
            { 1, 0,  0, -1 },
            { 1, 0, -1, -1 },
            // left corner
            { 0, 1,  1,  1 },
            { 0, 1,  0,  1 },
            { 0, 1, -1,  1 },
            { 0, 1, -1,  0 },
            { 0, 1, -1, -1 },
            // top corner
            { 1, 1, -1,  1 },
            { 1, 1,  0,  1 },
            { 1, 1,  1,  1 },
            { 1, 1,  1,  0 },
            { 1, 1,  1, -1 },
        };
        int ndirs = sizeof(dirs) / sizeof(struct neighbour_dirn);

        double ptx, pty, ptdx, ptdy;
        int pthp;

        for (i=0; i<ndirs; i++) {
            double pt[3];
            double ptstepx[3];
            double ptstepy[3];
            double across[3];
            double step = 0.1; // 1e-3;
            double d2;
            double stepdirx, stepdiry;
            struct neighbour_dirn* dir = dirs+i;
            ptx = dir->x;
            pty = dir->y;
            ptdx = dir->dx;
            ptdy = dir->dy;

            // pt = point on the edge nearest to the query point.
            // FIXME -- check that this is true, esp in the polar regions!
            healpix_to_xyzarr(hp, Nside, ptx, pty, pt);
            d2 = distsq(pt, xyz, 3);

            // delta vector should be outside the healpix
            assert((ptx+step*ptdx < 0) ||
                   (ptx+step*ptdx > 1) ||
                   (pty+step*ptdy < 0) ||
                   (pty+step*ptdy > 1));

            if (d2 > range*range)
                continue;

            // compute dx and dy directions that are toward the interior of
            // the healpix.
            stepdirx = (ptx < step) ? 1 : -1;
            stepdiry = (pty < step) ? 1 : -1;

            // take steps in those directions.
            healpix_to_xyzarr(hp, Nside, ptx + stepdir * step, pty, ptstepx);
            healpix_to_xyzarr(hp, Nside, ptx, pty + stepdir * step, ptstepy);

            // convert the steps into dx,dy vectors.
            for (j=0; j<3; j++) {
                ptstepx[j] = stepdirx * (ptstepx[j] - pt[j]);
                ptstepx[j] = stepdiry * (ptstepy[j] - pt[j]);
            }

            // take a small step in the specified direction.
            for (j=0; j<3; j++)
                across[j] = pt[j] + ptdx * ptstepx[j] + ptdy * ptstepy[j];

            // see which healpix is at the end of the step.
            normalize_3(across);
            pthp = xyzarrtohealpix(across, Nside);

            healpixes[nhp] = pthp;
            nhp++;
        }
    }

    printf("in range:");
	for (i=0; i<nhp; i++)
        printf(" %i", healpixes[i]);
    printf("\n");

	// Remove duplicates...
	for (i=0; i<nhp; i++) {
		for (j=i+1;  j<nhp; j++) {
            int k;
            printf("comparing %i to %i: vals %i, %i.  list",
                   i, j, healpixes[i], healpixes[j]);
            for (k=0; k<nhp; k++)
                printf(" %i", healpixes[k]);
            printf("\n");

			if (healpixes[i] == healpixes[j]) {
				int k;
				for (k=j+1; k<nhp; k++)
					healpixes[k-1] = healpixes[k];
				nhp--;
				i=-1;
				break;
			}
		}
	}

	for (i=0; i<nhp; i++)
        out_healpixes[i] = healpixes[i];

	return nhp;
}


