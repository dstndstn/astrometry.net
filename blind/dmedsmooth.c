/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Michael Blanton, Keir Mierle, David W. Hogg,
  Sam Roweis and Dustin Lang.

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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/param.h>

#include "simplexy-common.h"

/*
 * dmedsmooth.c
 *
 * Median smooth an image -- actually, compute median values for a grid of points,
 * then interpolate.
 *
 * Mike Blanton
 * 1/2006 */


float dselip(unsigned long k, unsigned long n, float *arr);

int dmedsmooth(float *image,
               int nx,
               int ny,
               int halfbox,
               float *smooth)
{
	int i, j, ip, jp, ist, jst, nb, ind, jnd, sp;
	int xoff, yoff, nm, nxgrid, nygrid;
	int ypsize, ymsize, xpsize, xmsize;
	float dx, dy, xkernel, ykernel;

        float *arr = NULL;
        int *xgrid = NULL;
        int *ygrid = NULL;
        float *grid = NULL;
        int *xlo = NULL;
        int *xhi = NULL;
        int *ylo = NULL;
        int *yhi = NULL;

	/* get grids */
	sp = halfbox;
	nxgrid = nx / sp + 2;

	// "xgrid" are the centers.
	// "xlo" are the (inclusive) lower-bounds
	// "xhi" are the (inclusive) upper-bounds
	// the grid cells may overlap.
	xgrid = (int *) malloc((size_t)nxgrid * sizeof(int));
	xlo = (int *) malloc((size_t)nxgrid * sizeof(int));
	xhi = (int *) malloc((size_t)nxgrid * sizeof(int));
	xoff = (nx - 1 - (nxgrid - 3) * sp) / 2;
	for (i = 1;i < nxgrid - 1;i++)
		xgrid[i] = (i - 1) * sp + xoff;
	xgrid[0] = xgrid[1] - sp;
	xgrid[nxgrid - 1] = xgrid[nxgrid - 2] + sp;
	for (i = 0;i < nxgrid;i++) {
		xlo[i] = MAX(xgrid[i] - sp, 0);
		xhi[i] = MIN(xgrid[i] + sp, nx-1);
	}

	nygrid = ny / sp + 2;
	ylo = (int *) malloc(nygrid * sizeof(int));
	yhi = (int *) malloc(nygrid * sizeof(int));
	ygrid = (int *) malloc(nygrid * sizeof(int));
	yoff = (ny - 1 - (nygrid - 3) * sp) / 2;
	for (i = 1;i < nygrid - 1;i++)
		ygrid[i] = (i - 1) * sp + yoff;
	ygrid[0] = ygrid[1] - sp;
	ygrid[nygrid - 1] = ygrid[nygrid - 2] + sp;

	for (i = 0;i < nygrid;i++) {
		ylo[i] = MAX(ygrid[i] - sp, 0);
		yhi[i] = MIN(ygrid[i] + sp, ny-1);
	}

	// the median-filtered image (subsampled on a grid).
	grid = (float *) malloc((size_t)(nxgrid * nygrid) * sizeof(float));

	arr = (float *) malloc((size_t)((sp * 2 + 5) * (sp * 2 + 5)) * sizeof(float));

	for (j=0; j<nygrid; j++) {
		for (i=0; i<nxgrid; i++) {
			nb = 0;
			for (jp=ylo[j]; jp<=yhi[j]; jp++)
				for (ip=xlo[i]; ip<=xhi[i]; ip++) {
					arr[nb] = image[ip + jp * nx];
					nb++;
				}
			if (nb > 1) {
				nm = nb / 2;
				grid[i + j*nxgrid] = dselip(nm, nb, arr);
			} else {
				grid[i + j*nxgrid] = image[(long) xlo[i] + ((long) ylo[j]) * nx];
			}
		}
	}
	FREEVEC(xlo);
	FREEVEC(ylo);
	FREEVEC(xhi);
	FREEVEC(yhi);
	FREEVEC(arr);

	for (j = 0;j < ny;j++)
		for (i = 0;i < nx;i++)
			smooth[i + j*nx] = 0.;
	for (j = 0;j < nygrid;j++) {
		jst = (long) ( (float) ygrid[j] - sp * 1.5);
		jnd = (long) ( (float) ygrid[j] + sp * 1.5);
		if (jst < 0)
			jst = 0;
		if (jnd > ny - 1)
			jnd = ny - 1;
		ypsize = sp;
		ymsize = sp;
		if (j == 0)
			ypsize = ygrid[1] - ygrid[0];
		if (j == 1)
			ymsize = ygrid[1] - ygrid[0];
		if (j == nygrid - 2)
			ypsize = ygrid[nygrid - 1] - ygrid[nygrid - 2];
		if (j == nygrid - 1)
			ymsize = ygrid[nygrid - 1] - ygrid[nygrid - 2];
		for (i = 0;i < nxgrid;i++) {
			ist = (long) ( (float) xgrid[i] - sp * 1.5);
			ind = (long) ( (float) xgrid[i] + sp * 1.5);
			if (ist < 0)
				ist = 0;
			if (ind > nx - 1)
				ind = nx - 1;
			xpsize = sp;
			xmsize = sp;
			if (i == 0)
				xpsize = xgrid[1] - xgrid[0];
			if (i == 1)
				xmsize = xgrid[1] - xgrid[0];
			if (i == nxgrid - 2)
				xpsize = xgrid[nxgrid - 1] - xgrid[nxgrid - 2];
			if (i == nxgrid - 1)
				xmsize = xgrid[nxgrid - 1] - xgrid[nxgrid - 2];

			for (jp = jst;jp <= jnd;jp++) {
				dy = ((float) jp - ygrid[j]);
				ykernel = 0.;
				if (dy > -1.5*ymsize && dy <= -0.5*ymsize)
					ykernel = 0.5 * (dy / ymsize + 1.5) * (dy / ymsize + 1.5);
				else if (dy > -0.5*ymsize && dy < 0.)
					ykernel = -(dy * dy / ymsize / ymsize - 0.75);
				else if (dy < 0.5*ypsize && dy >= 0.)
					ykernel = -(dy * dy / ypsize / ypsize - 0.75);
				else if (dy >= 0.5*ypsize && dy < 1.5*ypsize)
					ykernel = 0.5 * (dy / ypsize - 1.5) * (dy / ypsize - 1.5);
				for (ip = ist;ip <= ind;ip++) {
					dx = ((float) ip - xgrid[i]);
					xkernel = 0.;
					if (dx > -1.5*xmsize && dx <= -0.5*xmsize)
						xkernel = 0.5 * (dx / xmsize + 1.5) * (dx / xmsize + 1.5);
					else if (dx > -0.5*xmsize && dx < 0.)
						xkernel = -(dx * dx / xmsize / xmsize - 0.75);
					else if (dx < 0.5*xpsize && dx >= 0.)
						xkernel = -(dx * dx / xpsize / xpsize - 0.75);
					else if (dx >= 0.5*xpsize && dx < 1.5*xpsize)
						xkernel = 0.5 * (dx / xpsize - 1.5) * (dx / xpsize - 1.5);
					smooth[ip + jp*nx] += xkernel * ykernel * grid[i + j * nxgrid];
				}
			}
		}
	}

	FREEVEC(grid);
	FREEVEC(xgrid);
	FREEVEC(ygrid);

	return 1;
}
