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

#ifndef TIC_H
#define TIC_H

#include <time.h>
#include <sys/time.h>

void tic();
int get_resource_stats(double* p_usertime, double* p_systime, long* p_maxrss);
void toc();

double millis_between(struct timeval* tv1, struct timeval* tv2);

// Returns the number of seconds since (approximately) Jan 1, 2000 UTC.
// You probably only want to look at differences in the values returned by this function.
double timenow();

#endif
