/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
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
