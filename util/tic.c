/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 2, or
  (at your option) any later version.

  The Astrometry.net suite is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the Astrometry.net suite ; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
*/

#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>

#include "tic.h"
#include "errors.h"
#include "log.h"

static time_t starttime;
static double starttime2;
static double startutime, startstime;

double timenow() {
    struct timeval tv;
    if (gettimeofday(&tv, NULL)) {
        ERROR("Failed to get time of day");
        return -1.0;
    }
    return (double)(tv.tv_sec - 3600*24*365*30) + tv.tv_usec * 1e-6;
}

double millis_between(struct timeval* tv1, struct timeval* tv2) {
	return
		(tv2->tv_usec - tv1->tv_usec)*1e-3 +
		(tv2->tv_sec  - tv1->tv_sec )*1e3;
}

void tic() {
	starttime = time(NULL);
	starttime2 = timenow();
	if (get_resource_stats(&startutime, &startstime, NULL)) {
		ERROR("Failed to get_resource_stats()");
		return;
	}
}

int get_resource_stats(double* p_usertime, double* p_systime, long* p_maxrss) {
	struct rusage usage;
	if (getrusage(RUSAGE_SELF, &usage)) {
		SYSERROR("Failed to get resource stats (getrusage)");
		return 1;
	}
	if (p_usertime) {
		*p_usertime = usage.ru_utime.tv_sec + 1e-6 * usage.ru_utime.tv_usec;
	}
	if (p_systime) {
		*p_systime = usage.ru_stime.tv_sec + 1e-6 * usage.ru_stime.tv_usec;
	}
	if (p_maxrss) {
		*p_maxrss = usage.ru_maxrss;
	}
	return 0;
}

void toc() {
	double utime, stime;
	long rss;
	//int dtime;
	double dtime2;
	//time_t endtime = time(NULL);
	//dtime = (int)(endtime - starttime);
	dtime2 = timenow() - starttime2;
	if (get_resource_stats(&utime, &stime, &rss)) {
		ERROR("Failed to get_resource_stats()");
		return;
	}
	logmsg("Used %g s user, %g s system (%g s total), %g s wall time since last check\n",
		   utime-startutime, stime-startstime, (utime + stime)-(startutime+startstime), dtime2);
}
