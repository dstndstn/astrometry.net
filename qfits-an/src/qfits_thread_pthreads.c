/**
 This file was added to the QFITS library by Astrometry.net

 Copyright 2009 Dustin Lang

 The Astrometry.net suite is free software; you can redistribute it
 and/or modify it under the terms of the GNU General Public License as
 published by the Free Software Foundation, version 2.

 The Astrometry.net suite is distributed in the hope that it will be
 useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with the Astrometry.net suite; if not, write to the Free Software
 Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
 */

#include <pthread.h>

#include "qfits_thread.h"

//#define DEBUG_LOCKS 1

#ifdef DEBUG_LOCKS
//#define debug(args...) fprintf(stderr, args)

#include <sys/time.h>
#include <time.h>
#include <stdio.h>

static double timenow() {
    struct timeval tv;
    if (gettimeofday(&tv, NULL)) {
        return -1.0;
    }
    return (double)(tv.tv_sec - 3600*24*365*30) + tv.tv_usec * 1e-6;
}

void qfits_lock_init(qfits_lock_t* lock) {
	pthread_mutex_init(lock, NULL);
}

void qfits_lock_lock(qfits_lock_t* lock) {
	double t0 = timenow();
	pthread_mutex_lock(lock);
	double t1 = timenow();
	fprintf(stderr, "qfits_lock_lock(%p) took %g sec\n", lock, t1-t0);
}

void qfits_lock_unlock(qfits_lock_t* lock) {
	double t0 = timenow();
	pthread_mutex_unlock(lock);
	double t1 = timenow();
	fprintf(stderr, "qfits_lock_unlock(%p) took %g sec\n", lock, t1-t0);
}

#else
//#define debug(args...)

void qfits_lock_init(qfits_lock_t* lock) {
	pthread_mutex_init(lock, NULL);
}

void qfits_lock_lock(qfits_lock_t* lock) {
	pthread_mutex_lock(lock);
}

void qfits_lock_unlock(qfits_lock_t* lock) {
	pthread_mutex_unlock(lock);
}

#endif


