/*
  This file is part of the Astrometry.net suite.
  Copyright 2009 Dustin Lang.

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

#ifndef AN_THREAD_PTHREADS_H
#define AN_THREAD_PTHREADS_H

#include <pthread.h>

#define AN_THREAD_DECLARE_ONCE(X) pthread_once_t X = PTHREAD_ONCE_INIT

#define AN_THREAD_DECLARE_STATIC_ONCE(X) static pthread_once_t X = PTHREAD_ONCE_INIT

#define AN_THREAD_CALL_ONCE(X, F) pthread_once(&X, F)

#define AN_THREAD_DECLARE_MUTEX(X) pthread_mutex_t X = PTHREAD_MUTEX_INITIALIZER

#define AN_THREAD_DECLARE_STATIC_MUTEX(X) static pthread_mutex_t X = PTHREAD_MUTEX_INITIALIZER

#define AN_THREAD_LOCK(X) pthread_mutex_lock(&X)

/*
DEBUG

#define AN_THREAD_DECLARE_STATIC_MUTEX(X) \
    static pthread_mutex_t X; \
static inline void init_mutex_ ## X() { \
    pthread_mutexattr_t attr; \
    pthread_mutexattr_init(&attr); \
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK); \
    pthread_mutex_init(&X, &attr); \
    pthread_mutexattr_destroy(&attr); \
} \
    static pthread_once_t X_once = PTHREAD_ONCE_INIT

 #define AN_THREAD_LOCK(X) {				 \
 pthread_once(&X_once, init_mutex_ ## X);	 \
 pthread_mutex_lock(&X);					 \
 }
*/

#define AN_THREAD_UNLOCK(X) pthread_mutex_unlock(&X)

#endif
