/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
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
