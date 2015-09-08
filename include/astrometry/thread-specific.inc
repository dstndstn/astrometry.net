/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#include <pthread.h>

#define GLUE2(x, y) x ## _ ## y
#define GLUE(x, y) GLUE2(x, y)
#define TSMANGLE(x) GLUE(TSNAME, x)

static pthread_key_t TSMANGLE(key);
static pthread_once_t TSMANGLE(key_once) = PTHREAD_ONCE_INIT;

static void TSMANGLE(make_key)() {
    pthread_key_create(&TSMANGLE(key), NULL);
}

static void* TSMANGLE(get_key)(void* initdata) {
    void *ptr;
    pthread_once(&TSMANGLE(key_once), TSMANGLE(make_key));
    ptr = pthread_getspecific(TSMANGLE(key));
    if (!ptr) {
        ptr = TSMANGLE(init_key)(initdata);
        pthread_setspecific(TSMANGLE(key), ptr);
    }
    return ptr;
}

