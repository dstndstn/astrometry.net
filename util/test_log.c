/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include "log.h"

#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <string.h>

#include "cutest.h"
#include "bl.h"
#include "ioutils.h"

#define STRING1A "I'm thread 1 -- you should see this message."
#define STRING1B "I'm thread 1 -- you should NOT see this message."
#define STRING2A "I'm thread 2 -- you should see this message."
#define STRING2B "I'm thread 2 -- you should see this message B."

void* thread1(void* v) {
    FILE* flog = v;
    logmsg("I'm thread 1.\n");
    log_set_level(LOG_MSG);
    log_to(flog);
    logmsg("%s\n", STRING1A);
    sleep(1);
    logverb("%s\n", STRING1B);
    return NULL;
}

void* thread2(void* v) {
    FILE* flog = v;
    logmsg("I'm thread 2.\n");
    log_set_level(LOG_VERB);
    log_to(flog);
    logmsg("%s\n", STRING2A);
    logverb("%s\n", STRING2B);
    return NULL;
}

typedef struct {
    int magic;
} teststruc_t;

static void logfunc(void* baton, enum log_level loglvl, const char* file, int line, const char* func, const char* format, va_list va) {
    teststruc_t* t = baton;
    printf("Magic: %i.  %s:%i(%s): level %i ", t->magic, file, line, func, loglvl);
    vprintf(format, va);
}

void test_log_ts(CuTest* tc) {
    pthread_t t1, t2;
    FILE *f1, *f2;
    char *fn1, *fn2;
    sl* lst;

    log_init(LOG_VERB);
    logmsg("Logging initialized.\n");

    log_set_thread_specific();
    logmsg("Logging set thread specific.\n");

    fn1 = create_temp_file("log", NULL);
    fn2 = create_temp_file("log", NULL);

    logmsg("File 1 is %s\n", fn1);
    logmsg("File 2 is %s\n", fn2);

    f1 = fopen(fn1, "w");
    f2 = fopen(fn2, "w");

    CuAssertIntEquals(tc, 0, pthread_create(&t1, NULL, thread1, f1));
    CuAssertIntEquals(tc, 0, pthread_create(&t2, NULL, thread2, f2));

    CuAssertIntEquals(tc, 0, pthread_join(t1, NULL));
    CuAssertIntEquals(tc, 0, pthread_join(t2, NULL));

    fclose(f1);
    fclose(f2);

    lst = file_get_lines(fn1, FALSE);
    CuAssertIntEquals(tc, 0, strcmp(sl_get(lst, 0), STRING1A));
    CuAssertIntEquals(tc, 1, sl_size(lst));
    sl_free2(lst);

    lst = file_get_lines(fn2, FALSE);
    CuAssertIntEquals(tc, 0, strcmp(sl_get(lst, 0), STRING2A));
    CuAssertIntEquals(tc, 0, strcmp(sl_get(lst, 1), STRING2B));
    CuAssertIntEquals(tc, 2, sl_size(lst));
    sl_free2(lst);

    unlink(fn1);
    unlink(fn2);
    free(fn1);
    free(fn2);
}



void test_log_func(CuTest* tc) {
    log_init(LOG_VERB);

    teststruc_t ts;
    ts.magic = 42;
    log_use_function(logfunc, &ts);
    log_to(NULL);

    logmsg("Testing 1 2 3\n");
    logdebug("Testing 1 2 3\n");

    log_use_function(NULL, NULL);
    log_to(stdout);

    logmsg("Testing 1 2 3 4\n");
    logdebug("Testing 1 2 3 4\n");

}

