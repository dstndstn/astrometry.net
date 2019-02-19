/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include "errors.h"
#include "cutest.h"

typedef struct {
    int magic;
} teststruc_t;

static void errfunc(void* baton, err_t* errstate, const char* file, int line, const char* func, const char* format, va_list va) {
    teststruc_t* t = baton;
    printf("Magic: %i.  %s:%i(%s): ", t->magic, file, line, func);
    vprintf(format, va);
    printf("\n");

    error_stack_add_entryv(errstate, file, line, func, format, va);
}

static void funky() {
    ERROR("I'm funky.");
}

static void errorprone(int n) {
    if (n > 0) {
        errorprone(n-1);
        ERROR("errorprone(%i) failed", n);
        return;
    }
    funky();
}


void test_err_func(CuTest* tc) {
    teststruc_t ts;
    ts.magic = 42;
    errors_use_function(errfunc, &ts);

    errorprone(4);

    printf("At end:\n");
    errors_print_stack(stdout);
}

