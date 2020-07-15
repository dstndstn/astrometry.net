/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <stdio.h>
#include <unistd.h>
#include "solvedfile.h"

int main(int argc, char** args) {
    int field = 1;
    char* fn;
    if (argc < 2) {
        printf("%s <solved-file>\n", args[0]);
        exit(-1);
    }
    fn = args[1];
    printf("Field %i solved: %s\n", field,
           solvedfile_get(fn, field) ? "yes" : "no");
    printf("Setting solved.\n");
    solvedfile_set(fn, field);
    printf("Field %i solved: %s\n", field,
           solvedfile_get(fn, field) ? "yes" : "no");
    return 0;
}
