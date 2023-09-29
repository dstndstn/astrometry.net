/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <stdlib.h>
#include <stdio.h>

#include <netpbm/pam.h>
int main(int argc, char** args) {
    struct pam img;
    pm_init(args[0], 0);
    img.size = 42;
    printf("the answer is %i\n", img.size);
    return 0;
}
