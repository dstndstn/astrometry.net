/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>

#include "ioutils.h"
#include "errors.h"

const char* OPTIONS = "hv:c:";

void printHelp(char* progname) {
    fprintf(stderr, "\nUsage: %s <desired-length> <input file> [<input file> ...]\n"
            "    The file will be padded in-place.\n"
            "\n"
            "    By default, the file is padded with zeros, but:\n"
            "      [-v <numerical value of character to pad with>]\n"
            "      [-c <character to pad with>]\n"
            "\n", progname);
}


int main(int argc, char** args) {
    int argchar;
    char* progname = args[0];
    char* infn;
    size_t padtolen;
    char padchar = '\0';
    int nargs = 0;
    int i;

    while ((argchar = getopt (argc, args, OPTIONS)) != -1) {
        switch (argchar) {
        case 'v':
            padchar = atoi(optarg);
            printf("Padding with value %i, character \"%c\"\n", (int)padchar, padchar);
            break;
        case 'c':
            padchar = optarg[0];
            printf("Padding with character \"%c\"\n", padchar);
            break;
        case 'h':
        default:
            printHelp(progname);
            exit(-1);
        }
    }
    if (optind < argc) {
        nargs = argc - optind;
        args += optind;
    }
    if (nargs < 2) {
        printHelp(progname);
        exit(-1);
    }
    padtolen = atol(args[0]);

    for (i=1; i<nargs; i++) {
        infn = args[i];
        printf("Padding file \"%s\" to length %lli.\n", infn, (long long int)padtolen);
        if (pad_file(infn, padtolen, padchar)) {
            ERROR("Failed to pad file");
            exit(-1);
        }
    }
    return 0;
}
