/*
 This file is part of the Astrometry.net suite.
 Copyright 2007 Dustin Lang.

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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <math.h>
#include <errno.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>
#include <libgen.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>

int main(int argc, char** args) {
    int rtn;
    char** child_argv;
    int be_daemon = 1;

    if (argc >= 2 && strcmp(args[1], "-D")==0) {
        be_daemon = 0;
        args++;
        argc--;
    }

    child_argv = calloc(argc, sizeof(char*));
    memcpy(child_argv, args+1, (argc-1) * sizeof(char*));

    /*
     int i;
     printf("Child args:\n");
     for (i=0; i<argc; i++) {
     printf("  %i: %s\n", i, child_argv[i]);
     }
     */

    if (be_daemon) {
        fprintf(stderr, "Daemon starting.\n");
        if (daemon(0, 0) == -1) {
            fprintf(stderr, "Failed to set daemon process: %s\n", strerror(errno));
            exit(-1);
        }
    }

    rtn = execvp(child_argv[0], child_argv);

    fprintf(stderr, "execvp() failed: %s\n", strerror(errno));
    return rtn;
}


