/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdio.h>
#include <getopt.h>
#include <ctype.h>
#include <string.h>

#include "an-opts.h"
#include "ioutils.h"
#include "bl.h"
#include "log.h"

void opts_print_help(bl* opts, FILE* fid,
                     void (*special_case)(an_option_t* opt, bl* allopts, int index,
                                          FILE* fid, void* extra), void* extra) {
    int i;
    for (i=0; i<bl_size(opts); i++) {
        an_option_t* opt = bl_access(opts, i);
        int nw = 0;
        sl* words;
        int j;
        if (opt->help) {
            if ((opt->shortopt >= 'a' && opt->shortopt <= 'z') ||
                (opt->shortopt >= 'A' && opt->shortopt <= 'Z') ||
                (opt->shortopt >= '0' && opt->shortopt <= '9'))
                nw += fprintf(fid, "  -%c / --%s", opt->shortopt, opt->name);
            else
                nw += fprintf(fid, "  --%s", opt->name);
            if (opt->has_arg == optional_argument)
                nw += fprintf(fid, " [<%s>]", opt->argname);
            else if (opt->has_arg == required_argument)
                nw += fprintf(fid, " <%s>", opt->argname);
            nw += fprintf(fid, ": ");
            if (!opt->help)
                continue;
            words = split_long_string(opt->help, 80-nw, 70, NULL);
            for (j=0; j<sl_size(words); j++)
                fprintf(fid, "%s%s\n", (j==0 ? "" : "          "), sl_get(words, j));
        } else if (special_case)
            special_case(opt, opts, i, fid, extra);
    }
}

int opts_getopt(bl* opts, int argc, char** argv) {
    size_t i, j, N;
    char* optstring;
    int c;
    struct option* longoptions;

    N = bl_size(opts);
    // create the short options string.
    optstring = malloc(3 * N + 1);
    j = 0;
    for (i=0; i<N; i++) {
        an_option_t* opt = bl_access(opts, i);
        if (!opt->shortopt)
            continue;
        if (iscntrl((unsigned)(opt->shortopt)))
            continue;
        optstring[j] = opt->shortopt;
        j++;
        if (opt->has_arg == no_argument)
            continue;
        optstring[j] = ':';
        j++;
        if (opt->has_arg == required_argument)
            continue;
        optstring[j] = ':';
        j++;
    }
    optstring[j] = '\0';
    // create long options.
    longoptions = calloc(N+1, sizeof(struct option));
    j = 0;
    for (i=0; i<N; i++) {
        an_option_t* opt = bl_access(opts, i);
        if (!opt->shortopt)
            continue;
        //if (iscntrl(opt->shortopt))
        //continue;
        longoptions[j].name = opt->name;
        longoptions[j].has_arg = opt->has_arg;
        longoptions[j].val = opt->shortopt;
        j++;
    }

    // DEBUG
    //printf("%s\n", optstring);

    c = getopt_long(argc, argv, optstring, longoptions, NULL);

    free(optstring);
    free(longoptions);

    return c;
}

bl* opts_from_array(const an_option_t* opts, int N, bl* lst) {
    int i;
    if (!lst)
        lst = bl_new(4, sizeof(an_option_t));
    for (i=0; i<N; i++)
        bl_append(lst, opts + i);
    return lst;
}


