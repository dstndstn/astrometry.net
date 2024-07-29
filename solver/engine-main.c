/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

/**
 * Accepts an augmented xylist that describes a field or set of fields to solve.
 * Reads a config file to find local indices, and merges information about the
 * indices with the job description to create an input file for 'onefield'.  Runs
 * and merges the results.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <time.h>
#include <libgen.h>
#include <getopt.h>
#include <dirent.h>
#include <assert.h>
#include <glob.h>

// Some systems (Solaris) don't have these glob symbols.  Don't really need.
#ifndef GLOB_BRACE
#define GLOB_BRACE 0
#endif
#ifndef GLOB_TILDE
#define GLOB_TILDE 0
#endif

#include "os-features.h"
#include "tic.h"
#include "fileutils.h"
#include "ioutils.h"
#include "bl.h"
#include "an-bool.h"
#include "solver.h"
#include "math.h"
#include "fitsioutils.h"
#include "solverutils.h"
#include "onefield.h"
#include "log.h"
#include "errors.h"
#include "engine.h"
#include "an-opts.h"
#include "gslutils.h"

#include "datalog.h"

static an_option_t myopts[] = {
    {'h', "help", no_argument, NULL, "print this help"},
    {'v', "verbose", no_argument, NULL, "+verbose"},
    {'c', "config",  required_argument, "file",
     "Use this config file (default: \"astrometry.cfg\" in the directory ../etc/ relative to the directory containing the \"astrometry-engine\" executable); 'none' for no config file"},
    {'d', "base-dir",  required_argument, "dir", 
     "set base directory of all output filenames."},
    {'C', "cancel",  required_argument, "file", 
     "quit solving if this file appears" },
    {'s', "solved",  required_argument, "file",
     "write to this file when a field is solved"},
    {'E', "to-stderr", no_argument, NULL,
     "send log message to stderr"},
    {'f', "inputs-from", required_argument, "file",
     "read input filenames from the given file, \"-\" for stdin"},
    {'i', "index", required_argument, "file(s)",
     "use the given index files (in addition to any specified in the config file); put in quotes to use wildcards, eg: \" -i 'index-*.fits' \""},
    {'I', "index-dir", required_argument, "directory",
     "search for index files in the given directory (in addition to any specified in the config file)"},
    {'p', "in-parallel", no_argument, NULL,
     "run the index files in parallel"},
    {'D', "data-log file", required_argument, "file",
     "log data to the given filename"},
    {'j', "job-id", required_argument, "jobid",
     "IGNORED; purely to allow process to contain the job id!"},
};

static void print_help(const char* progname, bl* opts) {
    printf("Usage:   %s [options] <augmented xylist (axy) file(s)>\n", progname);
    opts_print_help(opts, stdout, NULL, NULL);
}

FILE* datalogfid = NULL;
static void close_datalogfid() {
    if (datalogfid) {
        data_log_end();
        if (fclose(datalogfid)) {
            SYSERROR("Failed to close data log file");
        }
    }
}

int main(int argc, char** args) {
    char* default_configfn = "astrometry.cfg";
    char* default_config_path = "../etc";

    int c;
    char* configfn = NULL;
    int i;
    engine_t* engine;
    char* mydir = NULL;
    char* basedir = NULL;
    char* me;
    anbool help = FALSE;
    sl* strings = sl_new(4);
    char* cancelfn = NULL;
    char* solvedfn = NULL;
    int loglvl = LOG_MSG;
    anbool tostderr = FALSE;
    char* infn = NULL;
    FILE* fin = NULL;
    anbool fromstdin = FALSE;

    bl* opts = opts_from_array(myopts, sizeof(myopts)/sizeof(an_option_t), NULL);
    sl* index_files = sl_new(4);
    sl* index_dirs = sl_new(4);

    char* datalog = NULL;

    engine = engine_new();

    while (1) {
        c = opts_getopt(opts, argc, args);
        if (c == -1)
            break;
        switch (c) {
	case 'j':
	    break;
        case 'D':
            datalog = optarg;
            break;
        case 'p':
            engine->inparallel = TRUE;
            break;
        case 'i':
            sl_append(index_files, optarg);
            break;
        case 'I':
            sl_append(index_dirs, optarg);
            break;
        case 'd':
            basedir = optarg;
            break;
        case 'f':
            infn = optarg;
            fromstdin = streq(infn, "-");
            break;
        case 'E':
            tostderr = TRUE;
            break;
        case 'h':
            help = TRUE;
            break;
        case 'v':
            loglvl++;
            break;
        case 's':
            solvedfn = optarg;
        case 'C':
            cancelfn = optarg;
            break;
        case 'c':
            configfn = strdup(optarg);
            break;
        case '?':
            break;
        default:
            printf("Unknown flag %c\n", c);
            exit( -1);
        }
    }

    if (optind == argc && !infn) {
        // Need extra args: filename
        printf("You must specify at least one input file!\n\n");
        help = TRUE;
    }
    if (help) {
        print_help(args[0], opts);
        exit(0);
    }
    bl_free(opts);

    gslutils_use_error_system();

    log_init(loglvl);
    if (tostderr)
        log_to(stderr);

    if (datalog) {
        datalogfid = fopen(datalog, "wb");
        if (!datalogfid) {
            SYSERROR("Failed to open data log file \"%s\" for writing", datalog);
            return -1;
        }
        atexit(close_datalogfid);
        data_log_init(100);
        data_log_enable_all();
        data_log_to(datalogfid);
        data_log_start();
    }

    if (infn) {
        logverb("Reading input filenames from %s\n", (fromstdin ? "stdin" : infn));
        if (!fromstdin) {
            fin = fopen(infn, "rb");
            if (!fin) {
                ERROR("Failed to open file %s for reading input filenames", infn);
                exit(-1);
            }
        } else
            fin = stdin;
    }

    // directory containing the 'engine' executable:
    me = find_executable(args[0], NULL);
    if (!me)
        me = strdup(args[0]);
    mydir = sl_append(strings, dirname(me));
    free(me);

    // Read config file
    if (!configfn) {
        int i;
        sl* trycf = sl_new(4);
        sl_appendf(trycf, "%s/%s/%s", mydir, default_config_path, default_configfn);
        // if I'm in /usr/bin, look for config file in /etc
        if (streq(mydir, "/usr/bin")) {
            sl_appendf(trycf, "/etc/%s", default_configfn);
        }
        sl_appendf(trycf, "%s/%s", mydir, default_configfn);
        sl_appendf(trycf, "./%s", default_configfn);
        sl_appendf(trycf, "./%s/%s", default_config_path, default_configfn);
        for (i=0; i<sl_size(trycf); i++) {
            char* cf = sl_get(trycf, i);
            if (file_exists(cf)) {
                configfn = strdup(cf);
                logverb("Using config file \"%s\"\n", cf);
                break;
            } else {
                logverb("Config file \"%s\" doesn't exist.\n", cf);
            }
        }
        if (!configfn) {
            char* cflist = sl_join(trycf, "\n  ");
            logerr("Couldn't find config file: tried:\n  %s\n", cflist);
            free(cflist);
        }
        sl_free2(trycf);
    }

    if (!streq(configfn, "none")) {
        if (engine_parse_config_file(engine, configfn)) {
            logerr("Failed to parse (or encountered an error while interpreting) config file \"%s\"\n", configfn);
            exit( -1);
        }
    }

    if (sl_size(index_dirs)) {
        // save the engine_t state, add the search paths & auto-index them, then revert.
        sl* saved_paths = engine->index_paths;
        engine->index_paths = index_dirs;
        if (engine_autoindex_search_paths(engine)) {
            logerr("Failed to search directories for index files: [%s]", sl_join(index_dirs, ", "));
            exit( -1);
        }
        engine->index_paths = saved_paths;
    }

    if (sl_size(index_files)) {
        // Expand globs.
        for (i=0; i<sl_size(index_files); i++) {
            char* s = sl_get(index_files, i);
            glob_t myglob;
            int flags = GLOB_TILDE | GLOB_BRACE;
            if (glob(s, flags, NULL, &myglob)) {
                SYSERROR("Failed to expand wildcards in index-file path \"%s\"", s);
                exit(-1);
            }
            for (c=0; c<myglob.gl_pathc; c++) {
                if (engine_add_index(engine, myglob.gl_pathv[c])) {
                    ERROR("Failed to add index \"%s\"", myglob.gl_pathv[c]);
                    exit(-1);
                }
            }
            globfree(&myglob);
        }
    }

    if (!pl_size(engine->indexes)) {
        logerr("\n\n"
               "---------------------------------------------------------------------\n"
               "You must list at least one index in the config file (%s)\n\n"
               "See http://astrometry.net/use.html about how to get some index files.\n"
               "---------------------------------------------------------------------\n"
               "\n", configfn);
        exit(-1);
    }

    if (engine->minwidth <= 0.0 || engine->maxwidth <= 0.0) {
        logerr("\"minwidth\" and \"maxwidth\" in the config file %s must be positive!\n", configfn);
        exit(-1);
    }

    free(configfn);

    if (!il_size(engine->default_depths)) {
        parse_depth_string(engine->default_depths,
                           "10 20 30 40 50 60 70 80 90 100 "
                           "110 120 130 140 150 160 170 180 190 200");
    }

    engine->cancelfn = cancelfn;
    engine->solvedfn = solvedfn;

    i = optind;
    while (1) {
        char* jobfn;
        job_t* job;
        struct timeval tv1, tv2;

        if (infn) {
            // Read name of next input file to be read.
            logverb("\nWaiting for next input filename...\n");
            jobfn = read_string_terminated(fin, "\n\r\0", 3, FALSE);
            if (strlen(jobfn) == 0)
                break;
        } else {
            if (i == argc)
                break;
            jobfn = args[i];
            i++;
        }
        gettimeofday(&tv1, NULL);
        logmsg("Reading file \"%s\"...\n", jobfn);
        job = engine_read_job_file(engine, jobfn);
        if (!job) {
            ERROR("Failed to read job file \"%s\"", jobfn);
            exit(-1);
        }

	if (basedir) {
            logverb("Setting job's output base directory to %s\n", basedir);
            job_set_output_base_dir(job, basedir);
	}

        if (engine_run_job(engine, job))
            logerr("Failed to run_job()\n");

        job_free(job);
        gettimeofday(&tv2, NULL);
        logverb("Spent %g seconds on this field.\n", millis_between(&tv1, &tv2)/1000.0);
    }

    engine_free(engine);
    sl_free2(strings);
    sl_free2(index_files);
    sl_free2(index_dirs);

    if (fin && !fromstdin)
        fclose(fin);

    return 0;
}
