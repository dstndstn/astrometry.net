/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef ONEFIELD_H
#define ONEFIELD_H

#include "astrometry/an-bool.h"
#include "astrometry/solver.h"
#include "astrometry/matchobj.h"
#include "astrometry/matchfile.h"
#include "astrometry/rdlist.h"
#include "astrometry/bl.h"

#define DEFAULT_QSF_LO 0.1
#define DEFAULT_QSF_HI 1.0

struct onefield_params {
    solver_t solver;

    anbool indexes_inparallel;

    double logratio_tosolve;

    // How many solving quads are required before we stop?
    int nsolves;
    int nsolves_sofar;

    // Filenames
    char *fieldfname;
    char *matchfname;
    char *indexrdlsfname;
    char *corr_fname;
    char* scamp_fname;

    // WCS filename template (sprintf format with %i for field number)
    char* wcs_template;

    // List of WCS filenames to run verification on.
    sl* verify_wcsfiles;

    // WCS instances to verify.  (sip_t structs)
    bl* verify_wcs_list;

    // Output solved file.
    char *solved_out;
    // Input solved file.
    char* solved_in;

    // Indexes to use (base filenames)
    sl* indexnames;

    // Indexes to use (index_t objects)
    pl* indexes;

    int index_options;

    // Quad size fraction: select indexes that contain quads of size fraction
    // [quad_size_fraction_lo, quad_size_fraction_hi] of the image size.
    double quad_size_fraction_lo;
    double quad_size_fraction_hi;

    // Fields to try
    il* fieldlist;

    // Which field in a multi-HDU xyls file is this?
    int fieldnum;
    // A unique ID for the whole multi-HDU xyls file.
    int fieldid;

    // xylist column names.
    char *xcolname, *ycolname;
    // FITS keyword to copy from xylist to matchfile.
    char *fieldid_key;

    // The fields to solve!
    xylist_t* xyls;

    // Output files
    matchfile* mf;
    rdlist_t* indexrdls;

    // extra fields to add to index rdls file:
    sl* rdls_tagalong;
    anbool rdls_tagalong_all;

    // internal use only: have I grabbed "all" rdls fields already?
    //anbool done_rdls_tagalong_all;

    // field to sort RDLS file by; prefix by "-" for descending order.
    char* sort_rdls;

    // extra fields to add from the xyls file:
    sl* xyls_tagalong;
    anbool xyls_tagalong_all;

    // List of MatchObjs with logodds >= logodds_tokeep
    bl* solutions;

    float cpulimit;
    float cpu_start;
    anbool hit_cpulimit;

    int timelimit;
    time_t time_start;
    anbool hit_timelimit;

    float total_cpulimit;
    float cpu_total_start;
    anbool hit_total_cpulimit;

    double total_timelimit;
    double time_total_start;
    anbool hit_total_timelimit;

    anbool single_field_solved;

    // filename for cancelling
    char* cancelfname;
    anbool cancelled;

    anbool best_hit_only;
};
typedef struct onefield_params onefield_t;

void onefield_set_field_file(onefield_t* bp, const char* fn);
void onefield_set_cancel_file(onefield_t* bp, const char* fn);
void onefield_set_solved_file(onefield_t* bp, const char* fn);
void onefield_set_solvedin_file(onefield_t* bp, const char* fn);
void onefield_set_solvedout_file(onefield_t* bp, const char* fn);
void onefield_set_match_file(onefield_t* bp, const char* fn);
void onefield_set_rdls_file(onefield_t* bp, const char* fn);
void onefield_set_scamp_file(onefield_t* bp, const char* fn);
void onefield_set_corr_file(onefield_t* bp, const char* fn);
void onefield_set_wcs_file(onefield_t* bp, const char* fn);
void onefield_set_xcol(onefield_t* bp, const char* x);
void onefield_set_ycol(onefield_t* bp, const char* x);

void onefield_add_verify_wcs(onefield_t* bp, sip_t* wcs);
void onefield_add_loaded_index(onefield_t* bp, index_t* ind);
void onefield_add_index(onefield_t* bp, const char* index);

void onefield_clear_verify_wcses(onefield_t* bp);
void onefield_clear_indexes(onefield_t* bp);
void onefield_clear_solutions(onefield_t* bp);

void onefield_add_field(onefield_t* bp, int field);
void onefield_add_field_range(onefield_t* bp, int lo, int hi);

void onefield_run(onefield_t* bp);

void onefield_init(onefield_t* bp);

void onefield_cleanup(onefield_t* bp);

int onefield_parameters_are_okay(onefield_t* bp, solver_t* sp);

int onefield_is_run_obsolete(onefield_t* bp, solver_t* sp);

void onefield_log_run_parameters(onefield_t* bp);

void onefield_free_matchobj(MatchObj* mo);

void onefield_matchobj_deep_copy(const MatchObj* mo, MatchObj* dest);

#endif
