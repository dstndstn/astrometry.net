/*
 This file is part of the Astrometry.net suite.
 Copyright 2008 Dustin Lang.

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

#ifndef ENGINE_H
#define ENGINE_H

#include <stdio.h>

#include "blind.h"
#include "bl.h"
#include "an-bool.h"
#include "index.h"

struct engine {
    // search paths (directories)
	sl* index_paths;

    // contains "index_t" objects.
	// if "inparallel" is not set, they will be "metadata-only" until
	// they need to be loaded.
    pl* indexes;

	// indexes that need to be freed
    pl* free_indexes;
	// multiindexes that need to be freed
    pl* free_mindexes;

	il* ibiggest;
	il* ismallest;
    il* default_depths;
	double sizesmallest;
	double sizebiggest;
	anbool inparallel;
	double minwidth;
	double maxwidth;
    float cpulimit;
    char* cancelfn;
    char* solvedfn;
};
typedef struct engine engine_t;

struct job_t {
	dl* scales;
	il* depths;
    anbool include_default_scales;
    double ra_center;
    double dec_center;
    double search_radius;
    anbool use_radec_center;
    blind_t bp;
};
typedef struct job_t job_t;

engine_t* engine_new();
void engine_add_search_path(engine_t* engine, char* path);
char* engine_find_index(engine_t*, char* name);
// note that "path" must be a full path name.
int engine_add_index(engine_t* engine, char* path);
// look in all the search path directories for index files.
int engine_autoindex_search_paths(engine_t* engine);
int engine_parse_config_file_stream(engine_t* engine, FILE* fconf);
int engine_parse_config_file(engine_t* engine, const char* fn);
int engine_run_job(engine_t* engine, job_t* job);
void engine_free(engine_t* engine);

job_t* engine_read_job_file(engine_t* engine, const char* jobfn);
int job_set_base_dir(job_t* job, const char* dir);
int job_set_input_base_dir(job_t* job, const char* dir);
int job_set_output_base_dir(job_t* job, const char* dir);
void job_set_cancel_file(job_t* job, const char* fn);
void job_set_solved_file(job_t* job, const char* fn);
void job_free(job_t* job);


#endif
