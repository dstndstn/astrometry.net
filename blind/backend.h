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

#ifndef BACKEND_H
#define BACKEND_H

#include <stdio.h>

#include "bl.h"
#include "an-bool.h"
#include "index.h"

struct backend {
    // search paths (directories)
	sl* index_paths;

    // contains "index_meta_t" objects.
	bl* indexmetas;
    // if "inparallel" is set: contains "index_t" objects.
    pl* indexes;

	il* ibiggest;
	il* ismallest;
    il* default_depths;
	double sizesmallest;
	double sizebiggest;
	bool inparallel;
	double minwidth;
	double maxwidth;
    int cpulimit;
    char* cancelfn;
    bool verbose;
};
typedef struct backend backend_t;

struct job_t {
	dl* scales;
	il* depths;
    bool include_default_scales;
    double quad_sizefraction_min;
    double quad_sizefraction_max;
    blind_t bp;
};
typedef struct job_t job_t;

backend_t* backend_new();
void backend_free(backend_t* backend);
int backend_parse_config_file_stream(backend_t* backend, FILE* fconf);
int backend_parse_config_file(backend_t* backend, char* fn);
job_t* backend_read_job_file(backend_t* backend, const char* jobfn);
//int backend_set_base_dir(backend_t* backend, const char* dir);
int backend_run_job(backend_t* backend, job_t* job);
void job_free(job_t* job);


#endif
