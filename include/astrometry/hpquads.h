/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef HPQUADS_H
#define HPQUADS_H

#include "astrometry/an-bool.h"
#include "astrometry/starkd.h"
#include "astrometry/codefile.h"
#include "astrometry/quadfile.h"

int hpquads(startree_t* starkd,
            codefile_t* codes,
            quadfile_t* quads,
            int Nside,
            double scale_min_arcmin,
            double scale_max_arcmin,
            int dimquads,
            int passes,
            int Nreuses,
            int Nloosen,
            int id,
            anbool scanoccupied,

            void* sort_data,
            int (*sort_func)(const void*, const void*),
            int sort_size,

            char** args, int argc);

int hpquads_files(const char* skdtfn,
                  const char* codefn,
                  const char* quadfn,
                  int Nside,
                  double scale_min_arcmin,
                  double scale_max_arcmin,
                  int dimquads,
                  int passes,
                  int Nreuses,
                  int Nloosen,
                  int id,
                  anbool scanoccupied,

                  void* sort_data,
                  int (*sort_func)(const void*, const void*),
                  int sort_size,

                  char** args, int argc);

#endif



