/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef WCS_PV2SIP_H
#define WCS_PV2SIP_H

#include "qfits_header.h"
#include "sip.h"

sip_t* wcs_pv2sip_header(qfits_header* hdr,
                         double* xy, int Nxy,
               
                         double stepsize,
                         double xlo, double xhi,
                         double ylo, double yhi,

                         int imageW, int imageH,
                         int order,
                         anbool forcetan,
                         int doshift);

int wcs_pv2sip(const char* wcsinfn, int ext,
               const char* wcsoutfn,
               anbool scamp_head_file,

               double* xy, int Nxy,
               
               double stepsize,
               double xlo, double xhi,
               double ylo, double yhi,

               int imageW, int imageH,
               int order,
               anbool forcetan,
               int doshift);

#endif
