/*
 This file is part of the Astrometry.net suite.
 Copyright 2010 Dustin Lang.

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

#ifndef TWEAK_2_H
#define TWEAK_2_H

#include "sip.h"

sip_t* tweak2(const double* fieldxy, int Nfield,
			  double fieldjitter,
			  int W, int H,
			  const double* indexradec, int Nindex,
			  double indexjitter,
			  const double* quadcenter, double quadR2,
			  double distractors,
			  double logodds_bail,
			  int sip_order,
			  const sip_t* startwcs,
			  sip_t* destwcs,
			  int** newtheta, double** newodds);


#endif

