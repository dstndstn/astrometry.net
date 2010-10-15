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

/**
 Computed SIP parameters given a set of corresponding points.

 If 'weights' is NULL, uniform weighting will be used.

 You MUST set:
   sip->a_order = sip->b_order = X;
   sip->ap_order = sip->bp_order = Y;
   sip->wcstan.imagew = W;
   sip->wcstan.imageh = H;
   sip->wcstan.crval = {RA,Dec};
   sip->wcstan.crpix = {x0,y0};

 */
void tweak2_from_correspondences(const double* fieldxy,
								 const double* refxyz,
								 const double* weights,
								 int N,
								 sip_t* sip);


/**
 Given an initial WCS solution, compute SIP polynomial distortions
 using an annealing-like strategy.  That is, it finds matches between
 image and reference catalog by searching within a radius, and that
 radius is small near a region of confidence, and grows as you move
 away.  That makes it possible to pick up more distant matches, but
 they are downweighted in the fit.  The annealing process reduces the
 slope of the growth of the matching radius with respect to the
 distance from the region of confidence.
 
 In Astrometry.net, the confidence region is the center of the quad
 that matched.

 fieldxy: source positions, x0,y0,x1,y1,...
 Nfield: number of source positions
 fieldjitter: standard deviation of field sources, in pixels;
 (FIXME: this should be per-source!)
 W, H: size of the image.
 indexradec: reference star positions, in degrees, ra0,dec0,ra1,dec1,...
 Nindex: number of reference stars
 indexjitter: standard deviation of reference sources, in arcsec.
 (FIXME: this should be per-source)
 quadcenter: the center of the quad that matched, in pixels.
 quadR2: the radius-squared, in pixels, of the quad that matched.
 distractors: how often you find unexpected test stars; we've always kept this fixed at 0.25
 logodds_bail: totally uninteresting; shouldn't really be here.  Set to, say, -100.
 sip_order: polynomial distortion order you want.  1=linear works.
 startwcs: initial WCS solution.  See util/sip.h : sip_wrap_tan() if you have a tan_t rather than a sip_t.
 destwcs: where to put the solution; NULL to allocate a new sip_t.
 newtheta: "theta" maps field stars to reference stars in the final matching that we produce.  Set this non-NULL to pull it out.
 newodds: this tells the confidence in the matches.  Use verify_logodds_to_weight() to turn these into a weight in [0,1].
 crpix: if you want to keep the reference point fixed, set this to a (2-element) array of the image reference position.
 */
sip_t* tweak2(const double* fieldxy, int Nfield,
			  double fieldjitter,
			  int W, int H,
			  const double* indexradec, int Nindex,
			  double indexjitter,
			  const double* quadcenter, double quadR2,
			  double distractors,
			  double logodds_bail,
			  int sip_order,
			  int sip_invorder,
			  const sip_t* startwcs,
			  sip_t* destwcs,
			  int** newtheta, double** newodds,
			  double* crpix,
			  double* p_logodds,
			  int* p_besti,
			  int* testperm, int startorder);


#endif

