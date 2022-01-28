/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#ifndef HEALPIX_H
#define HEALPIX_H

#include <sys/types.h>
#include <stdint.h>

#include "astrometry/keywords.h"

//#undef Const
//#define Const 

/**
 The HEALPix paper is here:
 http://iopscience.iop.org/0004-637X/622/2/759/pdf/0004-637X_622_2_759.pdf
 See:
 http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2005ApJ...622..759G&db_key=AST&high=41069202cf02947
*/

/**
   In this documentation we talk about "base healpixes": these are the big,
   top-level healpixes.  There are 12 of these, with indices [0, 11].

   We say "fine healpixes" or "healpixes" or "pixels" when we mean the fine-
   scale healpixes; there are Nside^2 of these in each base healpix,
   for a total of 12*Nside^2, indexed from zero.
 */

/**
   Some notes about the different indexing schemes:

   The healpix paper discusses two different ways to number healpixes, and
   there is a third way, which we prefer, which is (in my opinion) more
   sensible and easy.


   -RING indexing.  Healpixes are numbered first in order of decreasing DEC,
   then in order of increasing RA of the center of the pixel, ie:

   .       0       1       2       3
   .     4   5   6   7   8   9  10  11
   .  12  13  14  15  16  17  18  19
   .    20  21  22  23  24  25  26  27
   .  28  29  30  31  32  33  34  35
   .    36  37  38  39  40  41  42  43
   .      44      45      46      47

   Note that 12, 20 and 28 are part of base healpix 4, as is 27; it "wraps
   around".

   The RING index can be decomposed into the "ring number" and the index
   within the ring (called "longitude index").  Note that different rings
   contain different numbers of healpixes.  Also note that the ring number
   starts from 1, but the longitude index starts from zero.


   -NESTED indexing.  This only works for Nside parameters that are powers of
   two.  This scheme is hierarchical in the sense that each pair of bits of
   the index tells you where the pixel center is to finer and finer
   resolution.  It doesn't really show with Nside=2, but here it is anyway:

   .       3       7      11      15
   .     2   1   6   5  10   9  14  13
   .  19   0  23   4  27   8  31  12
   .    17  22  21  26  25  30  29  18
   .  16  35  20  39  24  43  28  47
   .    34  33  38  37  42  41  46  45
   .      32      36      40      44

   Note that all the base healpixes have the same pattern; they're just
   offset by factors of Nside^2.

   Here's a zoom-in of the first base healpix, turned 45 degrees to the
   right, for Nside=4:

   .   10  11  14  15
   .    8   9  12  13
   .    2   3   6   7
   .    0   1   4   5

   Note that the bottom-left block of 4 have the smallest values, and within
   that the bottom-left corner has the smallest value, followed by the
   bottom-right, top-left, then top-right.

   The NESTED index can't be decomposed into 'orthogonal' directions.


   -XY indexing.  This is arguably the most natural, at least for the
   internal usage of the healpix code.  Within each base healpix, the
   healpixes are numbered starting with 0 for the southmost pixel, then
   increasing first in the "y" (north-west), then in the "x" (north-east)
   direction.  In other words, within each base healpix there is a grid
   and we number the pixels "lexicographically" (mod a 135 degree turn).

   .       3       7      11      15
   .     1   2   5   6   9  10  13  14
   .  19   0  23   4  27   8  31  12
   .    18  21  22  25  26  29  30  17
   .  16  35  20  39  24  43  28  47
   .    33  34  37  38  41  42  45  46
   .      32      36      40      44

   Zooming in on the first base healpix, turning 45 degrees to the right,
   for Nside=4 we get:

   .    3   7  11  15
   .    2   6  10  14
   .    1   5   9  13
   .    0   4   8  12

   Notice that the numbers first increase from bottom to top (y), then left to
   right (x).

   The XY indexing can be decomposed into 'x' and 'y' coordinates
   (in case that wasn't obvious), where the above figure becomes (x,y):

   .    (0,3)  (1,3)  (2,3)  (3,3)
   .    (0,2)  (1,2)  (2,2)  (3,2)
   .    (0,1)  (1,1)  (2,1)  (3,1)
   .    (0,0)  (1,0)  (2,0)  (3,0)

   Note that "x" increases in the north-east direction, and "y" increases in
   the north-west direction.

   The major advantage to this indexing scheme is that it extends to
   fractional coordinates in a natural way: it is meaningful to talk about
   the position (x,y) = (0.25, 0.6) and you can compute its position.




   In this code, all healpix indexing uses the XY scheme.  If you want to
   use the other schemes you will have to use the conversion routines:
   .   healpix_xy_to_ring
   .   healpix_ring_to_xy
   .   healpix_xy_to_nested
   .   healpix_nested_to_xy
*/

// The maximum healpix Nside that leads to int-sized healpix indices.
// 12 * (13377+1)^2 > 2^31 (since we use signed ints)
// This corresponds to about 16 arcsec side length.
#define HP_MAX_INT_NSIDE 13377

/**
   Converts a healpix index from the XY scheme to the RING scheme.
*/
Const int healpix_xy_to_ring(int hp, int Nside);

/**
   Converts a healpix index from the RING scheme to the XY scheme.
*/
Const int healpix_ring_to_xy(int ring_index, int Nside);

/**
   Converts a healpix index from the XY scheme to the NESTED scheme.
 */
Const int healpix_xy_to_nested(int hp, int Nside);

/**
   Converts a healpix index from the NESTED scheme to the XY scheme.
 */
Const int healpix_nested_to_xy(int nested_index, int Nside);

/**
   Decomposes a RING index into the "ring number" (each ring contain
   healpixels of equal latitude) and "longitude index".  Pixels within a
   ring have longitude index starting at zero for the first pixel with
   RA >= 0.  Different rings contain different numbers of healpixels.
*/
void healpix_decompose_ring(int ring_index, int Nside,
							int* p_ring_number, int* p_longitude_index);

/**
   Composes a RING index given the "ring number" and "longitude index".

   Does NOT check that the values are legal!  Garbage in, garbage out.
*/
Const int healpix_compose_ring(int ring, int longind, int Nside);

/**
   Decomposes an XY index into the "base healpix" and "x" and "y" coordinates
   within that healpix.
*/
void healpix_decompose_xy(int finehp, int* bighp, int* x, int* y, int Nside);

void healpix_decompose_xyl(int64_t finehp, int* bighp, int* x, int* y, int Nside);

/**
   Composes an XY index given the "base healpix" and "x" and "y" coordinates
   within that healpix.
*/
Const int healpix_compose_xy(int bighp, int x, int y, int Nside);

Const int64_t healpix_compose_xyl(int bighp, int x, int y, int Nside);

/**
 Given (x,y) coordinates of resolution "nside" within a base-level
 healpixel, and an output resolution "outnside", returns the output
 (x,y) coordinates at the output resolution.
 */
void healpix_convert_xy_nside(int x, int y, int nside, int outnside,
							  int* outx, int* outy);

/**
 Given a healpix index (in the XY scheme) of resolution "nside", and
an output resolution "outnside", returns the healpix index at the
output resolution.
 */
void healpix_convert_nside(int hp, int nside, int outnside, int* outhp);
void healpix_convert_nsidel(int64_t hp, int nside, int outnside, int64_t* outhp);

/**
   Converts (RA, DEC) coordinates (in radians) to healpix index.
*/
Const int radectohealpix(double ra, double dec, int Nside);

int radectohealpixf(double ra, double dec, int Nside, double* dx, double* dy);

Const int64_t radectohealpixl(double ra, double dec, int Nside);

int64_t radectohealpixlf(double ra, double dec, int Nside, double* dx, double* dy);

/**
   Converts (RA, DEC) coordinates (in degrees) to healpix index.
*/
Const int radecdegtohealpix(double ra, double dec, int Nside);

int radecdegtohealpixf(double ra, double dec, int Nside, double* dx, double* dy);

Const int64_t radecdegtohealpixl(double ra, double dec, int Nside);

int64_t radecdegtohealpixlf(double ra, double dec, int Nside, double* dx, double* dy);

/**
   Converts (x,y,z) coordinates on the unit sphere into a healpix index.
 */
Const int xyztohealpix(double x, double y, double z, int Nside);

Const int64_t xyztohealpixl(double x, double y, double z, int Nside);

int xyztohealpixf(double x, double y, double z, int Nside,
                  double* p_dx, double* p_dy);

int64_t xyztohealpixlf(double x, double y, double z, int Nside,
					   double* p_dx, double* p_dy);

/**
   Converts (x,y,z) coordinates (stored in an array) on the unit sphere into
   a healpix index.
*/
int xyzarrtohealpix(const double* xyz, int Nside);

int64_t xyzarrtohealpixl(const double* xyz, int Nside);

int xyzarrtohealpixf(const double* xyz,int Nside, double* p_dx, double* p_dy);

/**
   Converts a healpix index, plus fractional offsets (dx,dy), into (x,y,z)
   coordinates on the unit sphere.  (dx,dy) must be in [0, 1].  (0.5, 0.5)
   is the center of the healpix.  (0,0) is the southernmost corner, (1,1) is
   the northernmost corner, (1,0) is the easternmost, and (0,1) the
   westernmost.
*/
void healpix_to_xyz(int hp, int Nside, double dx, double dy,
                    double* p_x, double *p_y, double *p_z);

/**
   Same as healpix_to_xyz, but (x,y,z) are stored in an array.
*/
void healpix_to_xyzarr(int hp, int Nside, double dx, double dy,
					   double* xyz);

void healpixl_to_xyzarr(int64_t hp, int Nside, double dx, double dy,
                        double* xyz);


/**
   Same as healpix_to_xyz, but returns (RA,DEC) in radians.
*/
void healpix_to_radec(int hp, int Nside, double dx, double dy,
					  double* ra, double* dec);

void healpix_to_radecdeg(int hp, int Nside, double dx, double dy,
                         double* ra, double* dec);

void healpixl_to_radecdeg(int64_t hp, int Nside, double dx, double dy,
						  double* ra, double* dec);

/**
   Same as healpix_to_radec, but (RA,DEC) are stored in an array.
 */
void healpix_to_radecarr(int hp, int Nside, double dx, double dy,
						 double* radec);

void healpix_to_radecdegarr(int hp, int Nside, double dx, double dy,
                            double* radec);

/**
   Computes the approximate side length of a healpix, in arcminutes.
 */
Const double healpix_side_length_arcmin(int Nside);

/**
 Computes the approximate Nside you need to get healpixes with side
 length about "arcmin" arcminutes.  (inverse of
 healpix_side_length_arcmin)
 */
double healpix_nside_for_side_length_arcmin(double arcmin);

/**
   Finds the healpixes neighbouring the given healpix, placing them in the
   array "neighbour".  Returns the number of neighbours.  You must ensure
   that "neighbour" has 8 elements.

   Healpixes in the interior of a large healpix will have eight neighbours;
   pixels near the edges can have fewer.
*/
int healpix_get_neighbours(int hp, int* neighbours, int Nside);

/**
 Same as above, but for Nsides big enough that it overflows 32-bit int.
 */
int healpix_get_neighboursl(int64_t pix, int64_t* neighbour, int Nside);

/**
 Finds the healpixes containing and neighbouring the given xyz
 position which are within distance 'range' (in units of distance of
 the unit sphere).  Places the results in 'healpixes', which must have
 at least 9 elements.  Returns the number of 'healpixes' set.

 Returns -1 if "Nside" < 0.
 */
int healpix_get_neighbours_within_range(double* xyz, double range, int* healpixes,
										int Nside);

/**
 Same as above, but RA,Dec,radius in degrees.
 */
int healpix_get_neighbours_within_range_radec(double ra, double dec, double radius,
											  int* healpixes, int Nside);

/**
 Returns the minimum distance (in degrees) between the given healpix
 and the given RA,Dec (in degrees).
 */
double healpix_distance_to_radec(int hp, int Nside, double ra, double dec,
								 double* closestradec);

/**
 Returns the minimum distance (in degrees) between the given healpix
 and the given xyz (point on unit sphere).
 */
double healpix_distance_to_xyz(int hp, int Nside, const double* xyz,
							   double* closestxyz);

/**
 Returns true if the closest distance between the given healpix and
 the given RA,Dec (in degrees) is less than then given radius (in degrees).
 */
int healpix_within_range_of_radec(int hp, int Nside, double ra, double dec,
								   double radius);
int healpix_within_range_of_xyz(int hp, int Nside, const double* xyz,
								double radius);


/**
 Computes the RA,Dec bounding-box of the given healpix.  Results are
 in degrees.  RA may be wacky for healpixes spanning RA=0.
 */
void healpix_radec_bounds(int hp, int nside,
						  double* ralo, double* rahi,
						  double* declo, double* dechi);

#endif
