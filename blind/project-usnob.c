/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

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

#include <stdio.h>

#include "math.h"
#include "usnob-fits.h"
#include "usnob.h"
#include "starutil.h"
#include "healpix.h"
#include "boilerplate.h"

#define OPTIONS "h"

void print_help(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage:\n"
		   "  %s <usnob-file.fits> <output-dir> [FLAT]\n"
		   , progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** args) {
    int c;
	char* infn;
	char* filepath;
	char galaxyfilename[255];
	char pointfilename[255];
	char bandfilename[255];
	char tilefilename[255];
	char motionfilename[255];
	usnob_fits* usnob;
	int i, j, N;
	int hp, Nside;
	double center[3];
	FILE *galaxy_file;
	FILE *point_file;
	FILE *band_file;
	FILE *tile_file;
	FILE *motion_file;
	int galaxyBuffer[5];
	double pointbuffer[3];
	double bandbuffer[5];
	unsigned int tilebuffer[5];
	double xyz[3];
	int motionbuffer[2];
	int numMags;
	usnob_entry* star;
	int doProject;

	while ((c = getopt(argc, args, OPTIONS)) != -1) {
		switch (c) {
		case '?':
		case 'h':
			print_help(args[0]);
			exit(0);
		}
	}

	// make sure there is one non-option argument: the usnob fits filename.
	if (argc < 3) {
		print_help(args[0]);
		exit(-1);
	}

	// try to open the file.
	infn = args[1];
	fprintf(stderr, "Reading USNOB catalog file %s\n", infn);
	usnob = usnob_fits_open(infn);
	if (!usnob) {
		fprintf(stderr, "Failed to read USNOB catalog from file %s.\n", infn);
		exit(-1);
	}

	// find out which healpix the file covers.
	hp = qfits_header_getint(usnob->header, "HEALPIX", -1);
	Nside = qfits_header_getint(usnob->header, "NSIDE", -1);
	if ((hp == -1) || (Nside == -1)) {
		fprintf(stderr, "Failed to find \"HEALPIX\" and \"NSIDE\" headers in file %s.\n", infn);
		exit(-1);
	}

	// pull out the center of the healpix.
	healpix_to_xyzarr(hp, Nside, 0.5, 0.5, center);

	// for each star...
	N = usnob_fits_count_entries(usnob);
	fprintf(stderr, "File contains %i stars.\n", N);

	filepath = args[2];

	strcpy(galaxyfilename, filepath);
	strcpy(pointfilename, filepath);
	strcpy(bandfilename, filepath);
	strcpy(tilefilename, filepath);
	strcpy(motionfilename, filepath);

	strcat(galaxyfilename, "_galaxy.bin");
	strcat(pointfilename, "_points.bin");
	strcat(bandfilename, "_bands.bin");
	strcat(tilefilename, "_tiles.bin");
	strcat(motionfilename, "_motion.bin");

	fprintf(stderr, "Writing point info to %s\n", pointfilename);
	fprintf(stderr, "Writing band info to %s\n", bandfilename);
	fprintf(stderr, "Writing tile info to %s\n", tilefilename);
	fprintf(stderr, "Writing galaxy info to %s\n", galaxyfilename);
	fprintf(stderr, "Writing proper motions to %s\n", motionfilename);

	galaxy_file = fopen(galaxyfilename, "wb");
	point_file = fopen(pointfilename, "wb");
	band_file = fopen(bandfilename, "wb");
	tile_file = fopen(tilefilename, "wb");
	motion_file = fopen(motionfilename, "wb");

	doProject = 1;
	if(argc >= 4){
	  if(strcmp(args[3], "FLAT") == 0){
	    doProject = 0;
	  }
	}

	if(doProject == 1){
	  fprintf(stderr, "Projecting coords to plane\n");
	}
	else{
	  fprintf(stderr, "Not projecting coords to plane, output = (RA,Dec) in radians\n");
	}

	for (i=0; i<N; i++) {
	  numMags = 0;
	  pointbuffer[2] = 0.0;
	  // grab the star...
	  star = usnob_fits_read_entry(usnob);

	  if(doProject){
	    // find its xyz position
	    radec2xyzarr(deg2rad(star->ra), deg2rad(star->dec), xyz);
	    // project it around the center
	    star_coords(xyz, center, &pointbuffer[0], &pointbuffer[1]);
	  }
	  else{
	    pointbuffer[0] = deg2rad(star->ra);
	    pointbuffer[1] = deg2rad(star->dec);
	  }

	  for (j=0; j<5; j++){
	    if(usnob_is_observation_valid(&star->obs[j])){
	      galaxyBuffer[j] = star->obs[j].star_galaxy;
	    }
	    else{
	      galaxyBuffer[j] = -1;
	    }

	    if(usnob_is_observation_valid(&star->obs[j]) || !usnob_is_usnob_star(star)){
	      numMags++;
	      pointbuffer[2] += star->obs[j].mag;
	      tilebuffer[j] = star->obs[j].field;
	    }
	    else{
	      tilebuffer[j] = -1;;
	    }
	    
	    bandbuffer[j] = star->obs[j].mag;
	  }
	  if (numMags > 0){
	    pointbuffer[2] = pointbuffer[2] / (double)numMags;
	  }

	  motionbuffer[0] = round(1000*star->mu_ra);
	  motionbuffer[1] = round(1000*star->mu_dec);
	  
	  fwrite(motionbuffer, sizeof(int), 2, motion_file);
	  fwrite(galaxyBuffer, sizeof(int),5,galaxy_file);
	  fwrite(pointbuffer, sizeof(double), 3, point_file);
	  fwrite(bandbuffer, sizeof(double), 5, band_file);
	  fwrite(tilebuffer, sizeof(int), 5, tile_file);
	}
	
	fclose(point_file);
	fclose(band_file);
	fclose(tile_file);
	fclose(galaxy_file);

	usnob_fits_close(usnob);

	return 0;
}


