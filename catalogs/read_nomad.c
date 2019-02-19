/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

/*
 Reads raw NOMAD data files and prints them out in text format to allow
 verification of our FITS versions.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <endian.h>
#include <netinet/in.h>
#include <byteswap.h>

#include "nomad.h"

int main(int argc, char** args) {
    int j;

    printf("ra dec sigma_racosdec sigma_dec mu_racosdec mu_dec "
           "sigma_mu_racosdec sigma_mu_dec epoch_ra epoch_dec "
           "mag_B mag_V mag_R mag_J mag_H mag_K usnob_id twomass_id "
           "yb6_id ucac2_id tycho2_id astrometry_src blue_src visual_src "
           "red_src usnob_fail twomass_fail tycho_astrometry "
           "alt_radec alt_2mass alt_ucac alt_tycho blue_o red_e "
           "twomass_only hipp_astrometry diffraction confusion "
           "bright_confusion bright_artifact standard external\n");

    for (j=1; j<argc; j++) {
        char* infn;
        FILE* fid;
        unsigned char* map;
        size_t map_size;
        int i;

        infn = args[j];
        fprintf(stderr, "Reading file %s...\n", infn);
        fid = fopen(infn, "rb");
        if (!fid) {
            fprintf(stderr, "Couldn't open input file %s: %s\n", infn, strerror(errno));
            exit(-1);
        }
        if (fseeko(fid, 0, SEEK_END)) {
            fprintf(stderr, "Couldn't seek to end of input file %s: %s\n", infn, strerror(errno));
            exit(-1);
        }
        map_size = ftello(fid);
        fseeko(fid, 0, SEEK_SET);
        map = mmap(NULL, map_size, PROT_READ, MAP_SHARED, fileno(fid), 0);
        if (map == MAP_FAILED) {
            fprintf(stderr, "Couldn't mmap input file %s: %s\n", infn, strerror(errno));
            exit(-1);
        }
        fclose(fid);
        if (map_size % NOMAD_RECORD_SIZE) {
            fprintf(stderr, "Warning, input file %s has size %u which is not divisible into %i-byte records.\n",
                    infn, (unsigned int)map_size, NOMAD_RECORD_SIZE);
        }

        for (i=0; i<map_size; i+=NOMAD_RECORD_SIZE) {
            nomad_entry e;
            if (nomad_parse_entry(&e, map + i)) {
                fprintf(stderr, "Failed to parse NOMAD entry: offset %i in file %s.\n",
                        i, infn);
                exit(-1);
            }

            printf("%g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g ",
                   e.ra, e.dec, e.sigma_racosdec, e.sigma_dec,
                   e.mu_racosdec, e.mu_dec, e.sigma_mu_racosdec,
                   e.sigma_mu_dec, e.epoch_ra, e.epoch_dec, e.mag_B,
                   e.mag_V, e.mag_R, e.mag_J, e.mag_H, e.mag_K);
            printf("%u %u %u %u %u %i %i %i %i ",
                   e.usnob_id, e.twomass_id, e.yb6_id, e.ucac2_id,
                   e.tycho2_id, (int)e.astrometry_src, (int)e.blue_src,
                   (int)e.visual_src, (int)e.red_src);
            printf("%i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i\n",
                   (int)e.usnob_fail, (int)e.twomass_fail,
                   (int)e.tycho_astrometry, (int)e.alt_radec,
                   (int)e.alt_2mass, (int)e.alt_ucac, (int)e.alt_tycho,
                   (int)e.blue_o, (int)e.red_e, (int)e.twomass_only,
                   (int)e.hipp_astrometry, (int)e.diffraction,
                   (int)e.confusion, (int)e.bright_confusion,
                   (int)e.bright_artifact, (int)e.standard,
                   (int)e.external);
        }

        munmap(map, map_size);
    }

    return 0;
}
