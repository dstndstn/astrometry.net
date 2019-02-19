/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <stdio.h>
#include <sys/types.h>
#include <sys/mman.h>

#include "usnob-fits.h"
#include "usnob.h"
#include "starutil.h"
#include "healpix.h"
#include "boilerplate.h"
#include "assert.h"
#include "an-endian.h"

#define OPTIONS "h"

void print_help(char* progname) {
    BOILERPLATE_HELP_HEADER(stdout);
    printf("\nUsage:\n"
           "  %s\n"
           , progname);
    //-H <healpix> -N <nside> 
}



int main(int argc, char** args) {
    int c;
    int i;
    int fnum = 0, fnum1 = 0, d = 10;


    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
        case '?':
        case 'h':
            print_help(args[0]);
            exit(0);
        }
    }


    assert(d<180);
    // checks the first d degrees of the sky (must be less than 180)
    while (fnum1<d){
        FILE *fid;
        unsigned char* map; 
        size_t map_size;
        char fname[40]; 
        char *fn = "/w/284/stars284/USNOB10/%03d/b%03d%d.cat";
        sprintf(fname, fn, fnum1, fnum1,fnum);
        printf("%s\n", fname);

        // try to open the file.
        fid = fopen(fname, "r"); 
        if (fid == NULL){
            printf("crap\n");
            return 1;
        }
	
		
        //move to end of file
        if (fseeko(fid, 0, SEEK_END)) { 
            printf("Couldn't seek to end of input file: \n"); 
            exit(-1); 
        }
        //get file size
        map_size = ftello(fid);
        //move back to beginning of file
        fseeko(fid, 0, SEEK_SET);  
        //read map into memory 
        map = mmap(NULL, map_size, PROT_READ, MAP_SHARED, fileno(fid), 0);
				
   	     
        printf("mapsize: %d", map_size/USNOB_RECORD_SIZE);	
	
        //for each entry, prase entry, alert when diffraction_spike flag is set
        for (i=0; i<map_size; i+=USNOB_RECORD_SIZE) { 
            usnob_entry entry; 
            if (i && (i % 10000000 * USNOB_RECORD_SIZE == 0)) {  
                printf("o"); 
                fflush(stdout);  
            } 
            if (usnob_parse_entry(map + i, &entry)) { 
                printf("Failed to parse USNOB entry: offset %i.\n", i); 
                exit(-1);            	
            }else{
                //printf(".");
            }
            if (entry.diffraction_spike){
                uint ival;
                printf("\ndiffraction spike, entry %d\n", i/USNOB_RECORD_SIZE);
                // print bytes 12-15 in base 10 of USNOB entry prior to spike
                ival = u32_letoh(*((uint*)(map+i+ 12 - USNOB_RECORD_SIZE)));
                printf("entry prior spike: %010d\n", ival);
                // print bytes 12-15 in base 10 of USNOB entry with spike
                ival = u32_letoh(*((uint*)(map+i+ 12)));
                printf("diffraction spike: %010d\n", ival);
                // print bytes 12-15 in base 10 of USNOB entry after spike
                ival = u32_letoh(*((uint*)(map+i+ 12 + USNOB_RECORD_SIZE)));
                printf("entry after spike: %010d\n", ival);
            }

		
        }
        printf("\n");
        fclose(fid);

        // increment filenumber (degree counter)
        if (fnum%10 == 9){
            fnum1++;
            fnum = 0;
        } else {	
            fnum++;
        }
    }
    return 0;
}


