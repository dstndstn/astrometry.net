/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

/*-------------------------------------------------------------
 * This file is used to make a table with one column containing
 * whether the star was flagged as a diffraction spike by our
 * algorithm (ie. erroneous).
 * ------------------------------------------------------------
 */

#include <stdio.h>
#include "fitsioutils.h"

int create_table(FILE* inFid, FILE* outFid){
    int c, nEntries = 0;
    int x = 0;
    qfits_table* tbl;
    qfits_header* tbl_hdr;

    // create table header
    qfits_header* hdr = qfits_table_prim_header_default();
    qfits_header_dump(hdr, outFid);

    // create table
    tbl = qfits_table_new("spikes",QFITS_BINTABLE,1,1,0);
    // add column
    fits_add_column(tbl, 0,TFITS_BIN_TYPE_X, 1, " ", "diffraction spike"); 
	
    tbl_hdr = qfits_table_ext_header_default(tbl); 
    qfits_header_dump(tbl_hdr, outFid);

    // fill in content of table
    while((c = fgetc(inFid)) != EOF){
        // show progress
        if (x==100000){
            printf(".");
            x=0;
        }else{
            x++;
        }
        //printf("%c", c);
        fits_write_data_X(outFid,((unsigned char)c));
        nEntries++;
    }
    fits_pad_file(outFid);
    // fix the headers
    tbl->nr = nEntries;
    tbl_hdr = qfits_table_ext_header_default(tbl);
    fix_headers(outFid, hdr, tbl_hdr);
	
    qfits_header_destroy(hdr);
    qfits_header_destroy(tbl_hdr);
    qfits_table_close(tbl);
    return 0;
}

int fix_headers(FILE* fId, qfits_header* hdr, qfits_header* tbl_hdr){
    char val[32];
    off_t offset;
    offset = ftello(fId);
    fseeko(fId, 0, SEEK_SET);

    // write header
    qfits_header_dump(hdr, fId);

    // write table header again
    qfits_header_dump(tbl_hdr, fId);

    // point to the end of the file again
    fseek(fId, offset, SEEK_SET);
    return 0;
}

int main(int argc, char* argv[]){
    if (argc==3){
        FILE* inFid,outFid;
	   
        printf("processing %s ", argv[1]);

        // open files for reading/writing
        inFid = fopen(argv[1], "r");
        outFid = fopen(argv[2], "w");

        // create table and write to output file
        create_table(inFid, outFid);

        fclose(inFid);
        fclose(outFid);
        printf(" finished \n");
    } else {
        printf("Usage: spike_join inputFile outputFile\n");
    }
    return 1;
}
