/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <arpa/inet.h>
#include <assert.h>

#include "os-features.h"
#include "healpix.h"
#include "healpix-utils.h"
#include "starutil.h"
#include "errors.h"
#include "log.h"
#include "boilerplate.h"
#include "fitsioutils.h"
#include "bl.h"
#include "fitstable.h"
#include "ioutils.h"
#include "mathutil.h"

/**
 Accepts a list of input FITS tables, all with exactly the same
 structure, and including RA,Dec columns.

 Accepts a big-healpix Nside, and a margin in degrees.

 Writes an output file for each of the big-healpixes, containing those
 rows that are within (or within range) of the healpix.
 */

const char* OPTIONS = "hvn:r:d:m:o:gc:e:t:b:RC";

void printHelp(char* progname) {
    BOILERPLATE_HELP_HEADER(stdout);
    printf("\nUsage: %s [options] <input-FITS-catalog> [...]\n"
           "    -o <output-filename-pattern>  with %%i printf-pattern\n"
           "    [-r <ra-column-name>]: name of RA in FITS table (default RA)\n"
           "    [-d <dec-column-name>]: name of DEC in FITS table (default DEC)\n"
           "    [-n <healpix Nside>]: default is 1\n"
           "    [-R]: use ring indexing, rather than xy indexing, for healpixes\n"
           "    [-m <margin in deg>]: add a margin of this many degrees around the healpixes; default 0\n"
           "    [-g]: gzip'd inputs\n"
           "    [-c <name>]: copy given column name to the output files\n"
           "    [-e <name>]: copy given column name to the output files, converting to FITS type E (float)\n"
           "    [-C]: close output files after each input file has been read\n"
           "    [-t <temp-dir>]: use the given temp dir; default is /tmp\n"
           "    [-b <backref-file>]: save the filenumber->filename map in this file; enables writing backreferences too\n"
           "    [-v]: +verbose\n"
           "\n\n\n"
           "WARNING: The input FITS files MUST have EXACTLY the same format!!",
           progname);
}


struct cap_s {
    double xyz[3];
    double r2;
};
typedef struct cap_s cap_t;

static int refill_rowbuffer(void* baton, void* buffer,
                            unsigned int offset, unsigned int nelems) {
    fitstable_t* table = baton;
    //printf("refill_rowbuffer: offset %i, n %i\n", offset, nelems);
    return fitstable_read_nrows_data(table, offset, nelems, buffer);
}


int main(int argc, char *argv[]) {
    int argchar;
    char* progname = argv[0];
    sl* infns = sl_new(16);
    char* outfnpat = NULL;
    char* racol = "RA";
    char* deccol = "DEC";
    char* tempdir = "/tmp";
    anbool gzip = FALSE;
    sl* cols = sl_new(16);
    sl* e_cols = sl_new(16);
    int loglvl = LOG_MSG;
    int nside = 1;
    double margin = 0.0;
    int NHP;
    double md;
    char* backref = NULL;
    anbool ringindex = FALSE;
    anbool closefiles = FALSE;
    off_t* resume_offsets = NULL;
    
    fitstable_t* intable;
    fitstable_t* intable2;
    fitstable_t** outtables;

    anbool anycols = FALSE;

    char** myargs;
    int nmyargs;
    int i;

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
        case 'C':
            closefiles = TRUE;
            break;
        case 'R':
            ringindex = TRUE;
            break;
        case 'b':
            backref = optarg;
            break;
        case 't':
            tempdir = optarg;
            break;
        case 'c':
            sl_append(cols, optarg);
            break;
        case 'e':
            sl_append(e_cols, optarg);
            break;
        case 'g':
            gzip = TRUE;
            break;
        case 'o':
            outfnpat = optarg;
            break;
        case 'r':
            racol = optarg;
            break;
        case 'd':
            deccol = optarg;
            break;
        case 'n':
            nside = atoi(optarg);
            break;
        case 'm':
            margin = atof(optarg);
            break;
        case 'v':
            loglvl++;
            break;
        case '?':
            fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        case 'h':
            printHelp(progname);
            return 0;
        default:
            return -1;
        }

    if (sl_size(cols) == 0) {
        sl_free2(cols);
        cols = NULL;
    }
    if (sl_size(e_cols) == 0) {
        sl_free2(e_cols);
        e_cols = NULL;
    }
    anycols = (cols || e_cols);

    nmyargs = argc - optind;
    myargs = argv + optind;

    for (i=0; i<nmyargs; i++)
        sl_append(infns, myargs[i]);
        
    if (!sl_size(infns)) {
        printHelp(progname);
        printf("Need input filenames!\n");
        exit(-1);
    }
    log_init(loglvl);
    fits_use_error_system();

    NHP = 12 * nside * nside;
    logmsg("%i output healpixes\n", NHP);
    outtables = calloc(NHP, sizeof(fitstable_t*));
    assert(outtables);

    if (closefiles) {
        // In order to reduce the number of open output files, we're
        // going to close output files after we finished reading each
        // input file.  When we close a file, we'll remember where we
        // left off if we need to re-open it in the future.  (We do
        // this rather than just using the file size because FITS
        // files are always padded out to fill an integer number of
        // FITS blocks of 2880 bytes).
        resume_offsets = calloc(NHP, sizeof(off_t));
        assert(resume_offsets);
    }
    
    md = deg2dist(margin);

    /**
     About the mincaps/maxcaps:

     These have a center and radius-squared, describing the region
     inside a small circle on the sphere.

     The "mincaps" describe the regions that are definitely owned by a
     single healpix -- ie, more than MARGIN distance from any edge.
     That is, the mincap is the small circle centered at (0.5, 0.5) in
     the healpix and with radius = the distance to the closest healpix
     boundary, MINUS the margin distance.

     Below, we first check whether a new star is within the "mincap"
     of any healpix.  If so, we stick it in that healpix and continue.

     Otherwise, we check all the "maxcaps" -- these are the healpixes
     it could *possibly* be in.  We then refine with
     healpix_within_range_of_xyz.  The maxcap distance is the distance
     to the furthest boundary point, PLUS the margin distance.
     */


    cap_t* mincaps = malloc(NHP * sizeof(cap_t));
    cap_t* maxcaps = malloc(NHP * sizeof(cap_t));
    for (i=0; i<NHP; i++) {
        // center
        double r2;
        double xyz[3];
        double* cxyz;
        double step = 1e-3;
        double v;
        double r2b, r2a;

        cxyz = mincaps[i].xyz;
        healpix_to_xyzarr(i, nside, 0.5, 0.5, mincaps[i].xyz);
        memcpy(maxcaps[i].xyz, cxyz, 3 * sizeof(double));
        logverb("Center of HP %i: (%.3f, %.3f, %.3f)\n", i, cxyz[0], cxyz[1], cxyz[2]);

        // radius-squared:
        // max is the easy one: max of the four corners (I assume)
        r2 = 0.0;
        healpix_to_xyzarr(i, nside, 0.0, 0.0, xyz);
        logverb("  HP %i corner 1: (%.3f, %.3f, %.3f), distsq %.3f\n", i, xyz[0], xyz[1], xyz[2], distsq(xyz, cxyz, 3));
        r2 = MAX(r2, distsq(xyz, cxyz, 3));
        healpix_to_xyzarr(i, nside, 1.0, 0.0, xyz);
        logverb("  HP %i corner 1: (%.3f, %.3f, %.3f), distsq %.3f\n", i, xyz[0], xyz[1], xyz[2], distsq(xyz, cxyz, 3));
        r2 = MAX(r2, distsq(xyz, cxyz, 3));
        healpix_to_xyzarr(i, nside, 0.0, 1.0, xyz);
        logverb("  HP %i corner 1: (%.3f, %.3f, %.3f), distsq %.3f\n", i, xyz[0], xyz[1], xyz[2], distsq(xyz, cxyz, 3));
        r2 = MAX(r2, distsq(xyz, cxyz, 3));
        healpix_to_xyzarr(i, nside, 1.0, 1.0, xyz);
        logverb("  HP %i corner 1: (%.3f, %.3f, %.3f), distsq %.3f\n", i, xyz[0], xyz[1], xyz[2], distsq(xyz, cxyz, 3));
        r2 = MAX(r2, distsq(xyz, cxyz, 3));
        logverb("  max distsq: %.3f\n", r2);
        logverb("  margin dist: %.3f\n", md);
        maxcaps[i].r2 = square(sqrt(r2) + md);
        logverb("  max cap distsq: %.3f\n", maxcaps[i].r2);
        r2a = r2;

        r2 = 1.0;
        r2b = 0.0;
        for (v=0; v<=1.0; v+=step) {
            healpix_to_xyzarr(i, nside, 0.0, v, xyz);
            r2 = MIN(r2, distsq(xyz, cxyz, 3));
            r2b = MAX(r2b, distsq(xyz, cxyz, 3));
            healpix_to_xyzarr(i, nside, 1.0, v, xyz);
            r2 = MIN(r2, distsq(xyz, cxyz, 3));
            r2b = MAX(r2b, distsq(xyz, cxyz, 3));
            healpix_to_xyzarr(i, nside, v, 0.0, xyz);
            r2 = MIN(r2, distsq(xyz, cxyz, 3));
            r2b = MAX(r2b, distsq(xyz, cxyz, 3));
            healpix_to_xyzarr(i, nside, v, 1.0, xyz);
            r2 = MIN(r2, distsq(xyz, cxyz, 3));
            r2b = MAX(r2b, distsq(xyz, cxyz, 3));
        }
        mincaps[i].r2 = square(MAX(0, sqrt(r2) - md));
        logverb("\nhealpix %i: min rad    %g\n", i, sqrt(r2));
        logverb("healpix %i: max rad    %g\n", i, sqrt(r2a));
        logverb("healpix %i: max rad(b) %g\n", i, sqrt(r2b));
        assert(r2a >= r2b);
    }

    if (backref) {
        fitstable_t* tab = fitstable_open_for_writing(backref);
        int maxlen = 0;
        char* buf;
        for (i=0; i<sl_size(infns); i++) {
            char* infn = sl_get(infns, i);
            maxlen = MAX(maxlen, strlen(infn));
        }
        fitstable_add_write_column_array(tab, fitscolumn_char_type(), maxlen,
                                         "filename", NULL);
        fitstable_add_write_column(tab, fitscolumn_i16_type(), "index", NULL);
        if (fitstable_write_primary_header(tab) ||
            fitstable_write_header(tab)) {
            ERROR("Failed to write header of backref table \"%s\"", backref);
            exit(-1);
        }
        buf = malloc(maxlen+1);
        assert(buf);

        for (i=0; i<sl_size(infns); i++) {
            char* infn = sl_get(infns, i);
            int16_t ind;
            memset(buf, 0, maxlen);
            strcpy(buf, infn);
            ind = i;
            if (fitstable_write_row(tab, buf, &ind)) {
                ERROR("Failed to write row %i of backref table: %s = %i",
                      i, buf, ind);
                exit(-1);
            }
        }
        if (fitstable_fix_header(tab) ||
            fitstable_close(tab)) {
            ERROR("Failed to fix header & close backref table");
            exit(-1);
        }
        logmsg("Wrote backref table %s\n", backref);
        free(buf);
    }

    for (i=0; i<sl_size(infns); i++) {
        char* infn = sl_get(infns, i);
        char* originfn = infn;
        int r, NR;
        tfits_type any, dubl;
        il* hps = NULL;
        bread_t* rowbuf;
        int R;
        char* tempfn = NULL;
        char* padrowdata = NULL;
        int ii;

        logmsg("Reading input \"%s\"...\n", infn);

        if (gzip) {
            char* cmd;
            int rtn;
            tempfn = create_temp_file("hpsplit", tempdir);
            asprintf_safe(&cmd, "gunzip -cd %s > %s", infn, tempfn);
            logmsg("Running: \"%s\"\n", cmd);
            rtn = run_command_get_outputs(cmd, NULL, NULL);
            if (rtn) {
                ERROR("Failed to run command: \"%s\"", cmd);
                exit(-1);
            }
            free(cmd);
            infn = tempfn;
        }

        intable = fitstable_open(infn);
        if (!intable) {
            ERROR("Couldn't read catalog %s", infn);
            exit(-1);
        }
        NR = fitstable_nrows(intable);
        logmsg("Got %i rows\n", NR);

        // For '-e', we need to endian-flip the input rows, which requires
        // knowing the columns... we use 'intable2' just for that.
        intable2 = fitstable_open(infn);
        if (!intable2) {
            ERROR("Couldn't read catalog %s", infn);
            exit(-1);
        }
        fitstable_add_fits_columns_as_struct(intable2);

        any = fitscolumn_any_type();
        dubl = fitscolumn_double_type();

        fitstable_add_read_column_struct(intable, dubl, 1, 0, any, racol, TRUE);
        fitstable_add_read_column_struct(intable, dubl, 1, sizeof(double), any, deccol, TRUE);

        fitstable_use_buffered_reading(intable, 2*sizeof(double), 1000);

        R = fitstable_row_size(intable);
        rowbuf = buffered_read_new(R, 1000, NR, refill_rowbuffer, intable);

        if (fitstable_read_extension(intable, 1)) {
            ERROR("Failed to find RA and DEC columns (called \"%s\" and \"%s\" in the FITS file)", racol, deccol);
            exit(-1);
        }

        for (r=0; r<NR; r++) {
            int hp = -1;
            double ra, dec;
            int j;
            double* rd;
            void* rowdata;
            void* rdata;
            anbool flipped;

            if (r && ((r % 100000) == 0)) {
                logmsg("Reading row %i of %i\n", r, NR);
            }

            //printf("reading RA,Dec for row %i\n", r);
            rd = fitstable_next_struct(intable);
            ra = rd[0];
            dec = rd[1];

            logverb("row %i: ra,dec %g,%g\n", r, ra, dec);
            if (margin == 0) {
                hp = radecdegtohealpix(ra, dec, nside);
                logverb("  --> healpix %i\n", hp);
            } else {

                double xyz[3];
                anbool gotit = FALSE;
                double d2;
                if (!hps)
                    hps = il_new(4);
                radecdeg2xyzarr(ra, dec, xyz);
                for (j=0; j<NHP; j++) {
                    d2 = distsq(xyz, mincaps[j].xyz, 3);
                    if (d2 <= mincaps[j].r2) {
                        logverb("  -> in mincap %i  (dist %g vs %g)\n", j, sqrt(d2), sqrt(mincaps[j].r2));
                        il_append(hps, j);
                        gotit = TRUE;
                        break;
                    }
                }
                if (!gotit) {
                    for (j=0; j<NHP; j++) {
                        d2 = distsq(xyz, maxcaps[j].xyz, 3);
                        if (d2 <= maxcaps[j].r2) {
                            logverb("  -> in maxcap %i  (dist %g vs %g)\n", j, sqrt(d2), sqrt(maxcaps[j].r2));
                            if (healpix_within_range_of_xyz(j, nside, xyz, margin)) {
                                logverb("  -> and within range.\n");
                                il_append(hps, j);
                            }
                        }
                    }
                }

                //hps = healpix_rangesearch_radec(ra, dec, margin, nside, hps);

                logverb("  --> healpixes: [");
                for (j=0; j<il_size(hps); j++)
                    logverb(" %i", il_get(hps, j));
                logverb(" ]\n");
            }

            //printf("Reading rowdata for row %i\n", r);
            rowdata = buffered_read(rowbuf);
            assert(rowdata);

            flipped = FALSE;
            j=0;
            while (1) {
                if (hps) {
                    if (j >= il_size(hps))
                        break;
                    hp = il_get(hps, j);
                    j++;
                }
                assert(hp < NHP);
                assert(hp >= 0);

                // Open output file if necessary
                if (!outtables[hp]) {
                    char* outfn;
                    fitstable_t* out;
                    
                    // MEMLEAK the output filename.  You'll live.
                    if (ringindex) {
                        int ringhp = healpix_xy_to_ring(hp, nside);
                        logverb("Ring-indexed healpix: %i (xy index: %i)\n", ringhp,hp);
                        asprintf_safe(&outfn, outfnpat, ringhp);
                    } else {
                        asprintf_safe(&outfn, outfnpat, hp);
                    }

                    logmsg("Opening output file \"%s\"...\n", outfn);
                    out = fitstable_open_for_writing(outfn);
                    if (!out) {
                        ERROR("Failed to open output table \"%s\"", outfn);
                        exit(-1);
                    }
                    // Set the output table structure.
                    if (anycols) {
                        if (cols)
                            fitstable_add_fits_columns_as_struct3(intable, out, cols, 0);
                        if (e_cols)
                            fitstable_add_fits_columns_as_struct4(intable, out, e_cols, 0, TFITS_BIN_TYPE_E);

                    } else
                        fitstable_add_fits_columns_as_struct2(intable, out);

                    if (backref) {
                        tfits_type i16type;
                        tfits_type i32type;
                        // R = fitstable_row_size(intable);
                        int off = R;
                        i16type = fitscolumn_i16_type();
                        i32type = fitscolumn_i32_type();
                        fitstable_add_read_column_struct(out, i16type, 1, off,
                                                         i16type, "backref_file", TRUE);
                        off += sizeof(int16_t);
                        fitstable_add_read_column_struct(out, i32type, 1, off,
                                                         i32type, "backref_index", TRUE);
                    }

                    if (fitstable_write_primary_header(out) ||
                        fitstable_write_header(out)) {
                        ERROR("Failed to write output file headers for \"%s\"", outfn);
                        exit(-1);
                    }

                    outtables[hp] = out;
                }

                if (backref) {
                    int16_t brfile;
                    int32_t brind;
                    if (!padrowdata) {
                        padrowdata = malloc(R + sizeof(int16_t) + sizeof(int32_t));
                        assert(padrowdata);
                    }
                    // convert to FITS endian
                    brfile = htons(i);
                    brind  = htonl(r);
                    // add backref data to rowdata
                    memcpy(padrowdata, rowdata, R);
                    memcpy(padrowdata + R, &brfile, sizeof(int16_t));
                    memcpy(padrowdata + R + sizeof(int16_t), &brind, sizeof(int32_t));
                    rdata = padrowdata;
                } else {
                    rdata = rowdata;
                }

                if (closefiles && (outtables[hp]->fid == NULL)) {
                    char* outfn = outtables[hp]->fn;
                    logverb("Re-opening healpix %i file %s at offset %lu\n",
                            hp, outfn, (long)resume_offsets[hp]);
                    outtables[hp]->fid = fopen(outfn, "r+b");
                    fseeko(outtables[hp]->fid, resume_offsets[hp], SEEK_SET);
                }

                if (anycols) {
                    if (!flipped) {
                        // if we're writing to multiple output
                        // healpixes, only flip once!
                        flipped = TRUE;
                        fitstable_endian_flip_row_data(intable2, rdata);
                    }
                    if (fitstable_write_struct(outtables[hp], rdata)) {
                        ERROR("Failed to copy a row of data from input table \"%s\" to output healpix %i", infn, hp);
                    }
                    
                } else {
                    if (fitstable_write_row_data(outtables[hp], rdata)) {
                        ERROR("Failed to copy a row of data from input table \"%s\" to output healpix %i", infn, hp);
                    }
                }

                if (!hps)
                    break;
            }
            if (hps)
                il_remove_all(hps);

        }
        buffered_read_free(rowbuf);
        // wack... buffered_read_free() just frees its internal buffer,
        // not the "rowbuf" struct itself.
        // who wrote this crazy code?  Oh, me of 5 years ago.  Jerk.
        free(rowbuf);

        fitstable_close(intable);
        il_free(hps);

        if (tempfn) {
            logverb("Removing temp file %s\n", tempfn);
            if (unlink(tempfn)) {
                SYSERROR("Failed to unlink() temp file \"%s\"", tempfn);
            }
            tempfn = NULL;
        }

        // fix headers so that the files are valid at this point.
        for (ii=0; ii<NHP; ii++) {
            if (!outtables[ii])
                continue;
            if (closefiles && (outtables[ii]->fid == NULL))
                continue;

            off_t offset = ftello(outtables[ii]->fid);
            if (closefiles) {
                resume_offsets[ii] = offset;
                logverb("Closing healpix %i (saving offset %lu)\n", ii, (long)offset);
                if (fitstable_fix_header(outtables[ii])) {
                    ERROR("Failed to fix header for healpix %i after reading input file \"%s\"", ii, originfn);
                    exit(-1);
                }
                if (fclose(outtables[ii]->fid)) {
                    SYSERROR("Failed to close file %s\n", outtables[ii]->fn);
                    exit(-1);
                }
                outtables[ii]->fid = NULL;
            } else {
                // the "fitstable_fix_header" call (via
                // fitsfile_fix_header) adds padding to the file to
                // bring it up to a FITS block size, so we ftell and
                // fseek afterward.
                if (fitstable_fix_header(outtables[ii])) {
                    ERROR("Failed to fix header for healpix %i after reading input file \"%s\"", ii, originfn);
                    exit(-1);
                }
                fseeko(outtables[ii]->fid, offset, SEEK_SET);
            }
        }

        if (padrowdata) {
            free(padrowdata);
            padrowdata = NULL;
        }

    }

    for (i=0; i<NHP; i++) {
        if (!outtables[i])
            continue;
        if (closefiles && (outtables[i]->fid == NULL)) {
            if (fitstable_close(outtables[i])) {
                ERROR("Failed to close output table for healpix %i", i);
                exit(-1);
            }
            continue;
        }
        if (fitstable_fix_header(outtables[i]) ||
            fitstable_fix_primary_header(outtables[i]) ||
            fitstable_close(outtables[i])) {
            ERROR("Failed to close output table for healpix %i", i);
            exit(-1);
        }
    }

    free(outtables);
    sl_free2(infns);
    sl_free2(cols);
    sl_free2(e_cols);

    free(mincaps);
    free(maxcaps);

    free(resume_offsets);
    
    return 0;
}



