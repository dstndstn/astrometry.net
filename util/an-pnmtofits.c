/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdio.h>

#include "an-bool.h"
#include "os-features-config.h" // for HAVE_NETPBM.

#if HAVE_NETPBM
#include <netpbm/pam.h>
#else
#include <arpa/inet.h>
#endif

#include "log.h"
#include "errors.h"
#include "fitsioutils.h"
#include "ioutils.h"
#include "bl.h"

static const char* OPTIONS = "hvqo:";

static void printHelp(char* progname) {
    printf("%s    [options]  [<input-file>, default stdin]\n"
           "      or         [<input-file> <output-file>]\n"
           "      [-o <output-file>]       (default stdout)\n"
           "      [-v]: verbose\n"
           "      [-q]: quiet\n"
           "\n", progname);
}

#if HAVE_NETPBM
#else

static int skip_whitespace(FILE* fid, int nmax) {
    int c;
    int i;
    for (i=0; (nmax == 0) || i<nmax; i++) {
        c = getc(fid);
        if (c == EOF) {
            if (feof(fid)) {
                SYSERROR("Failed reading whitespace in PNM header: end-of-file");
            } else if (ferror(fid )) {
                SYSERROR("Failed reading whitespace in PNM header");
            } else {
                SYSERROR("Failed reading whitespace in PNM header");
            }
            return -1;
        }
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t')
            continue;
        // finished... push back.
        ungetc(c, fid);
        return 0;
    }
    return 0;
}

static int parse_pnm_header(FILE* fid, int* W, int* H, int* depth, int* maxval) {
    // P6 == ppm
    unsigned char p = '\0';
    unsigned char n = '\0';
    if (read_u8(fid, &p) ||
        read_u8(fid, &n)) {
        ERROR("Failed to read P* from PNM header");
        return -1;
    }
    if (p != 'P') {
        ERROR("File doesn't start with 'P': not pnm.");
        return -1;
    }
    if (n == '6') {
        // PPM
        *depth = 3;
    } else if (n == '5') {
        // PGM
        *depth = 1;
    } else {
        ERROR("File starts with code \"%c%c\": not understood as pnm.", p,n);
        return -1;
    }
    if (skip_whitespace(fid, 0))
        return -1;
    if (fscanf(fid, "%d", W) != 1) {
        ERROR("Failed to parse width from PNM header");
        return -1;
    }
    if (skip_whitespace(fid, 0))
        return -1;
    if (fscanf(fid, "%d", H) != 1) {
        ERROR("Failed to parse height from PNM header");
        return -1;
    }
    if (skip_whitespace(fid, 0))
        return -1;
    if (fscanf(fid, "%d", maxval) != 1) {
        ERROR("Failed to parse maxval from PNM header");
        return -1;
    }
    if (skip_whitespace(fid, 1))
        return -1;
    return 0;
}

static int maxval_to_bytes(int maxval) {
    if (maxval <= 255)
        return 1;
    return 2;
}

static int read_pnm_row(FILE* fid, int W, int depth, int maxval, void* buffer) {
    int bps = maxval_to_bytes(maxval);
    size_t n = (size_t)W * (size_t)depth;
    if (fread(buffer, bps, n, fid) != n) {
        SYSERROR("Failed to read PNM row");
        return -1;
    }
    if (bps == 2) {
        // big-endian
        uint16_t* u = buffer;
        int i;
        for (i=0; i<W*depth; i++)
            u[i] = ntohs(u[i]);
    }
    return 0;
}

#endif

	


int main(int argc, char** args) {
    int argchar;
    char* infn = NULL;
    char* outfn = NULL;
    unsigned int row;
    int bits;
    FILE* fid = stdin;
    FILE* fout = stdout;
    int loglvl = LOG_MSG;
    char* progname = args[0];
    int bzero = 0;
    int outformat;
    qfits_header* hdr;
    unsigned int plane;
    off_t datastart;
    anbool onepass = FALSE;
    bl* pixcache = NULL;

#if HAVE_NETPBM
    struct pam img;
    tuple * tuplerow;
#else
    void* rowbuf;
#endif
    int W, H, depth, maxval;

    while ((argchar = getopt (argc, args, OPTIONS)) != -1) {
        switch (argchar) {
        case '?':
        case 'h':
            printHelp(progname);
            exit(0);
        case 'v':
            loglvl++;
            break;
        case 'q':
            loglvl--;
            break;
        case 'o':
            outfn = optarg;
            break;
        }
    }
    log_init(loglvl);
    log_to(stderr);
    fits_use_error_system();

    if (optind == argc) {
        // ok, stdin to stdout.
    } else if (optind == argc-1) {
        infn = args[optind];
    } else if (optind == argc-2) {
        infn = args[optind];
        outfn = args[optind+1];
    } else {
        printHelp(progname);
        exit(-1);
    }

    if (infn && !streq(infn, "-")) {
        fid = fopen(infn, "rb");
        if (!fid) {
            SYSERROR("Failed to open input file %s", infn);
            exit(-1);
        }
    }
    if (outfn) {
        fout = fopen(outfn, "wb");
        if (!fid) {
            SYSERROR("Failed to open output file %s", outfn);
            exit(-1);
        }
    } else
        outfn = "stdout";

#if HAVE_NETPBM
    pm_init(args[0], 0);
    pnm_readpaminit(fid, &img, 
                    // PAM_STRUCT_SIZE isn't defined until Netpbm 10.23 (July 2004)
#if defined(PAM_STRUCT_SIZE)
                    PAM_STRUCT_SIZE(tuple_type)
#else
                    sizeof(struct pam)
#endif
                    );
    W = img.width;
    H = img.height;
    depth = img.depth;
    maxval = img.maxval;

    tuplerow = pnm_allocpamrow(&img);
    bits = pm_maxvaltobits(img.maxval); 
    bits = (bits <= 8) ? 8 : 16;

#else // No NETPBM

    if (parse_pnm_header(fid, &W, &H, &depth, &maxval)) {
        ERROR("Failed to parse PNM header from file: %s\n", infn ? infn : "<stdin>");
        exit(-1);
    }
    bits = 8 * maxval_to_bytes(maxval);

    rowbuf = malloc((size_t)W * (size_t)depth * (size_t)(bits/8));

#endif

    logmsg("Read file %s: %i x %i pixels x %i color(s); maxval %i\n",
           infn ? infn : "stdin", W, H, depth, maxval);
    if (bits == 8)
        outformat = BPP_8_UNSIGNED;
    else {
        outformat = BPP_16_SIGNED;
        if (maxval >= INT16_MAX)
            bzero = 0x8000;
    }
    logmsg("Using %i-bit output\n", bits);

    hdr = fits_get_header_for_image3(W, H, outformat, depth, NULL);
    if (depth == 3)
        qfits_header_add(hdr, "CTYPE3", "RGB", "Tell Aladin this is RGB", NULL);
    if (bzero)
        fits_header_add_int(hdr, "BZERO", bzero, "Number that has been subtracted from pixel values");
    if (qfits_header_dump(hdr, fout)) {
        ERROR("Failed to write FITS header to file %s", outfn);
        exit(-1);
    }
    qfits_header_destroy(hdr);

    datastart = ftello(fid);
    // Figure out if we can seek backward in this input file...
    if ((fid == stdin) ||
        (fseeko(fid, 0, SEEK_SET) ||
         fseeko(fid, datastart, SEEK_SET)))
        // Nope!
        onepass = TRUE;
    if (onepass && depth > 1) {
        logmsg("Reading in one pass\n");
        pixcache = bl_new(16384, bits/8);
    }

    for (plane=0; plane<depth; plane++) {
        if (plane > 0) {
            if (fseeko(fid, datastart, SEEK_SET)) {
                SYSERROR("Failed to seek back to start of image data");
                exit(-1);
            }
        }
        for (row = 0; row<H; row++) {
            unsigned int column;

#if HAVE_NETPBM
            pnm_readpamrow(&img, tuplerow);
#else
            read_pnm_row(fid, W, depth, maxval, rowbuf);
#endif

            for (column = 0; column<W; column++) {
                int rtn;
                int pixval;

#if HAVE_NETPBM
                pixval = tuplerow[column][plane];
#else
                pixval = (bits == 8 ?
                          ((uint8_t *)rowbuf)[column*depth + plane] :
                          ((uint16_t*)rowbuf)[column*depth + plane]);
#endif
                if (outformat == BPP_8_UNSIGNED)
                    rtn = fits_write_data_B(fout, pixval);
                else
                    rtn = fits_write_data_I(fout, pixval-bzero, TRUE);
                if (rtn) {
                    ERROR("Failed to write FITS pixel");
                    exit(-1);
                }
            }
            if (onepass && depth > 1) {
                for (column = 0; column<W; column++) {
                    for (plane=1; plane<depth; plane++) {
                        int pixval;
#if HAVE_NETPBM
                        pixval = tuplerow[column][plane];
#else
                        pixval = (bits == 8 ?
                                  ((uint8_t *)rowbuf)[column*depth + plane] :
                                  ((uint16_t*)rowbuf)[column*depth + plane]);
#endif
                        if (outformat == BPP_8_UNSIGNED) {
                            uint8_t pix = pixval;
                            bl_append(pixcache, &pix);
                        } else {
                            int16_t pix = pixval - bzero;
                            bl_append(pixcache, &pix);
                        }
                    }
                }
            }
        }
    }
	
#if HAVE_NETPBM
    pnm_freepamrow(tuplerow);
#else
    free(rowbuf);
#endif

    if (pixcache) {
        int i, j;
        int step = (depth - 1);
        logverb("Writing %zu queued pixels\n", bl_size(pixcache));
        for (plane=1; plane<depth; plane++) {
            j = (plane - 1);
            for (i=0; i<(W * H); i++) {
                int rtn;
                if (outformat == BPP_8_UNSIGNED) {
                    uint8_t* pix = bl_access(pixcache, j);
                    rtn = fits_write_data_B(fout, *pix);
                } else {
                    int16_t* pix = bl_access(pixcache, j);
                    rtn = fits_write_data_I(fout, *pix, TRUE);
                }
                if (rtn) {
                    ERROR("Failed to write FITS pixel");
                    exit(-1);
                }
                j += step;
            }
        }
        bl_free(pixcache);
    }

    if (fid != stdin)
        fclose(fid);

    if (fits_pad_file(fout)) {
        ERROR("Failed to pad output file \"%s\"", outfn);
        return -1;
    }

    if (fout != stdout)
        if (fclose(fout)) {
            SYSERROR("Failed to close output file %s", outfn);
            exit(-1);
        }

    return 0;
}
