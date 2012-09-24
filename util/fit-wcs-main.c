#include "starutil.h"
#include "mathutil.h"
#include "fit-wcs.h"
#include "xylist.h"
#include "rdlist.h"
#include "boilerplate.h"
#include "sip.h"
#include "sip_qfits.h"
#include "fitsioutils.h"
#include "anwcs.h"
#include "log.h"
#include "errors.h"

static const char* OPTIONS = "hx:X:Y:R:D:c:r:o:v";

void print_help(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s\n"
		   "   -x <xyls input file>\n"
		   "     [-X <x-column-name> -Y <y-column-name>]\n"
		   "   -r <rdls input file>\n"
		   "     [-R <RA-column-name> -D <Dec-column-name>]\n"
		   " OR\n"
		   "   -c <corr file>\n"
		   "\n"
		   "   -o <WCS output file>\n"
           "   [-v]: verbose\n"
		   "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** args) {
	int c;
	char* xylsfn = NULL;
	char* rdlsfn = NULL;
	char* corrfn = NULL;
	char* outfn = NULL;
	char* xcol = NULL;
	char* ycol = NULL;
	char* rcol = NULL;
	char* dcol = NULL;

	xylist_t* xyls = NULL;
	rdlist_t* rdls = NULL;
	rd_t rd;
	starxy_t xy;
	int fieldnum = 1;
	int N;
	double* fieldxy = NULL;
	double* xyz = NULL;
	tan_t wcs;
	int rtn = -1;
    int loglvl = LOG_MSG;

	fits_use_error_system();

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
        case 'h':
			print_help(args[0]);
			exit(0);
		case 'c':
			corrfn = optarg;
			break;
		case 'r':
			rdlsfn = optarg;
			break;
		case 'R':
			rcol = optarg;
			break;
		case 'D':
			dcol = optarg;
			break;
		case 'x':
			xylsfn = optarg;
			break;
		case 'X':
			xcol = optarg;
			break;
		case 'Y':
			ycol = optarg;
			break;
		case 'o':
			outfn = optarg;
			break;
        case 'v':
            loglvl++;
            break;
		}
	}
	if (optind != argc) {
		print_help(args[0]);
		exit(-1);
	}
	if (! ((xylsfn && rdlsfn) || corrfn) || !outfn) {
		print_help(args[0]);
		exit(-1);
	}
    log_init(loglvl);

	if (corrfn) {
		xylsfn = corrfn;
		rdlsfn = corrfn;
		if (!xcol)
			xcol = "FIELD_X";
		if (!ycol)
			ycol = "FIELD_Y";
		if (!rcol)
			rcol = "INDEX_RA";
		if (!dcol)
			dcol = "INDEX_DEC";
	}

	// read XYLS.
	xyls = xylist_open(xylsfn);
	if (!xyls) {
		ERROR("Failed to read an xylist from file %s", xylsfn);
		goto bailout;
	}
    xylist_set_include_flux(xyls, FALSE);
    xylist_set_include_background(xyls, FALSE);
	if (xcol)
		xylist_set_xname(xyls, xcol);
	if (ycol)
		xylist_set_yname(xyls, ycol);

	// read RDLS.
	rdls = rdlist_open(rdlsfn);
	if (!rdls) {
		ERROR("Failed to read an RA,Dec list from file %s", rdlsfn);
        goto bailout;
	}
	if (rcol)
        rdlist_set_raname(rdls, rcol);
	if (dcol)
		rdlist_set_decname(rdls, dcol);

	if (!xylist_read_field_num(xyls, fieldnum, &xy)) {
		ERROR("Failed to read xyls file %s, field %i", xylsfn, fieldnum);
		goto bailout;
	}
	if (!rdlist_read_field_num(rdls, fieldnum, &rd)) {
		ERROR("Failed to read rdls field %i", fieldnum);
		goto bailout;
	}

	N = starxy_n(&xy);
	if (rd_n(&rd) != N) {
		ERROR("X,Y list and RA,Dec list must have the same number of entries, "
			  "but found %i vs %i", N, rd_n(&rd));
		goto bailout;
	}
	logverb("Read %i points from %s and %s\n", N, rdlsfn, xylsfn);

	xyz = (double*)malloc(sizeof(double) * 3 * N);
	if (!xyz) {
		ERROR("Failed to allocate %i xyz coords", N);
		goto bailout;
	}
	radecdeg2xyzarrmany(rd.ra, rd.dec, xyz, N);

	fieldxy = starxy_to_xy_array(&xy, NULL);
	if (!fieldxy) {
		ERROR("Failed to allocate %i xy coords", N);
		goto bailout;
	}

	logverb("Fitting WCS\n");
	if (fit_tan_wcs(xyz, fieldxy, N, &wcs, NULL)) {
		ERROR("Failed to fit for TAN WCS");
		goto bailout;
	}

	if (tan_write_to_file(&wcs, outfn)) {
		ERROR("Failed to write TAN WCS header to file \"%s\"", outfn);
		goto bailout;
	}
	logverb("Wrote WCS to %s\n", outfn);

	starxy_free_data(&xy);
	rd_free_data(&rd);

	rtn = 0;

 bailout:
    if (rdls)
        rdlist_close(rdls);
    if (xyls)
        xylist_close(xyls);
	if (fieldxy)
		free(fieldxy);
	if (xyz)
		free(xyz);

	return rtn;
}
