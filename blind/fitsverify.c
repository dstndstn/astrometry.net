/*
 This file is part of "fitsverify" and was imported from:
 http://heasarc.gsfc.nasa.gov/docs/software/ftools/fitsverify/
 */
#include <stdio.h>
#include <string.h>
#include "fverify.h"

/* prototypes for PIL interface routines, that are not actually needed
 for this standalone version of fverify 
 */
#define PIL_LINESIZE 1024
int PILGetFname(char *parname, char *filename);
int PILGetString(char *parname, char *stringname);
int PILGetBool(char *parname, int *intvalue);
int PILPutInt(char *parname, int intvalue);


int ftverify_work (char *infile, char *outfile,int prehead,
                   int prstat, char* errreport, int testdata, int testcsum,
                   int testfill, int heasarc_conv);


/*
 This file contains the main fverify routine, and dummy version of 
 various other headas system routines.  This is used for the stand
 alone version of fverify.
 */

int main(int argc, char *argv[])
{
    int status = 0, invalid = 0, ii, file1 = 0;
    char *filename, errormode[2] = {"w"};

    if (argc == 2 && !strcmp(argv[1],"-h")) {

        printf("fitsverify -- Verify that the input files conform to the FITS Standard.\n");
        printf("\n");
        printf("USAGE:   fitsverify filename ...  - verify one or more FITS files\n");
        printf("                                    (may use wildcard characters)\n");
        printf("   or    fitsverify @filelist.txt - verify a list of FITS files\n");
        printf("      \n");
        printf("   Optional flags:\n");
        printf("          -l  list all header keywords\n");
        printf("          -q  quiet; print one-line pass/fail summary per file\n");
        printf("          -e  only test for error conditions (ignore warnings)\n");
        printf(" \n");
        printf("   fitsverify exits with a status equal to the number of errors + warnings.\n");
        printf("        \n");
        printf("EXAMPLES:\n");
        printf("     fitsverify -l m101.fits    - produce a detailed verificaton report of\n");
        printf("                                  a single file, including a keyword listing\n");
        printf("     fitsverify -q *.fits *.fit - verify all files with .fits or .fit\n");
        printf("                                  extensions, writing a 1-line pass/fail\n");
        printf("                                  message for each file\n");
        printf(" \n");
        printf("DESCRIPTION:\n");
        printf("    \n");
        printf("    This task reads one or more input FITS files and verifies that the\n");
        printf("    files conform to the specifications of the FITS Standard, Definition\n");
        printf("    of the Flexible Image Transport System (FITS), Version 3.0, available");
        printf("    online  at http://fits.gsfc.nasa.gov/.  The input filename template may\n");
        printf("    contain wildcard characters, in which case all matching files will be \n");
        printf("    tested.  Alternatively, the name of an ASCII text file containing a list\n");
        printf("    of file names, one per line, may be entered preceded by an '@' character.\n"); 
        printf("    The following error or warning conditions will be reported:\n");
        printf("    \n");
        printf("    ERROR CONDITIONS\n");
        printf("    \n");
        printf("     - Mandatory keyword not present or out of order\n");
        printf("     - Mandatory keyword has wrong datatype or illegal value\n");
        printf("     - END header keyword is not present\n");
        printf("     - Sum of table column widths is inconsistent with NAXIS1 value\n");
        printf("     - BLANK keyword present in image with floating-point datatype\n");
        printf("     - TNULLn keyword present for floating-point binary table column\n");
        printf("     - Bit column has non-zero fill bits or is not left adjusted \n");
        printf("     - ASCII TABLE column contains illegal value inconsistent with TFORMn\n");
        printf("     - Address to a variable length array not within the data heap \n");
        printf("     - Extraneous bytes in the FITS file following the last HDU    \n");
        printf("     - Mandatory keyword values not expressed in fixed format\n");
        printf("     - Mandatory keyword duplicated elsewhere in the header\n");
        printf("     - Header contains illegal ASCII character (not ASCII 32 - 126)\n");
        printf("     - Keyword name contains illegal character\n");
        printf("     - Keyword value field has illegal format\n");
        printf("     - Value and comment fields not separated by a slash character\n");
        printf("     - END keyword not filled with blanks in columns 9 - 80\n");
        printf("     - Reserved keyword with wrong datatype or illegal value\n");
        printf("     - XTENSION keyword in the primary array\n");
        printf("     - Column related keyword (TFIELDS, TTYPEn,TFORMn, etc.) in an image\n");
        printf("     - SIMPLE, EXTEND, or BLOCKED keyword in any extension\n");
        printf("     - BSCALE, BZERO, BUNIT, BLANK, DATAMAX, DATAMIN keywords in a table\n");
        printf("     - Table WCS keywords (TCTYPn, TCRPXn, TCRVLn, etc.) in an image\n");
        printf("     - TDIMn or THEAP keyword in an ASCII table \n");
        printf("     - TBCOLn keyword in a Binary table\n");
        printf("     - THEAP keyword in a binary table that has PCOUNT = 0 \n");
        printf("     - XTENSION, TFORMn, TDISPn or TDIMn value contains leading space(s)\n");
        printf("     - WCSAXES keyword appears after other WCS keywords\n");
        printf("     - Index of any WCS keyword (CRPIXn, CRVALn, etc.) greater than \n");
        printf("       value of WCSAXES\n");
        printf("     - Index of any table column descriptor keyword (TTYPEn, TFORMn,\n");
        printf("       etc.) greater than value of TFIELDS\n");
        printf("     - TSCALn or TZEROn present for an ASCII, logical, or Bit column\n");
        printf("     - TDISPn value is inconsistent with the column datatype \n");
        printf("     - Length of a variable length array greater than the maximum \n");
        printf("       length as given by the TFORMn keyword\n");
        printf("     - ASCII table floating-point column value does not have decimal point(*)\n");
        printf("     - ASCII table numeric column value has embedded space character\n");
        printf("     - Logical column contains illegal value not equal to 'T', 'F', or 0\n");
        printf("     - Character string column contains non-ASCII text character\n");
        printf("     - Header fill bytes not all blanks\n");
        printf("     - Data fill bytes not all blanks in ASCII tables or all zeros \n");
        printf("       in any other type of HDU \n");
        printf("     - Gaps between defined ASCII table columns contain characters with\n");
        printf("       ASCII value > 127\n");
        printf("    \n");
        printf("    WARNING CONDITIONS\n");
        printf("    \n");
        printf("     - SIMPLE = F\n");
        printf("     - Presence of deprecated keywords BLOCKED or EPOCH\n");
        printf("     - 2 HDUs have identical EXTNAME, EXTVER, and EXTLEVEL values\n");
        printf("     - BSCALE or TSCALn value = 0.\n");
        printf("     - BLANK OR TNULLn value exceeds the legal range\n");
        printf("     - TFORMn has 'rAw' format and r is not a multiple of w\n");
        printf("     - DATE = 'dd/mm/yy' and yy is less than 10 (Y2K problem?)\n");
        printf("     - Index of any WCS keyword (CRPIXn, CRVALn, etc.) greater than\n");
        printf("       value of NAXIS, if the WCSAXES keyword is not present\n");
        printf("     - Duplicated keyword (except COMMENT, HISTORY, blank, etc.)\n");
        printf("     - Column name (TTYPEn) does not exist or contains characters \n");
        printf("       other than letter, digit and underscore\n");
        printf("     - Calculated checksum inconsistent with CHECKSUM or DATASUM keyword\n");
        printf("        \n");
        printf("    This is the stand alone version of the FTOOLS 'fverify' program.  It is\n");
        printf("    maintained by the HEASARC at NASA/GSFC.  Any comments about this program\n");
        printf("    should be submitted to http://heasarc.gsfc.nasa.gov/cgi-bin/ftoolshelp\n");

        return(0);
    }

    prhead = 0;           /* don't print header by default */
    prstat = 1;           /* print HDU summary by default */

    /* check for flags on the command line */
    for (ii = 1; ii < argc; ii++)
        {	 
            if ((*argv[ii] != '-') || !strcmp(argv[ii],"-") ){
                file1 = ii;
                break;
            }
	    
            if (!strcmp(argv[ii],"-l")) {
                prhead = 1;
            } else if (!strcmp(argv[ii],"-e")) {
                strcpy(errormode,"e");
            } else if (!strcmp(argv[ii],"-q")) {
                prstat = 0;
            } else {
                invalid = 1;
            }
        }

    if (invalid || argc == 1 || file1 == 0) {
        /*  invalid input, so print brief help message */

        printf("\n");
        printf("fitsverify - test if the input file(s) conform to the FITS format.\n");
        printf("\n");
        printf("Usage:  fitsverify filename ...   or   fitsverify @filelist.txt\n");
        printf("\n");
        printf("  where 'filename' is a filename template (with optional wildcards), and\n");
        printf("        'filelist.txt' is an ASCII text file with a list of\n");
        printf("         FITS file names, one per line.\n");
        printf("\n");
        printf("   Optional flags:\n");
        printf("          -l  list all header keywords\n");
        printf("          -q  quiet; print one-line pass/fail summary per file\n");
        printf("          -e  only test for error conditions; don't issue warnings\n");
        printf("\n");
        printf("Help:   fitsverify -h\n");
        return(0);
    }

    /* 
     call work function to verify that infile conforms to the FITS
     standard and write report to the output file.
     */
    for (ii = file1; ii < argc; ii++) 
        {
            status = ftverify_work(
                                   argv[ii],    /* name of file to verify */
                                   "STDOUT", /* write report to this stream */
                                   prhead,      /* print listing of header keywords? */
                                   prstat,      /* print detailed summary report */
                                   errormode,   /* report errors only, or errors and warnings */
                                   1,           /* test the data  */
                                   1,           /* test checksum, if checksum keywords are present */
                                   1,           /* test data fill areas (should contain all zeros */
                                   0);          /* do not test for conformance with HEASARC convensions */
            /*    that are not required by the FITS Standard */

            if (status)
                return(status);
        }

    if  ( (totalerr + totalwrn) > 255)
        return(255);
    else
        return(totalerr + totalwrn);
}

/*------------------------------------------------------------------
 The following are all dummy stub routines for functions that are
 only needed when ftverify is built in the HEADAS environment.
 --------------------------------------------------------------------*/

int PILGetFname(char *parname, char *filename)
{
    return(0);
}

int PILGetString(char *parname, char *stringname)
{
    return(0);
}

int PILGetBool(char *parname, int *intvalue)
{
    return(0);
}

int PILPutInt(char *parname, int intvalue)
{
    return(0);
}

void set_toolname(char *taskname);
void set_toolname(char *taskname)
{
    return;
}

void set_toolversion(char *taskname);
void set_toolversion(char *taskname)
{
    return;
}

void get_toolname(char *taskname);
void get_toolname(char *taskname)
{
    strcpy(taskname, "fitsverify");
    return;
}

void get_toolversion(char *taskvers);
void get_toolversion(char *taskvers)
{
    strcpy(taskvers, "4.16");
    return;
}

void headas_clobberfile(char *filename);
void headas_clobberfile(char *filename)
{
    return;
}
void HD_ERROR_THROW(char msg[256], int status);
void HD_ERROR_THROW(char msg[256], int status)
{
    return;
}
