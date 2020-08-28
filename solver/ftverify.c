/*
  This file is part of "fitsverify" and was imported from:
    http://heasarc.gsfc.nasa.gov/docs/software/ftools/fitsverify/
 */
#include "fverify.h"

#ifndef STANDALONE
#include "pil.h"
#include "headas.h"
#include "headas_error.h"
#endif

#define MAXMSG 256

/*
HISTORY
-------

  The original fverify.f Fortran program was written by William Pence in 1994.
  Ning Gan rewrote fverify in C in 1998, and continued to make enhancements
        throughout 1999 and 2000.
  Ziqin Pan adapted fverify to build in the new HEADAS environment 
        in 2002, renaming it to ftverify.
  William Pence made additional enhancements, and rationalized the code
        so that fverify and ftverify share many of the same source files
        in January 2003


*  Detailed modification History
*
* MODIFICATION HISTORY:
*      1998-08-24  Ning Gan, v3.0.0
*                            Beta version: xverify v1.0
*      1998-12-18  Ning Gan, v3.0.1
*                            Beta version: xverify v1.1
*      1999-02-18  Ning Gan, v3.0.2
*                            Beta version: xverify v1.2 
*                            Added more checks for keywords.
*      1999-03-04  Ning Gan, v3.0.3
*                            Added a feature of multiple input files. 
*      1999-05-19  Ning Gan, v3.0.5
*                            Wrote numwrns and numerrs to the par file.
*                            If the # of errors exceeds the MAXERRORS,
*                            quit and wrote the summary.
*                            Took out the limits on warnings. 
*      1999-06-03  Ning Gan, v3.0.6
*                            Wrote the version number of underlying
*                            cfitsio.
*      1999-06-07  Ning Gan, v3.0.7
*                            Improve the error handling. If there are
*                            errors on opening fitsfile, the program set
*                            numerr to 1 and quit. 
*      1999-06-30  Ning Gan, v3.0.8
*                            Improve the layout of the output.
*      1999-08-25  Ning Gan, v3.0.9
*                            Always write errors to stderr. 
*                            Added ffnchk
*                            Took out the checks of rejecting the
*                            TDISP like I2.0 and the column name
*                            startingnerror with a digit.
*      1999-10-25  Ning Gan, v3.1.0
*                            The TDISP can take format I4.
*                            Beutified the CFISIO error stack output
*                            The numerrs and numwrns are the accumulated
*                            number of errors and warnings if multiple
*                            FITS file are tested in on fverify session.
*                            Checked the X Column is left justified.
*      1999-12-12  Ning Gan, v3.1.1
*                            Added the basiconly and heasarc parameters.
*      1999-12-20  Ning Gan, v3.1.2
*                            Added the parameters of errreport and prstat, 
*                            removed the parameters of basiconly, erronly and 
*                            errpluswrn.
*      1999-12-29  Ning Gan, v3.1.3
*                            fixed a bug on solaris  
*      2000-05-03  Ning Gan, v3.1.4
*                            Skip the blank column names in column name
*                            tests.   
*      2000-06-09  Ning Gan, v3.1.5
*                            Fixed the memory problem(The bug will crash
*      2000-09-26  Ning Gan, v3.1.6
*                            Fixed the TDISP format bug (not accept
*                            format such as E15.5E3).
*      2002-09-30  Ziqin Pan converted fverify to ftverify for HEADAS environ.
*      2003-01-09  W Pence   v4.0
*                            Added support for the new WCSAXES keyword
*                            Added support for random groups
*                            several small changes to the output report format
*
*      2004-06-21  W Pence   fixed reporting error when prstat=no and when
*                            opening a nonexistent or non-FITS file.
*                            Also fixed elusive memory allocation error.
*
*      2009-06-08  W Pence   updates to comply with V3.0 of the FITS Standard
*      2010-07-26  W Pence   Updates to WCS keyword checks, plux other V3.0 issues
*/

#define TOOLSUB ftverify
/* headas_main() requires that TOOLSUB be defined first */

long totalerr, totalwrn;

#ifdef STANDALONE
#include "fitsverify.c"
#else
#include "headas_main.c"
#endif

/* Function Prototypes */
int ftverify (void);
int ftverify_getpar (char *infile, char *outfile,int * prehead,
    int* prstat, char* errreport, int* testdata, int* testcsum,
    int* testfill, int* heasarc_conv);
int ftverify_work (char *infile, char *outfile,int prehead,
    int prstat, char* errreport, int testdata, int testcsum,
    int testfill, int heasarc_conv);

int err_report=0;
int prhead=0;
int prstat=1;
int testdata=1;
int testcsum=1; 
int testfill=1; 
int heasarc_conv=1;
int totalhdu=0;


/*---------------------------------------------------------------------------*/
int ftverify (void)
{
/*  Read a FITS file and verify that it conforms to the FITS standard */

    char infile[PIL_LINESIZE],outfile[PIL_LINESIZE];
    int status;
    char errreport[PIL_LINESIZE];
    
    static char taskname[80] = "ftverify";
    static char version[8] = "4.16";

    /* Register taskname and version. */

    set_toolname(taskname);
    set_toolversion(version);

    /*  get input parameters */
    status = ftverify_getpar(infile, outfile,&prhead,&prstat,
             errreport,&testdata,&testcsum,&testfill,&heasarc_conv);

    /* call work function to verify that infile conforms to the FITS
       standard and write report to the output file */
    if (!status)
        status = ftverify_work(infile, outfile,prhead,prstat,
                  errreport,testdata,testcsum,testfill,heasarc_conv);

    return(status);
}
/*---------------------------------------------------------------------------*/
int ftverify_getpar(
    char *infile,   /* O - Input file name (Fits) */
    char *outfile,  /* O - Output file name (ASCII) */
    int * prhead,  
    int * prstat,
    char * errreport,
    int * testdata,
    int * testcsum,
    int * testfill,
    int * heasarc_conv)

/*  read input parameters for the ftverify task from the .par file */
{
    int status;
    char msg[MAXMSG];

    if ((status = PILGetString("infile", infile))) {
      sprintf(msg, "Error reading the 'infile' parameter.");
      HD_ERROR_THROW(msg,status);
    }


    else if ((status = PILGetString("outfile", outfile))) {
      sprintf(msg, "Error reading the 'outfile' parameter.");
      HD_ERROR_THROW(msg,status);
    }

    else if ((status = PILGetBool("prhead", prhead))) {
      sprintf(msg, "Error reading the 'prhead' parameter.");
      HD_ERROR_THROW(msg,status);
    }

    else if ((status = PILGetBool("prstat", prstat))) {
      sprintf(msg, "Error reading the 'prstat' parameter.");
      HD_ERROR_THROW(msg,status);
    }

    else if ((status = PILGetString("errreport", errreport))) {
      sprintf(msg, "Error reading the 'errreport' parameter.");
      HD_ERROR_THROW(msg,status);
    }

    else if ((status = PILGetBool("testdata", testdata))) {
      sprintf(msg, "Error reading the 'testdata' parameter.");
      HD_ERROR_THROW(msg,status);
    }

    else if ((status = PILGetBool("tchksum", testcsum))) {
      sprintf(msg, "Error reading the 'tchksum' parameter.");
      HD_ERROR_THROW(msg,status);
    }

    else if ((status = PILGetBool("testfill", testfill))) {
      sprintf(msg, "Error reading the 'testfill' parameter.");
      HD_ERROR_THROW(msg,status);
    }

    else if ((status = PILGetBool("heasarc", heasarc_conv))) {
      sprintf(msg, "Error reading the 'heasarc' parameter.");
      HD_ERROR_THROW(msg,status);
    }

    return(status);
}
/*---------------------------------------------------------------------------*/
int ftverify_work(
    char *infile,   /* I - Input file name (Fits) */
    char *outfile,  /* I - Output file name (ASCII) */
    int  prehead,  
    int  prstat,
    char * errreport,
    int  testdata,
    int  testcsum,
    int  testfill,
    int  heasarc_conv)

/* call work function to verify that infile conforms to the FITS
       standard and write report to the output file */
{
    FILE *runfile = 0;
    FILE *outfptr = 0;
    FILE *list=0;
    int status = 0, filestatus, runnum;
    char * p;
    char task[80],runchars[30];
    char tversion[80];
    float fversion;
    int i, nerrs, nwarns;
    char msg[MAXMSG];

    /* determine 'Severe error", "Error", or "Warning" report level */
    if( *errreport == 's' || *errreport == 'S') err_report = 2;
    if( *errreport == 'e' || *errreport == 'E') err_report = 1;

    p = infile;
    if (*p == '@') {
         p++;
         if( (list = fopen(p,"r")) == NULL ) {
                fprintf(stderr,"Cannot open the list file: %s.",p);
                leave_early(NULL);
                return (FILE_NOT_OPENED);
         }
    }

    headas_clobberfile(outfile);  /* delete existing file if clobber=YES */
    p = outfile;

    /* test if writing output log to a disk file */
    if(prstat && strlen(p) && strcmp(p, "STDOUT") && strcmp(p, "STDERR")
      && (outfptr = fopen(p,"r") ) != NULL ) {
      sprintf(msg,"Clobber is not set. Cannot overwrite the file%s",p);
      status = FILE_NOT_CREATED;
      HD_ERROR_THROW(msg,status);
      leave_early(NULL);
      fclose(outfptr);
      return (status);
    }

    if(prstat && (!strlen(p) || !strcmp(p, "STDOUT"))) {
       outfptr = stdout;
    }
    else if(prstat && (!strlen(p) || !strcmp(p, "STDERR"))) {
       outfptr = stderr;
    }
    else if (!prstat) {
       outfptr=NULL;
    }  
    else if( (outfptr = fopen(p,"w") ) == NULL) {
       fprintf(stderr,"Error open output file %s. Using stdout instead.",
           outfile);
       outfptr = stdout;
    }

#ifdef WEBTOOL
    /* try opening and incrementing the file containing cumulative # of runs */
    runfile=fopen("/tmp.shared/fits/tmpverify/counter.fitsverify","r+"); 
/*    runfile=fopen("counter.fitsverify","r+"); */
    if (runfile) {
        fgets(runchars,20,runfile);
	runnum=atoi(runchars);
	runnum++;
	sprintf(comm,"                                           Run Number %d",runnum);
	wrtout(outfptr,comm);
        sprintf(runchars, "%d", runnum);
	fflush(runfile);
	rewind(runfile);
	fputs(runchars, runfile);
    }

#endif

    wrtout(outfptr," ");
    fits_get_version(&fversion);
    get_toolname(task); 
    get_toolversion(tversion); 
    sprintf(comm,"%s %s (CFITSIO V%.3f)",task,tversion,fversion);
    wrtsep(outfptr,' ',comm,60);
    for(i = 0; comm[i]!='\0'; i++) comm[i] = '-';
    wrtsep(outfptr,' ',comm,60);
    wrtout(outfptr," ");
    switch (err_report) {
    case 2:
    sprintf(comm, "Caution: Only checking for the most severe FITS format errors.");
        wrtout(outfptr,comm);
        break;
    case 1:
        break;
    case 0:
        break;
    }

    if(heasarc_conv) {
        sprintf(comm, "HEASARC conventions are being checked.");
        wrtout(outfptr,comm);
    }

    /* process each file */
    if (list == NULL) {
        verify_fits(infile,outfptr);
        if (outfptr == NULL) {  /* print one-line file summary */
           nerrs = get_total_err();
           nwarns = get_total_warn();
           filestatus = ((nerrs+nwarns)>0) ? 1 : 0;
	   if (filestatus)
	   {
	     if (err_report)
                printf("verification FAILED: %-20s, %d errors\n", 
	           infile, nerrs);
             else 
                printf("verification FAILED: %-20s, %d warnings and %d errors\n", 
	           infile, nwarns, nerrs);
           }
	   else
             printf("verification OK: %-20s\n", infile);
        }        
    }
    else {
       while((p = fgets(infile, FLEN_FILENAME, list))!= NULL) {
           verify_fits(infile,outfptr);

           if (outfptr == NULL) { /* print one-line file summary */
              nerrs = get_total_err();
              nwarns = get_total_warn();
              filestatus = ((nerrs+nwarns) >0) ? 1 : 0;
	      if (filestatus)
	      {
	        if (err_report)
                   printf("verification FAILED: %-20s, %d errors\n", 
	              infile, nerrs);
                else 
                   printf("verification FAILED: %-20s, %d warnings and %d errors\n", 
	              infile, nwarns, nerrs);
              }
              else
                 printf("verification OK: %-20s\n", infile);
           }        

           for (i = 1; i < 3; i++) wrtout(outfptr," ");
       }
       fclose(list);
    }

    /* close the output file  */ 
    if (outfptr != stdout && outfptr != NULL) fclose(outfptr);

    return(status);
}

/******************************************************************************
* Function
*      update_parfile
*
*
* DESCRIPTION:
*      Update the numerrs and numwrns parameters in the parfile.
*
*******************************************************************************/
    void update_parfile(int nerr, int nwrn)
{
    int status = 0;
    char parname[32];

    totalerr += (long )nerr;
    totalwrn += (long )nwrn;
    /* write the total accumulated total warnings and errors to the
       parfile */
    strcpy(parname, "numwrns");
    status=PILPutInt(parname, totalwrn);
    if(status) {
       fprintf(stderr,"Error to update the numwrns keyword.\n");
       status = 0;
    }
    strcpy(parname, "numerrs");
    status=PILPutInt(parname, totalerr);
    if(status) {
       fprintf(stderr,"Error to update the numerrs keyword.\n");
       status = 0;
    }
}


