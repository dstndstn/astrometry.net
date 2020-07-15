/*
  This file is part of "fitsverify" and was imported from:
    http://heasarc.gsfc.nasa.gov/docs/software/ftools/fitsverify/
 */
#include "fverify.h"
static HduName  **hduname; 
static int total_err=1;  /* initialzed to 1 in case fail to open file */
static int total_warn=0;

int get_total_warn()
{
    return (total_warn);
}
int get_total_err()
{
    return (total_err);
}

/* Get the total hdu number and allocate the memory for hdu array */  
void init_hduname() 
{
    int i;
    /* allocate memories for the hdu structure array  */
    hduname = (HduName **)malloc(totalhdu*sizeof(HduName *));
    for (i=0; i < totalhdu; i++) { 
	hduname[i] = (HduName *)calloc(1, sizeof(HduName));
	hduname[i]->hdutype = -1;
        hduname[i]->errnum = 0; 
        hduname[i]->wrnno = 0;
        strcpy(hduname[i]->extname,"");
        hduname[i]->extver = 0;
    }
    return;
}
/* set the hduname memeber hdutype, extname, extver */
void set_hduname(  int hdunum,		/* hdu number */ 
		   int hdutype,		/* hdutype */
		   char* extname,	/* extension name */
                   int  extver 		/* extension version */
                )
{
    int i; 
    i = hdunum - 1;
    hduname[i]->hdutype = hdutype;
    if(extname!=NULL)
        strcpy (hduname[i]->extname,extname); 
    else 
        strcpy(hduname[i]->extname,"");
    hduname[i]->extver = extver;
    return;
} 


/* get the total errors and total warnings in this hdu */
void set_hduerr(int hdunum	/* hdu number */ 
                )
{
    int i; 
    i = hdunum - 1;
    num_err_wrn(&(hduname[i]->errnum), &(hduname[i]->wrnno));
    reset_err_wrn();   /* reset the error and warning counter */
    return;
} 

/* set the basic information for hduname structure */
void set_hdubasic(int hdunum,int hdutype)
{ 
   set_hduname(hdunum, hdutype, NULL, 0);
   set_hduerr(hdunum); 
   return;
}

/* test to see whether the two extension having the same name */
/* return 1: identical 0: different */
int test_hduname (int hdunum1,		/* index of first hdu */
		  int hdunum2		/* index of second hdu */
                  )
{ 
    HduName *p1;
    HduName *p2; 

    p1 = hduname[hdunum1-1];
    p2 = hduname[hdunum2-1];
    if(!strlen(p1->extname) || !strlen(p2->extname)) return 0;
    if(!strcmp(p1->extname,p2->extname) && p1->hdutype == p2->hdutype
       && p2->extver == p1->extver && hdunum1 != hdunum2){ 
	   return 1;
    } 
    return 0;
} 

/* Added the error numbers */
void total_errors (int *toterr, int * totwrn)  
{    
   int i = 0;
   int ierr, iwrn;
   *toterr = 0;
   *totwrn = 0;

   if (totalhdu == 0) { /* this means the file couldn't be opened */
       *toterr = 1;
       return;
   }

   for (i = 0; i < totalhdu; i++) { 
       *toterr += hduname[i]->errnum; 
       *totwrn += hduname[i]->wrnno;
   } 
   /*check the end of file errors */
   num_err_wrn(&ierr, &iwrn); 
   *toterr +=ierr; 
   *totwrn +=iwrn; 
   return;
}
    
/* print the extname, exttype, extver, errnum and wrnno in a  table */ 
void hdus_summary(FILE *out)
{
   HduName *p;
   int i;
   int ierr, iwrn;
   char temp[FLEN_VALUE];
   char temp1[FLEN_VALUE];

   wrtsep(out,'+'," Error Summary  ",60);
   wrtout(out," ");
   sprintf(comm," HDU#  Name (version)       Type             Warnings  Errors");
   wrtout(out,comm);

   sprintf(comm," 1                          Primary Array    %-4d      %-4d  ", 
	   hduname[0]->wrnno,hduname[0]->errnum); 
   wrtout(out,comm);
   for (i=2; i <= totalhdu; i++) { 
       p = hduname[i-1];
       strcpy(temp,p->extname);
       if(p->extver && p->extver!= -999) { 
           sprintf(temp1," (%-d)",p->extver);
           strcat(temp,temp1);
       }
       switch(hduname[i-1]->hdutype){ 
	   case IMAGE_HDU: 
               sprintf(comm," %-5d %-20s Image Array      %-4d      %-4d  ", 
	               i,temp, p->wrnno,p->errnum); 
               wrtout(out,comm); 
	       break;
	   case ASCII_TBL: 
               sprintf(comm," %-5d %-20s ASCII Table      %-4d      %-4d  ", 
	               i,temp, p->wrnno,p->errnum); 
               wrtout(out,comm); 
	       break;
	   case BINARY_TBL: 
               sprintf(comm," %-5d %-20s Binary Table     %-4d      %-4d  ", 
	               i,temp, p->wrnno,p->errnum); 
               wrtout(out,comm); 
	       break;
           default:
               sprintf(comm," %-5d %-20s Unknown HDU      %-4d      %-4d  ", 
	               i,temp, p->wrnno,p->errnum); 
               wrtout(out,comm); 
	       break; 
      }
   } 
   /* check the end of file */ 
   num_err_wrn(&ierr, &iwrn); 
   if (iwrn || ierr) {
     sprintf(comm," End-of-file %-30s  %-4d      %-4d  ", "", iwrn,ierr); 
     wrtout(out,comm); 
   }
   wrtout(out," ");
   return;
}

		   

void destroy_hduname() 
{ 
   int i;
   for (i=0; i < totalhdu; i++) free(hduname[i]);
   free(hduname);
   return;
} 

/* Routine to test the extra bytes at the end of file */ 
   void  test_end(fitsfile *infits, 
		  FILE *out) 

{   
   int status = 0; 
   LONGLONG headstart, datastart, dataend;
   int hdutype;

   /* check whether there are any HDU left */ 
   fits_movrel_hdu(infits,1, &hdutype, &status);
   if (!status) {
       wrtout(out,"< End-of-File >");
       sprintf(errmes, 
    "There are extraneous HDU(s) beyond the end of last HDU.");
       wrterr(out,errmes,2);
       wrtout(out," ");
       return;
   }

   if (status != END_OF_FILE) { 
      wrtserr(out,"Bad HDU? ",&status,2);
      return;
   } 

   status = 0;  
   fits_clear_errmsg();
   if(ffghadll(infits, &headstart, &datastart, &dataend, &status)) 
       wrtferr(out, "",&status,1);

   /* try to move to the last byte of this extension.  */
   if (ffmbyt(infits, dataend - 1,0,&status))
   {
       sprintf(errmes, 
   "Error trying to read last byte of the file at byte %ld.", (long) dataend);
       wrterr(out,errmes,2);
       wrtout(out,"< End-of-File >");
       wrtout(out," ");
       return;
   } 

   /* try to move to what would be the first byte of the next extension. 
     If successfull, we have a problem... */

   ffmbyt(infits, dataend,0,&status);
   if(status == 0) { 
       wrtout(out,"< End-of-File >");
       sprintf(errmes, 
     "File has extra byte(s) after last HDU at byte %ld.", (long) dataend);
       wrterr(out,errmes,2);
       wrtout(out," ");
   } 

   return;
}



/******************************************************************************
* Function
*      init_report
*
*
* DESCRIPTION:
*      Initialize the fverify report
*
*******************************************************************************/
void init_report(FILE *out,              /* output file */
                 char *rootnam          /* input file name */
                 )
{
    sprintf(comm,"\n%d Header-Data Units in this file.",totalhdu);
    wrtout(out,comm);
    wrtout(out," ");

    reset_err_wrn();
    init_hduname();
}

/******************************************************************************
* Function
*      close_report
*
*
* DESCRIPTION:
*      Close the fverify report
*
*******************************************************************************/
void close_report(FILE *out              /* output file */ )
{
    int numerrs = 0;                    /* number of the errors         */
    int numwrns = 0;                    /* number of the warnings       */

    /* print out a summary of all the hdus */
    if(prstat)hdus_summary(out);
    total_errors (&numerrs, &numwrns);

    total_warn = numwrns;
    total_err  = numerrs;

    /* get the total number of errors and warnnings */
    sprintf(comm,"**** Verification found %d warning(s) and %d error(s). ****",
              numwrns, numerrs);
    wrtout(out,comm);

    update_parfile(numerrs,numwrns);
    /* destroy the hdu name */
    destroy_hduname();
    return ;
} 

