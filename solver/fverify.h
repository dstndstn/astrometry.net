/*
  This file is part of "fitsverify" and was imported from:
    http://heasarc.gsfc.nasa.gov/docs/software/ftools/fitsverify/
 */
#ifndef _FVERIFY_H
#define _FVERIFY_H

#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include "fitsio.h"
#define MAXERRORS  200		
#define MAXWRNS  200		

static char errmes[256];
static char comm[256];
extern int prhead;
extern int testdata;
extern int testfill;
extern int testcsum;
extern int totalhdu;		/* total number of hdu */
extern int err_report;
extern int heasarc_conv;
extern int prstat;
/********************************
*				*   
*       Keywords 		*
*				*   
********************************/

typedef enum  {      STR_KEY,   /* string   key */
		     LOG_KEY, 	/* Logical key */
		     INT_KEY,	/* Integer key */
		     FLT_KEY,   /* Float key   */
		     CMI_KEY, 	/* Complex integer key */
		     CMF_KEY,	/* Complex float key */
		     COM_KEY,	/* history, comment, "  ", and end */
		     UNKNOWN	/* Unknown types */
		     } kwdtyp;
/* error number masks of  the keyword test */
#define 	BAD_STR			0X0001 
#define		NO_TRAIL_QUOTE		0X0002
#define		BAD_NUM			0X0004 
#define		LOWCASE_EXPO		0X0008 
#define		NO_TRAIL_PAREN		0X0010
#define		NO_COMMA		0X0020
#define		TOO_MANY_COMMA		0X0040
#define		BAD_REAL		0X0080
#define		BAD_IMG			0X0100
#define         BAD_LOGICAL		0x0200
#define         NO_START_SLASH		0X0400
#define         BAD_COMMENT		0x0800
#define         UNKNOWN_TYPE		0x1000

/* keyword structure */ 
typedef struct { 
    char kname[FLEN_KEYWORD];	/* fits keyword name */
    kwdtyp ktype;		/* fits keyword type */
    char kvalue[FLEN_VALUE];	/* fits keyword name */
    int kindex;			/* position at the header */
    int goodkey;		/* good keyword flag (=1 good)*/
}FitsKey;
int fits_parse_card(FILE *out, int pos, char *card, char *kname, kwdtyp *ktype,
		    char *kvalue, char *kcomm);
void get_str(char **p, char *kvalue, unsigned long *stat);
void get_log(char **p, char *kvalue, unsigned long *stat);
void get_num(char **p, char *kvalue, kwdtyp *ktype, unsigned long *stat);
void get_cmp(char **p, char *kvalue, kwdtyp *ktype, unsigned long *stat);
int check_str(FitsKey* pkey, FILE *out);
int check_int(FitsKey* pkey, FILE *out);
int check_flt(FitsKey* pkey, FILE *out);
int check_cmi(FitsKey* pkey, FILE *out);
int check_cmf(FitsKey* pkey, FILE *out);
int check_log(FitsKey* pkey, FILE *out);
int check_fixed_int(char *card, FILE *out);
int check_fixed_log(char *card, FILE *out);
int check_fixed_str(char *card, FILE *out);

void get_unknown(char **p, char *kvalue, kwdtyp *ktype, unsigned long *stat);
void get_comm(char **p, char *kcomm, unsigned long *stat);
void pr_kval_err(FILE *out, int pos, char *keyname, char *keyval,
      unsigned long stat);

/********************************
*				*   
*       Headers  		*
*				*   
********************************/
typedef struct { 
    int	 hdutype; 		/* hdutype */
    int	 hdunum; 		/* hdunum  */
    int  isgroup;		/* random group flag */ 
    int  istilecompressed;	/* tile compressed image */ 
    int  gcount;		/* gcount  */
    int  pcount;		/* pcount  */
    int  bitpix;		/* pix number */
    int  naxis;			/* number of the axis,used for image array*/
    LONGLONG *naxes;		/* dimension of each axis,used for image array*/
    int  ncols;			/* number of the columns, used for image only*/ 
    char extname[FLEN_VALUE];		/* EXTENSION NAME */
    int extver;			/* extension version */
    char **datamax;		/* strings for the maximum of the data in a column */
    char **datamin;		/* strings for the minimum of the data in a column */
    char **tnull;	        /* number of NULL values */
    int  nkeys; 		/* number of keys */
    int  tkeys; 		/* total of the keys tested*/
    int  heap;			/* heap */
    FitsKey **kwds;		/* keywords list starting from the 
				   last NAXISn keyword. The array 
				   is sorted in the ascending alphabetical 
				   order of keyword names. The last keyword END 
				   and commentary keywords are  excluded. 
				   The total number of element, tkey, is 
				   nkeys - 4 - naxis - ncomm. */
    int use_longstr;		/* flag indicates that the long string
                                   convention is used */
}FitsHdu;   

typedef struct {
     char * name;
     int index;
}ColName;

void verify_fits(char *infile, FILE *out);
void leave_early (FILE* out);
void close_err(FILE* out);
void init_hdu(fitsfile *infits, FILE *out, int hdunum, int hdutype,
	     FitsHdu *hduptr);
void test_hdu(fitsfile *infits, FILE *out, FitsHdu *hduptr);
void test_ext(fitsfile *infits, FILE *out, FitsHdu *hduptr);
void test_tbl(fitsfile *infits, FILE *out, FitsHdu *hduptr);
void test_array(fitsfile *infits, FILE *out, FitsHdu *hduptr);
void test_prm(fitsfile *infits, FILE *out, FitsHdu *hduptr);
void test_img_ext(fitsfile *infits, FILE *out, FitsHdu *hduptr);
void test_asc_ext(fitsfile *infits, FILE *out, FitsHdu *hduptr);
void test_bin_ext(fitsfile *infits, FILE *out, FitsHdu *hduptr);
void test_header(fitsfile *infits, FILE *out, FitsHdu *hduptr);
void key_match(char **strs, int nstr, char **pattern, int exact, 
	       int *ikey, int *mkey);
void test_colnam(FILE *out, FitsHdu *hduptr);
void parse_vtform(fitsfile *infits, FILE *out, FitsHdu *hduptr, 
	     int colnum, int *datacode, long *maxlen);
void print_title(FILE* out, int hdunum, int hdutype);
void print_header(FILE* out);
void print_summary(fitsfile *infits, FILE *out, FitsHdu *hduptr);
void close_hdu(FitsHdu *hduptr);


/********************************
*				*   
*       Data 	  		*
*				*   
********************************/

void test_data(fitsfile *infits, FILE *out, FitsHdu *hduptr);
void test_agap(fitsfile *infits, FILE *out, FitsHdu *hduptr);
void test_checksum(fitsfile *infits, FILE *out);
int iterdata(long totaln, long offset, long firstn, long nrows, 
	     int narrays, iteratorCol *iter_col, void *usrdata);
/********************************
*				*   
*       Files   		*
*				*   
********************************/
typedef struct { 
    int	 hdutype; 		/* hdutype */
    int	 hdunum; 		/* hdunum  */
    char extname[FLEN_VALUE];	/* extension name, used for extension*/
    int	 extver; 		/* extension version, used for extension */
    int  errnum;			/* number of errors in this hdu */
    int  wrnno;			/* number of warnning in this hdu */
}HduName;    
int get_total_warn();
int get_total_err();
void init_hduname();
void set_hduname(int hdunum,int hdutype, char* extname,int  extver);
void set_hduerr(int hdunum);
void set_hdubasic(int hdunum,int hdutype);
int test_hduname(int hdunum1, int hdunum2);
void total_errors (int *totalerr, int * totalwrn);
void hdus_summary(FILE *out);
void destroy_hduname();
void test_end(fitsfile *infits, FILE *out);
void init_report(FILE *out, char *rootnam);
void close_report(FILE *out);
void update_parfile(int numerr, int numwrn);


/********************************
*				*   
*       Miscellaneous	 	*
*				*   
********************************/
void print_fmt(FILE *out, char *temp, int nprompt);
int wrtout (FILE *out,char *comm);
int wrterr (FILE *out,char *comm, int severity);
int wrtwrn (FILE *out,char *comm, int heasarc);
int wrtferr(FILE *out, char* mess, int *status, int severity);
int wrtserr(FILE *out, char* mess, int *status, int severity);
void wrtsep (FILE *out,char fill, char *title, int nchar);
void num_err_wrn(int *num_err, int *num_wrn);
void reset_err_wrn();
int compkey (const void *key1, const void *key2);
int compcol (const void *col1, const void *col2);
int compcol (const void *col1, const void *col2);
int compstrp (const void *str1, const void *str2);
int compstre (const void *str1, const void *str2);

#endif
