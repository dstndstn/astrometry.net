/*
  This file is part of "fitsverify" and was imported from:
    http://heasarc.gsfc.nasa.gov/docs/software/ftools/fitsverify/
 */
#include "fverify.h"

/* 
the following are only needed if one calls wcslib 
#include <wcslib/wcshdr.h>
#include <wcslib/wcsfix.h>
#include <wcslib/wcs.h>
#include <wcslib/getwcstab.h>
*/

static char **cards;		/* array to store the keywords  */
static int ncards;		/* total number of the keywords */
static char **tmpkwds;          /* An  string array holding the keyword name. 
			          It is sorted in alphabetical ascending order 
				  and not includes the keywords before 
				  the first non-reserved keyword and END
				  keyword. */

static char **ttype; 
static char **tform;  
static char **tunit;

static  char temp[80];
static  char *ptemp;		/* it always pointed to the address of
                                   temp */  
static char snull[] = "";
static int curhdu;			/* current HDU index */
static int curtype;			/* current HDU type  */

/******************************************************************************
* Function
*      verify_fits 
*
* DESCRIPTION:
*      Verify individual fits file.
*
*******************************************************************************/
/* routine to verify individual fitsfile */
void verify_fits(char *infile, FILE *out)
{
    char rootnam[FLEN_FILENAME] = "";   /* Input Fits file root name */
    fitsfile *infits;                   /* input fits file pointer */
    FitsHdu fitshdu;                    /* hdu information */
    int hdutype;
    int status = 0;
    int i;
    int len;
    char *p;
    char *pfile;
    char xtension[80];

    /* take out the leading and trailing space and skip the empty line*/
    p = infile;
    while(isspace((int)*p) )p++;
    len = strlen(p);
    pfile = p;
    p += (len -1);
    for (i = len - 1; i >= 0 && isspace((int)*p); i--) {*p = '\0'; p--;}
    if(!strlen(pfile)) return;

#ifndef WEBTOOL
    wrtout(out," ");
    sprintf(comm,"File: %s",pfile);
    wrtout(out,comm);
#endif

    totalhdu = 0;

    /* discard the extension, rowfilter... */
    if(ffrtnm(pfile, rootnam, &status)) {
        wrtserr(out,"",&status,2);
        leave_early(out);
        return;
    }

    if(fits_open_file(&infits, rootnam, READONLY, &status)) {
        wrtserr(out,"",&status,2);
        leave_early(out);
        return;
    }

    /* get the total hdus */
    if(fits_get_num_hdus(infits, &totalhdu, &status)) {
        wrtserr(out,"",&status,2);
        leave_early(out);
        return;
    }

    /* initialize the report */
    init_report(out,rootnam);
    /*------------------  Hdu Loop --------------------------------*/
    for (i = 1; i <= totalhdu; i++) {
        /* move to the right hdu and do the CFITSIO test */
        hdutype = -1;
        if(fits_movabs_hdu(infits,i, &hdutype, &status) ) {
            print_title(out,i, hdutype);
            wrtferr(out,"",&status,2);
            set_hdubasic(i,hdutype);
            break;
        }

        if (i != 1 && hdutype == IMAGE_HDU) {
           /* test if this is a tile compressed image in a binary table */
           fits_read_key(infits, TSTRING, "XTENSION", xtension, NULL, &status);
           if (!strcmp(xtension, "BINTABLE") )
               print_title(out,i, BINARY_TBL);
	   else
	       print_title(out,i, hdutype);
        }  
        else
               print_title(out,i, hdutype);

        init_hdu(infits,out,i,hdutype,
            &fitshdu);                          /* initialize fitshdu  */

        test_hdu(infits,out,&fitshdu);          /* test hdu header */

        if(testdata)
            test_data(infits,out,&fitshdu);

        close_err(out);                         /* end of error report */

        if(prhead)
            print_header(out);
        if(prstat)
            print_summary(infits,out,&fitshdu);
        close_hdu(&fitshdu);                    /* clear the fitshdu  */
    }
    /* test the end of file  */
    test_end(infits,out);

    /*------------------ Closing  --------------------------------*/
    /* closing the report*/
    close_report(out);

    /* close the input fitsfile  */
    fits_close_file(infits, &status);
}

void leave_early (FILE* out)
{
    sprintf(comm,"**** Abort Verification: Fatal Error. ****");
    wrtout(out,comm);

    /* write the total number of errors and warnings to parfile*/
    update_parfile(1,0);
}

void close_err(FILE* out)
{
    int merr, mwrn;
    num_err_wrn(&merr, &mwrn);
    if(merr || mwrn ) wrtout(out," ");
    return;
}


/*************************************************************
*
*      init_hdu 
*
*   Initialize the FitsHdu, HduName and ttype, tform, tunit if 
* the hdu is a table. 
*
*	
*************************************************************/
void init_hdu(fitsfile *infits, 	/* input fits file   */ 
	     FILE*	out,	/* output ascii file */
	     int     hdunum,	/* hdu index 	     */
	     int     hdutype,	/* hdutype	     */
             FitsHdu *hduptr 
            )
{ 

    int morekeys; 
    int i,j,k,m,n;
    int status = 0;
    FitsKey ** kwds;
    char *p = 0;
    int numusrkey; 
    LONGLONG lv,lu=0L; 
    

    FitsKey tmpkey;

    hduptr->hdunum = hdunum;
    hduptr->hdutype = hdutype;

    /* curhdu and curtype are shared with print_title */
    curhdu = hdunum; /* set the current hdu number */
    curtype = hdutype; /* set the current hdu number */

    /* check the null character in the header.(only the first one will
       be recorded */ 
    lv = 0;
    lv = fits_null_check(infits, &status);
    if (lv > 0) { 
        m = (lv - 1)/80 + 1; 
        n = lv - (m - 1) * 80; 
        sprintf(errmes,
          "Byte #%d in Card#%d is a null(\\0).",n,m);
        wrterr(out,errmes,1);
        status = 0;
    } else { 
        if (status) { 
	    wrtserr(out,"",&status,1);  
            status = 0;
        } 
    }
 
    /* get the total number of keywords */
    hduptr->nkeys = 0; 
    morekeys = 0;
    if(fits_get_hdrspace(infits, &(hduptr->nkeys), &morekeys, &status))  
        wrtferr(out,"",&status,1);
    (hduptr->nkeys)++; 	/* include END keyword */

 
    /* read all the keywords  */
    ncards = hduptr->nkeys;
    cards = (char **)malloc(sizeof(char *) * ncards );
    for (i=0; i <  ncards; i++) { 
        cards[i] = (char *)malloc(sizeof(char )* FLEN_CARD );
    }
    for (i=1; i <= ncards; i++) { 
        if(fits_read_record(infits, i, cards[i-1], &status)) 
	    wrtferr(out,"",&status,1); 
    }

    /* Parse the XTENSION/SIMPLEX  keyword */ 
    fits_parse_card(out, 1, cards[0], tmpkey.kname, 
        &(tmpkey.ktype), tmpkey.kvalue,comm); 
    if( *(tmpkey.kvalue) == ' ') {
         sprintf(errmes,
     "Keyword #1, %s \"%s\" should not have leading space.",
                 tmpkey.kname,tmpkey.kvalue); 
         wrterr(out,errmes,1);  
    } 
    if(hdunum == 1) { /* SIMPLE should be logical T */
        if(strcmp(tmpkey.kname,"SIMPLE"))  
            wrterr(out, "The 1st keyword of a primary array is not SIMPLE.",1);
        if( !check_log(&tmpkey,out)|| strcmp(tmpkey.kvalue,"T")) 
	    wrtwrn(out,
    "SIMPLE != T indicates file may not conform to the FITS Standard.",0);

        check_fixed_log(cards[0], out);
    }
    else {
        if(strcmp(tmpkey.kname,"XTENSION"))  
            wrterr(out, "The 1st keyword of a extension is not XTENSION.",1);
	check_str(&tmpkey,out); 

        check_fixed_str(cards[0], out);

        /* Get the original string */ 
        p = cards[0];
        p +=10; 
        while (*p == ' ') p++; 
        p++;   /* skip the  quote */ 
	if( strncmp(p,"TABLE   ",8)  &&
	    strncmp(p,"BINTABLE",8)  && 
	    strncmp(p,"A3DTABLE",8)  && 
	    strncmp(p,"IUEIMAGE",8)  && 
	    strncmp(p,"FOREIGN ",8)  && 
	    strncmp(p,"DUMP    ",8)  && 
	    strncmp(p,"IMAGE   ",8)  )   { 
            sprintf(errmes, "Unregistered XTENSION value \"%8.8s\".",p);
            wrterr(out,errmes,1);
        }
        else { 
            if  (p[8] != '\'') { 
                sprintf(errmes, 
         "Extra \'%c\' follows the XTENSION value \"%8.8s\".",p[8],p);
                wrterr(out,errmes,1);
            }
        }

        /* test if this is a tile compressed image, stored in a binary table */
        /* If so then test the extension as binary table rather than an image */

        if (!strncmp(p,"BINTABLE",8) && hduptr->hdutype == IMAGE_HDU) {
          hduptr->hdutype = BINARY_TBL;
          hduptr->istilecompressed = 1;
        } else {
          hduptr->istilecompressed = 0;
        }
    }   


    /* read the BITPIX keywords */ 
    if(fits_read_key(infits, TINT, "BITPIX", &(hduptr->bitpix), NULL, &status))
         wrtferr(out,"",&status,2);
    check_fixed_int(cards[1], out);

    /* Read and Parse the NAXIS */
    hduptr->naxis = 0;
    if(fits_read_key(infits, TINT, "NAXIS", &(hduptr->naxis), NULL, &status))  
         wrtferr(out,"",&status,2);
    check_fixed_int(cards[2], out);

    if(hduptr->naxis!=0)  
	 hduptr->naxes = (LONGLONG *)malloc(hduptr->naxis*sizeof(LONGLONG));
    for (i = 0; i < hduptr->naxis; i++) hduptr->naxes[i] = -1;

    /* Parse the keywords NAXISn */ 
    for (j = 3; j < 3 + hduptr->naxis; j++){  
        fits_parse_card(out, 1+j,cards[j], tmpkey.kname, 
	    &(tmpkey.ktype), tmpkey.kvalue,comm); 
        p = tmpkey.kname+5; 
	if(!isdigit((int) *p))continue;
#if (USE_LL_SUFFIX == 1)
        if(check_int(&tmpkey,out)) lu = strtoll(tmpkey.kvalue,NULL,10);
#else
	if(check_int(&tmpkey,out)) lu = strtol(tmpkey.kvalue,NULL,10);
#endif
        lv = strtol(p,NULL,10); 
        if(lv > hduptr->naxis && lv <= 0) {      
            sprintf(errmes,
                  "Keyword #%d, %s is not allowed (with n > NAXIS = %d).", 
                   tmpkey.kindex,tmpkey.kname,hduptr->naxis);
            wrterr(out,errmes,1); 
        } 
        else {
             if(hduptr->naxes[lv-1] == -1) { 
                 hduptr->naxes[lv-1] = lu;
             }
             else { 
                 sprintf(errmes, "Keyword #%d, %s is duplicated.", 
                   tmpkey.kindex,tmpkey.kname);
                 wrterr(out,errmes,1);
             }
        }

        check_fixed_int(cards[j], out);
    } 

    /* check all the NAXISn are there */
    for (j = 0; j < hduptr->naxis; j++) { 
         if(hduptr->naxes[j] == -1) { 
             sprintf(errmes, 
            "Keyword NAXIS%d is not present or is out of order.", j+1);
             wrterr(out,errmes,2);
         } 
    }
       
    /* get the column number */
    hduptr->ncols = 1; 
    if(hduptr->hdutype == ASCII_TBL || hduptr->hdutype == BINARY_TBL) {  
        /* get the total number of columns  */
        if(fits_get_num_cols(infits, &(hduptr->ncols),&status)) 
            wrtferr(out,"",&status,2);	 
    }
           
    /* parse the keywords after NAXISn and prepare the array for 
       sorting. We only check the keywords after the NAXISn */ 
    n = hduptr->nkeys - 4 - hduptr->naxis ;   /* excluding the SIMPLE/XTENSION, 
						 BITPIX, NAXIS, NAXISn  
						 and END */ 
    hduptr->kwds = (FitsKey **)malloc(sizeof(FitsKey *)*n);
    for (i= 0; i < n; i++) 
        hduptr->kwds[i] = (FitsKey *)malloc(sizeof(FitsKey));	
    kwds = hduptr->kwds;
    k = 3 + hduptr->naxis;  /* index of first keyword following NAXISn. */
    m = hduptr->nkeys - 1;     /* last key  */	
    i = 0;   
    hduptr->use_longstr = 0;
    for (j = k ; j < m; j++) { 
        kwds[i]->kindex = j+1;  	/* record number */
	kwds[i]->goodkey=1;
	if(fits_parse_card(out,1+j,cards[j], kwds[i]->kname, 
		     &(kwds[i]->ktype), kwds[i]->kvalue,comm)) 
		     kwds[i]->goodkey=0;
		     
	if (kwds[i]->ktype == UNKNOWN && *(kwds[i]->kvalue) == 0)
	{
	    sprintf(errmes,
               "Keyword #%d, %s has a null value.", 
                j+1,kwds[i]->kname); 
            wrtwrn(out,errmes,0);
	}

        /* only count the non-commentary keywords */ 
	if (!strcmp(kwds[i]->kname,"CONTINUE")) { 
            hduptr->use_longstr = 1;
        }
        if( strcmp(kwds[i]->kname,"COMMENT") && 
	    strcmp(kwds[i]->kname,"HISTORY") &&
	    strcmp(kwds[i]->kname,"HIERARCH") &&
	    strcmp(kwds[i]->kname,"CONTINUE") &&
            strcmp(kwds[i]->kname,"") ) i++;
    }
    numusrkey = i; 
    hduptr->tkeys = i; 

    /* parse the END key */ 
    fits_parse_card(out,m+1,cards[hduptr->nkeys-1],
         tmpkey.kname,&(tmpkey.ktype),tmpkey.kvalue,comm) ; 
    
    /* sort the keyword in the ascending order of kname field*/ 
    qsort(kwds, numusrkey, sizeof(FitsKey *), compkey); 

    /* store addresses of sorted keyword names in a working
       array */
    tmpkwds = (char **)malloc(sizeof(char*) * numusrkey);
    for (i=0; i < numusrkey; i++)  tmpkwds[i] = kwds[i]->kname; 

    /* Initialize  the PCOUNT, GCOUNT and heap values */ 
    hduptr->pcount = -99;
    hduptr->gcount = -99; 
    hduptr->heap = -99; 
    
    /* set the random group flag (will be determined later) */ 
    hduptr->isgroup = 0; 

    /* allocate memory for datamax and datamin (will determined later)*/ 
    if(hduptr->ncols > 0) {
        hduptr->datamax = (char **)calloc(hduptr->ncols, sizeof(char *));
        hduptr->datamin = (char **)calloc(hduptr->ncols, sizeof(char *));
        hduptr->tnull   = (char **)calloc(hduptr->ncols, sizeof(char *));
        for (i = 0; i < hduptr->ncols; i++) { 
	    hduptr->datamax[i] = (char *)calloc(13,sizeof(char));
	    hduptr->datamin[i] = (char *)calloc(13,sizeof(char));
	    hduptr->tnull[i]   = (char *)calloc(12,sizeof(char));
	}     
    } 

    
    /* initialize  the extension  name and version */
    strcpy(hduptr->extname,"");
    hduptr->extver = -999;
     
    
    return;
}

/*************************************************************
*
*      test_hdu 
*
*   Test the  HDU header
*    This includes many tests of WCS keywords
*	
*************************************************************/
void test_hdu(fitsfile *infits, 	/* input fits file   */ 
	     FILE	*out,	/* output ascii file */
             FitsHdu *hduptr 
            )

{ 
    int status = 0;
    FitsKey **kwds;
    int numusrkey;
    int hdunum;
    char *p, *p2, *pname = 0;
    int i,j,k,m,n, wcsaxes = 0, taxes;
    int wcsaxesExists = 0, wcsaxesvalue = 0, wcsaxespos = 0, wcskeypos = 1000000000;
    FitsKey *pkey;
    int crota2_exists = 0, matrix_exists[2] = {0,0};  
    double dvalue;

    /* floating WCS keywords  */
    char *cfltkeys[] = {"CRPIX", "CRVAL","CDELT","CROTA",
                        "CRDER","CSYER", "PV"};
    int ncfltkeys = 7;  
    int keynum[] = {0,0,0,0,0,0,0}, nmax = 0;

    /* floating non-indexed WCS keywords  */
    char *cfltnkeys[] = {"RESTFRQ", "RESTFREQ", "RESTWAV",
			"OBSGEO-X", "OBSGEO-Y", "OBSGEO-Z", 
			"VELOSYS", "ZSOURCE", "VELANGL",
			"LONPOLE", "LATPOLE"};
    int ncfltnkeys = 11;      

    /* floating WCS keywords w/ underscore  */
    char *cflt_keys[] = {"PC","CD"};
    int ncflt_keys = 2;

    /* string WCS keywords  */
    char *cstrkeys[] = {"CTYPE", "CUNIT", "PS", "CNAME" };
    int ncstrkeys = 4;  
 
     /* string RADESYS keywords with list of allowed values  */
    char *rastrkeys[] = {"RADESYS", "RADECSYS" };
    int nrastrkeys = 2;  

     /* string spectral ref frame keywords with list of allowed values  */
    char *specstrkeys[] = {"SPECSYS", "SSYSOBS", "SSYSSRC" };
    int nspecstrkeys = 3;  


    numusrkey = hduptr->tkeys;
    kwds = hduptr->kwds;

    /* find the extension  name and version */
    strcpy(temp,"EXTNAME");
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,1,&k,&n);
    if(k> -1 ) {
         if(kwds[k]->ktype == STR_KEY)
              strcpy(hduptr->extname,kwds[k]->kvalue);
    }

    strcpy(temp,"EXTVER");
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,1,&k,&n);
    if(k> -1 ) { 
         if(kwds[k]->ktype == INT_KEY) 
               hduptr->extver = (int) strtol(kwds[k]->kvalue,NULL,10);
    }

    /* set the HduName structure */ 
    hdunum = hduptr->hdunum;
    set_hduname(hdunum,hduptr->hdutype,hduptr->extname, hduptr->extver);

    if(hduptr->hdunum == 1) { 
        test_prm(infits,out,hduptr);
    }
    else {
        /* test the keywords specific to the hdutype*/
        switch (hduptr->hdutype) { 
	    case IMAGE_HDU:  
                test_img_ext(infits,out,hduptr);
                break;
	    case ASCII_TBL:  
                test_asc_ext(infits,out,hduptr);
                break;
	    case BINARY_TBL:  
                test_bin_ext(infits,out,hduptr);
                break;
            default: 
	        break;
        }
    }
    /* test the general keywords */
    test_header(infits,out,hduptr); 

    /* test if CROTA2 exists; if so, then PCi_j must not exist */
    strcpy(temp,"CROTA2"); 
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,1,&k,&n);
    if (n == 1) {
        pkey = hduptr->kwds[k];
        crota2_exists = pkey->kindex;  
    }
    
    strcpy(temp,"WCSAXES"); 
    ptemp = temp;

    /* first find the primary WCSAXES value, if it exists */
    key_match(tmpkwds,numusrkey,&ptemp,1,&k,&n);  
    if (k >= 0) {
        j = k;
        if (check_int(kwds[j],out)) {
            pkey = hduptr->kwds[j]; 
	    wcsaxesvalue = (int) strtol(pkey->kvalue,NULL,10);
            nmax = wcsaxesvalue;
        }
    }

    /* Check and find max value of the WCSAXESa keywords */ 
    /* Use the max value when checking the range of the indexed WCS keywords. */
    /* This is a less rigorous test than if one were to test the range of the */
    /* keywords for each of the alternate WCS systems (A - Z) against the */
    /* corresponding WCSAXESa keyword.  */
    key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);  

    for (j = k; j< n + k ; j++){
	if (check_int(kwds[j],out)) {
            pkey = hduptr->kwds[j]; 
	    taxes = (int) strtol(pkey->kvalue,NULL,10);
            if (taxes > wcsaxes) wcsaxes = taxes;
            wcsaxesExists = 1;

	    /* store highest index of any wcsaxes keyword */
	    /*  (they must appear before other WCS keywords) */
	    if (pkey->kindex > wcsaxespos) wcsaxespos = pkey->kindex;
	}
    }

    /* test datatype of reserved indexed floating point WCS keywords */
    for (i = 0; i < ncfltkeys; i++) {
        strcpy(temp,cfltkeys[i]);
    	ptemp = temp;
    	key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);  
        if(k < 0) continue;

        for (j = k; j < k+n; j++) { 
            pkey = hduptr->kwds[j]; 

	    p = kwds[j]->kname; 
	    p += strlen(temp);
            if(!isdigit((int)*p)) continue;

	    if (!check_flt(pkey,out) )continue;

	    if (i == 2 ) {  /* test that CDELTi != 0 */
		dvalue = strtod(pkey->kvalue, NULL);
		if (dvalue == 0.) {
		    sprintf( errmes, 
            "Keyword #%d, %s: must have non-zero value.",
                   pkey->kindex,pkey->kname);
                   wrterr(out,errmes,1);
		}
	    }

	    if (i == 4 || i == 5 ) {  /* test that CRDERi and CSYSERi are non-negative */
		dvalue = strtod(pkey->kvalue, NULL);
		if (dvalue < 0.) {
		    sprintf( errmes, 
            "Keyword #%d, %s: must have non-negative value: %s",
                   pkey->kindex,pkey->kname,pkey->kvalue);
                   wrterr(out,errmes,1);
		}
	    }

            m = (int)strtol(p,&p2,10);
            if (wcsaxesExists) {     /* WCSAXES keyword exists */

              if (m < 1 || m > wcsaxes) { 
                 sprintf( errmes, 
            "Keyword #%d, %s: index %d is not in range 1-%d (WCSAXES).",
                   pkey->kindex,pkey->kname,m,wcsaxes);
                   wrterr(out,errmes,1);
              }

            } else {

                if (m < 1 || m > hduptr->naxis) { 
                  sprintf( errmes, 
                  "Keyword #%d, %s: index %d is not in range 1-%d (NAXIS).",
                   pkey->kindex,pkey->kname,m,hduptr->naxis);
                   wrtwrn(out,errmes,0);
                }
            }

            /* count the number of each keyword */
	    if (*p2 == 0) {  /* only test the primary set of WCS keywords */
        	keynum[i] = keynum[i] + 1;
		if (m > nmax) nmax = m;
            }

	    /* store lowest index of any wcs keyword */
	    if (pkey->kindex < wcskeypos) {
	        wcskeypos = pkey->kindex;
		pname = pkey->kname;
	    }
        } 
    }

    if (wcsaxesvalue == 0) {  /* limit value of nmax to the legal maximum */
        if (nmax > hduptr->naxis)
	    nmax = hduptr->naxis;
    } else {
        if (nmax > wcsaxesvalue)
	    nmax = wcsaxesvalue;
    }

    if (keynum[0] < nmax) { /* test number of CRPIXi keywords */
             sprintf( errmes, 
            "Some CRPIXi keywords appear to be missing; expected %d.",nmax);
             wrtwrn(out,errmes,0);
    }
    if (keynum[1] < nmax) { /* test number of CRVALi keywords */
             sprintf( errmes, 
            "Some CRVALi keywords appear to be missing; expected %d.",nmax);
             wrtwrn(out,errmes,0);
    }

    /* test datatype of reserved non-indexed floating point WCS keywords */
    for (i = 0; i < ncfltnkeys; i++) {
        strcpy(temp,cfltnkeys[i]);
    	ptemp = temp;
    	key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);  

        if(k < 0) continue;

        for (j = k; j < k+n; j++) { 
            pkey = hduptr->kwds[j]; 

	    if (!check_flt(pkey,out) )continue;
        } 
    }

    /* test datatype of reserved indexed floating point WCS keywords with "_" */
    for (i = 0; i < ncflt_keys; i++) {
        strcpy(temp,cflt_keys[i]);
    	ptemp = temp;
    	key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);  
        if(k < 0) continue;

        for (j = k; j < k+n; j++) { 
            pkey = hduptr->kwds[j]; 

	    p = kwds[j]->kname; 
	    p += strlen(temp);
            if(!isdigit((int)*p)) continue;
	    
	    p2 = strchr(p, '_');   /* 2 digits must be separated by a '_' */
	    if (!p2) continue;     

	    if (!check_flt(pkey,out) )continue;

            *p2 = '\0';   /* terminate string at the '_' */
	    
            /* test the first digit */
            m = (int)strtol(p,NULL,10);
            *p2 = '_';   /* replace the '_' */

            if (wcsaxesExists) {     /* WCSAXES keyword exists */

              if (m < 1 || m > wcsaxes) { 
                 sprintf( errmes, 
            "Keyword #%d, %s: 1st index %d is not in range 1-%d (WCSAXES).",
                   pkey->kindex,pkey->kname,m,wcsaxes);
                   wrterr(out,errmes,1);
              }

            } else {

              if (m < 1 || m > hduptr->naxis) { 
                sprintf( errmes, 
            "Keyword #%d, %s: 1st index %d is not in range 1-%d (NAXIS).",
                   pkey->kindex,pkey->kname,m,hduptr->naxis);
                   wrtwrn(out,errmes,0);
              }
 
            }

            /* test the second digit */
            p = p2 + 1;
            m = (int)strtol(p,&p2,10);

            if (wcsaxesExists) {     /* WCSAXES keyword exists */

              if (m < 1 || m > wcsaxes) { 
                 sprintf( errmes, 
            "Keyword #%d, %s: 2nd index %d is not in range 1-%d (WCSAXES).",
                   pkey->kindex,pkey->kname,m,wcsaxes);
                   wrterr(out,errmes,1);
              }

            } else {

                if (m < 1 || m > hduptr->naxis) { 
                sprintf( errmes, 
                "Keyword #%d, %s: 2nd index %d is not in range 1-%d (NAXIS).",
                   pkey->kindex,pkey->kname,m,hduptr->naxis);
                   wrtwrn(out,errmes,0);
                }
            }

	    if (*p2 == 0) { /* no alternate suffix on the PC or CD name */
	       matrix_exists[i] = pkey->kindex;
	    }

	    /* store lowest index of any wcs keyword */
	    if (pkey->kindex < wcskeypos) {
	        wcskeypos = pkey->kindex;
		pname = pkey->kname;
	    }
        } 
    }

    if (matrix_exists[0] > 0 && matrix_exists[1] > 0 ) {
       sprintf( errmes, 
            "Keywords PCi_j (#%d) and CDi_j (#%d) are mutually exclusive.",
                   matrix_exists[0],matrix_exists[1]);
                   wrterr(out,errmes,1);
    }

    if (matrix_exists[0] > 0 && crota2_exists > 0 ) {
       sprintf( errmes, 
            "Keywords PCi_j (#%d) and CROTA2 (#%d) are mutually exclusive.",
                   matrix_exists[0],crota2_exists);
                   wrterr(out,errmes,1);
    }

    /* test datatype of reserved indexed string WCS keywords */
    for (i = 0; i < ncstrkeys; i++) {
        strcpy(temp,cstrkeys[i]);
    	ptemp = temp;
        keynum[i] = 0;
    	key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);  

        if(k < 0) continue;

        for (j = k; j < k+n; j++) { 
            pkey = hduptr->kwds[j]; 

	    p = kwds[j]->kname; 
	    p += strlen(temp);
            if(!isdigit((int)*p)) continue;

	    if (!check_str(pkey,out) )continue;

            m = (int)strtol(p,&p2,10);

            if (wcsaxesExists) {     /* WCSAXES keyword exists */

              if (m < 1 || m > wcsaxes) { 
                 sprintf( errmes, 
            "Keyword #%d, %s: index %d is not in range 1-%d (WCSAXES).",
                   pkey->kindex,pkey->kname,m,wcsaxes);
                   wrterr(out,errmes,1);
              }

            } else {

                if (m < 1 || m > hduptr->naxis) { 
                   sprintf( errmes, 
                   "Keyword #%d, %s: index %d is not in range 1-%d (NAXIS).",
                   pkey->kindex,pkey->kname,m,hduptr->naxis);
                   wrtwrn(out,errmes,0);
                } 
 
            }

	    if (*p2 == 0) {  /* only test the primary set of WCS keywords */
        	keynum[i] = keynum[i] + 1;
            }

	    /* store lowest index of any wcs keyword */
	    if (pkey->kindex < wcskeypos) {
	        wcskeypos = pkey->kindex;
		pname = pkey->kname;
	    }
        } 
    }

    if (keynum[0] < nmax) {
             sprintf( errmes, 
            "Some CTYPEi keywords appear to be missing; expected %d.",nmax);
             wrtwrn(out,errmes,0);
    }

    if (wcskeypos < wcsaxespos) { 
             sprintf( errmes, 
            "WCSAXES keyword #%d appears after other WCS keyword %s #%d",
	       wcsaxespos, pname, wcskeypos);
             wrterr(out,errmes,1);
    }

    /* test datatype and value of reserved RADECSYS WCS keywords */
    for (i = 0; i < nrastrkeys; i++) {
        strcpy(temp,rastrkeys[i]);
    	ptemp = temp;
        keynum[i] = 0;
    	key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);  

        if(k < 0) continue;

        for (j = k; j < k+n; j++) { 
            pkey = hduptr->kwds[j]; 

	    p = kwds[j]->kname; 
	    p += strlen(temp);

	    if (!check_str(pkey,out) )continue;

            if (strcmp(pkey->kvalue, "ICRS") && strcmp(pkey->kvalue, "FK5") && 
	        strcmp(pkey->kvalue, "FK4") && strcmp(pkey->kvalue, "FK4-NO-E") && 
		strcmp(pkey->kvalue, "GAPPT")) {
                   sprintf( errmes, 
                   "Keyword #%d, %s has non-allowed value: %s",
                   pkey->kindex,pkey->kname,pkey->kvalue);
                   wrtwrn(out,errmes,0);
	    }
 
        } 
    }

    /* test datatype and value of reserved spectral ref frame WCS keywords */
    for (i = 0; i < nspecstrkeys; i++) {
        strcpy(temp,specstrkeys[i]);
    	ptemp = temp;
        keynum[i] = 0;
    	key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);  

        if(k < 0) continue;

        for (j = k; j < k+n; j++) { 
            pkey = hduptr->kwds[j]; 

	    p = kwds[j]->kname; 
	    p += strlen(temp);

	    if (!check_str(pkey,out) )continue;

            if (strcmp(pkey->kvalue, "TOPOCENT") && strcmp(pkey->kvalue, "GEOCENTR") && 
	        strcmp(pkey->kvalue, "BARYCENT") && strcmp(pkey->kvalue, "HELIOCEN") && 
	        strcmp(pkey->kvalue, "LSRK") && strcmp(pkey->kvalue, "LSRD") && 
	        strcmp(pkey->kvalue, "GALACTOC") && strcmp(pkey->kvalue, "LOCALGRP") && 
	        strcmp(pkey->kvalue, "CMBDIPOL") && strcmp(pkey->kvalue, "SOURCE")) {
                   sprintf( errmes, 
                   "Keyword #%d, %s has non-allowed value: %s",
                   pkey->kindex,pkey->kname,pkey->kvalue);
                   wrtwrn(out,errmes,0);
	    }
 
        } 
    }

    /* test the fill area */ 
    if(testfill) { 
	if(ffchfl(infits,&status)) { 
	    wrterr(out, 
          "The header fill area is not totally filled with blanks.",1);
        }
    }
    return ; 
}
    

/*************************************************************
*
*      test_prm 
*
*   Test the primary array header
*
*	
*************************************************************/
void test_prm(fitsfile *infits, 	/* input fits file   */ 
	     FILE*	out,	/* output ascii file */
             FitsHdu *hduptr    /* hdu information structure */ 
            )

{ 
    int i,j,k,n;
    FitsKey *pkey;
    FitsKey **kwds;
    int numusrkey;
    char *p;

    char *exlkey[] = {"XTENSION"};
    int nexlkey = 1;

    kwds = hduptr->kwds;
    numusrkey = hduptr->tkeys;
  
    /* The SIMPLE, BITPIX, NAXIS, and NAXISn keywords  have been 
       checked in CFITSIO */

    /* excluded keywords cannot be used. */ 
    for (i = 0; i < nexlkey; i++) {
        strcpy(temp,exlkey[i]);
        ptemp = temp;
        key_match(tmpkwds,numusrkey,&ptemp,1,&k,&n); 
        if( n > 0) { 
            pkey = hduptr->kwds[k]; 
	    sprintf(errmes,
               "Keyword #%d, %s is not allowed in a primary array.", 
                pkey->kindex,exlkey[i]); 
            wrterr(out,errmes,1);
        }
    } 
   
    /* Check if Random Groups file */   
    strcpy(temp,"GROUPS");
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,1,&k,&n);  
    if(k > -1){ 
        pkey = hduptr->kwds[k]; 
	if(*(pkey->kvalue) == 'T' && hduptr->naxis > 0 && hduptr->naxes[0]==0) {
          hduptr->isgroup = 1;

          check_fixed_log(cards[pkey->kindex - 1], out);
        }
    } 

    /* check the position of the EXTEND  */

/*  the EXTEND keyword is no longer required if the file contains extensions */

    if (hduptr->isgroup == 0) { 
       strcpy(temp,"EXTEND");
       ptemp = temp;
       key_match(tmpkwds,numusrkey,&ptemp,1,&k,&n); 
       if( k > 0) { 
           pkey = hduptr->kwds[k]; 

	   if(check_log(pkey,out) && *(pkey->kvalue)!='T' && totalhdu > 1) {
	      sprintf(errmes,"There are extensions but EXTEND = F.");
              wrterr(out,errmes,1); 
           }
       }
    }
      
    /* Check PCOUNT and GCOUNT  keyword */   
    strcpy(temp,"PCOUNT");
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,1,&k,&n);  
    if(k > -1) {  
        pkey = hduptr->kwds[k]; 
        /* Primary array cannot have PCOUNT */ 
	if (!hduptr->isgroup ){ 
	    sprintf(errmes,
           " Keyword #%d, %s is not allowed in a primary array.", 
            pkey->kindex,pkey->kname); 
            wrterr(out,errmes,1); 
        }
        else { 
	    if(check_int(pkey,out))   
	        hduptr->pcount = (int) strtol(pkey->kvalue,NULL,10);

            check_fixed_int(cards[pkey->kindex - 1], out);
        }
    }

    strcpy(temp,"GCOUNT");
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,1,&k,&n);  
    if(k > -1) {  
        pkey = hduptr->kwds[k]; 
        /* Primary array cannot have GCOUNT */ 
	if (!hduptr->isgroup ){ 
	    sprintf(errmes,
           " Keyword #%d, %s is not allowed in a primary array.", 
            pkey->kindex,pkey->kname); 
            wrterr(out,errmes,1); 
        }
        else { 
	    if(check_int(pkey,out))   
	        hduptr->gcount = (int) strtol(pkey->kvalue,NULL,10);

            check_fixed_int(cards[pkey->kindex - 1], out);
        }
    }

    strcpy(temp,"BLOCKED"); 
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,1,&k,&n);
    if(k > -1) { 
         pkey = hduptr->kwds[k]; 
         sprintf(errmes,
            "Keyword #%d, %s is deprecated.", 
             pkey->kindex, pkey->kname);
         wrtwrn(out,errmes,0);
	 check_log(pkey,out);

/*  no longer required
         if(pkey->kindex > 36) {
                  sprintf(errmes,
                   "Keyword #%d, BLOCKED, appears beyond keyword 36.", 
                    pkey->kindex);
                  wrterr(out,errmes,1);  
         }
*/

    }

    /*  Check PSCALn keywords (only in Random Groups) */ 
    strcpy(temp,"PSCAL");
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);  
    for (j = k; j< k + n ; j++){ 
	p = kwds[j]->kname; 
	p += 5;
        if(!isdigit((int)*p)) continue;

        if (!(hduptr->isgroup)) {
            sprintf(errmes,"Keyword #%d, %s ",
            kwds[j]->kindex,kwds[j]->kname);
            strcat(errmes,
              "is only allowed in Random Groups structures.");
            wrterr(out,errmes,1);
            continue;
        }

	if (check_flt(kwds[j],out) && strtod(kwds[j]->kvalue,NULL) == 0.0) { 
            sprintf(errmes,"Keyword #%d, %s: ",
            kwds[j]->kindex,kwds[j]->kname);
            strcat(errmes,
              "The scaling factor is zero.");
            wrtwrn(out,errmes,0); 
        }

	i = (int) strtol(p,NULL,10) -1 ;
        if(i< 0 || i >= hduptr->gcount) {
            sprintf(errmes,
      "Keyword #%d, %s: invalid index %d (> GCOUNT = %d).", 
            kwds[j]->kindex,kwds[j]->kname,i+1,hduptr->gcount); 
            wrterr(out,errmes,1);
            continue;
        }

    } 

    /*  Check PZEROn keywords (only in Random Groups) */ 
    strcpy(temp,"PZERO");
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);  
    for (j = k; j< k + n ; j++){ 
	p = kwds[j]->kname; 
	p += 5;
        if(!isdigit((int)*p)) continue;

        if (!(hduptr->isgroup)) {
            sprintf(errmes,"Keyword #%d, %s ",
            kwds[j]->kindex,kwds[j]->kname);
            strcat(errmes,
              "is only allowed in Random Groups structures.");
            wrterr(out,errmes,1);
            continue;
        }

	check_flt(kwds[j],out);
	i = (int) strtol(p,NULL,10) -1 ;
        if(i< 0 || i >= hduptr->gcount) {
            sprintf(errmes,
      "Keyword #%d, %s: invalid index %d (> GCOUNT = %d).", 
            kwds[j]->kindex,kwds[j]->kname,i+1,hduptr->gcount); 
            wrterr(out,errmes,1);
            continue;
        }     
    } 

    /*  Check PTYPEn keywords (only in Random Groups) */ 
    strcpy(temp,"PTYPE");
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);  
    for (j = k; j< k + n ; j++){ 
	p = kwds[j]->kname; 
	p += 5;
        if(!isdigit((int)*p)) continue;

        if (!(hduptr->isgroup)) {
            sprintf(errmes,"Keyword #%d, %s ",
            kwds[j]->kindex,kwds[j]->kname);
            strcat(errmes,
              "is only allowed in Random Groups structures.");
            wrterr(out,errmes,1);
            continue;
        }

	check_str(kwds[j],out);
	i = (int) strtol(p,NULL,10) -1 ;
        if(i< 0 || i >= hduptr->gcount) {
            sprintf(errmes,
      "Keyword #%d, %s: invalid index %d (> GCOUNT = %d).", 
            kwds[j]->kindex,kwds[j]->kname,i+1,hduptr->gcount); 
            wrterr(out,errmes,1);
            continue;
        }     
    } 
    test_array(infits, out, hduptr);
        
    return;
}

/*************************************************************
*
*      test_ext 
*
*   Test the extension header
*
*	
*************************************************************/
void test_ext(fitsfile *infits, 	/* input fits file   */ 
	     FILE*     out,	/* output ascii file */
	     FitsHdu  *hduptr	/* information about header */
            ) 
{
    FitsKey *pkey;
    FitsKey **kwds;
    int  i,j,k,n;
    int numusrkey;
    char *exlkey[] = {"SIMPLE","EXTEND", "BLOCKED", }; 
    int nexlkey = 3;
    char *exlnkey[] = {"PTYPE","PSCAL", "PZERO", "GROUPS", }; 
    int nexlnkey = 4;
    int hdunum;
    char *p;

    numusrkey = hduptr->tkeys;
    kwds = hduptr->kwds;
    hdunum = hduptr->hdunum;

    /* check the duplicate extensions */
    for (i = hdunum - 1; i > 0; i--) { 
        if(test_hduname(hdunum,i)) { 
            sprintf(comm, 
	    "The HDU %d and %d have identical type/name/version", 
                hdunum,i);
            wrtwrn(out,comm,0);
        }
    }

    /* check the position of the PCOUNT  */
    strcpy(temp,"PCOUNT");
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,1,&k,&n); 
    if( k < 0) { 
	sprintf(errmes,"cannot find the PCOUNT keyword.");
        wrterr(out,errmes,1);
    } 
    else {
        pkey = hduptr->kwds[k]; 
	if(check_int(pkey,out)) 
	    hduptr->pcount = (int) strtol(pkey->kvalue,NULL,10);
        if( pkey->kindex != 4 + hduptr->naxis ) {
	     sprintf(errmes,"PCOUNT is not in record %d of the header.",
                 hduptr->naxis + 4); 
             wrterr(out,errmes,1);
        } 

        check_fixed_int(cards[pkey->kindex - 1], out);
    }

    /* check the position of the GCOUNT */
    strcpy(temp,"GCOUNT");
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,1,&k,&n); 
    if( k < 0) { 
	sprintf(errmes,"cannot find the GCOUNT keyword.");
        wrterr(out,errmes,1);
    } 
    else {
        pkey = hduptr->kwds[k]; 
	if(check_int(pkey,out)) 
	    hduptr->gcount = (int) strtol(pkey->kvalue,NULL,10);
        if( pkey->kindex != 5 + hduptr->naxis ) {
	     sprintf(errmes,"GCOUNT is not in record %d of the header.",
                 hduptr->naxis + 5); 
             wrterr(out,errmes,1);
        } 

        check_fixed_int(cards[pkey->kindex - 1], out);
    }

    for (i = 0; i < nexlkey; i++) {
        strcpy(temp,exlkey[i]);
    	ptemp = temp;
    	key_match(tmpkwds,numusrkey,&ptemp,1,&k,&n);  
    	if(k > -1) {
            pkey = hduptr->kwds[k];
            sprintf( errmes, 
               "Keyword #%d, %s is not allowed in extensions.",
               pkey->kindex,pkey->kname);
            wrterr(out,errmes,1);
        } 
    }

    for (i = 0; i < nexlnkey; i++) {
        strcpy(temp,exlnkey[i]);
    	ptemp = temp;
    	key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);  
    	if(k > -1) {

          for (j = k; j< k + n ; j++){ 
	    p = kwds[j]->kname; 
	    p += 5;
            if(!isdigit((int)*p)) continue;

            pkey = hduptr->kwds[j];
            sprintf( errmes, 
               "Keyword #%d, %s is only allowed in Random Groups structures.",
               pkey->kindex,pkey->kname);
            wrterr(out,errmes,1);
          }
        } 
    }

    return;

}
/*************************************************************
*
*      test_img_ext 
*
*   Test the image extension header
*
*	
*************************************************************/
void test_img_ext(fitsfile *infits, 	/* input fits file   */ 
	     FILE*     out,	/* output ascii file */
	     FitsHdu  *hduptr	/* information about header */
            )
{
    int numusrkey;

    numusrkey = hduptr->tkeys;

    test_ext(infits,out,hduptr);
    
    /* The XTENSION, BITPIX, NAXIS, and NAXISn keywords  have been 
       checked in CFITSIO */

    if(hduptr->pcount != 0 && hduptr->pcount != -99){
        sprintf(errmes,
           "Illegal pcount value %d for image ext.",hduptr->pcount);
        wrterr(out,errmes,1);
    }

    if(hduptr->gcount !=1 && hduptr->gcount != -99){
        sprintf(errmes,
           "Illegal gcount value %d for image ext.",hduptr->gcount);
        wrterr(out,errmes,1);
    }

    test_array(infits, out, hduptr);
    
    return ;
}

/*************************************************************
*
*      test_array 
*
*   Test the keywords which are used by both the primary array 
* and image Extension. 
*
*	
*************************************************************/
void test_array(fitsfile *infits, 	/* input fits file   */ 
	     FILE*     out,	/* output ascii file */
	     FitsHdu  *hduptr	/* information about header */
            ) 
{
    int numusrkey;
    FitsKey **kwds;
    char *p;
    int i,j,k,n;
    FitsKey *pkey;

    /* excluded non-indexed keywords  */
    char *exlkeys[] = {"TFIELDS","THEAP"}; 
    int nexlkeys = 2;

    /* excluded indexed keywords */
    char *exlnkeys[] = {"TBCOL", "TFORM",
                      "TSCAL", "TZERO","TNULL",
                      "TTYPE", "TUNIT","TDISP","TDIM",
                      "TCTYP","TCUNI","TCRVL","TCDLT","TCRPX","TCROT"}; 
    int nexlnkeys = 15;
    
    /* non-indexed floating keywords  (excluding BSCALE) */
    char *fltkeys[] = {"BZERO","DATAMAX","DATAMIN"};
    int nfltkeys = 3;  

    /* non-indexed string keywords */
    char *strkeys[] = {"BUNIT"};
    int nstrkeys = 1;  
    
    numusrkey = hduptr->tkeys;
    kwds = hduptr->kwds;

    /*  Check BLANK, BSCALE keywords */ 
    strcpy(temp,"BLANK");
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,1,&k,&n);  
    if( k >= 0) {
	check_int(kwds[k],out);
        if(hduptr->bitpix < 0) { 
            sprintf(errmes,
          "Keyword #%d, %s must not be used with floating point data (BITPIX = %d).",
               kwds[k]->kindex,kwds[k]->kname, hduptr->bitpix); 
            wrterr(out,errmes,2);
        } 
    }

    strcpy(temp,"BSCALE");
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,1,&k,&n);  
    if( k >= 0) {
	if(check_flt(kwds[k],out) && strtod(kwds[k]->kvalue,NULL) == 0.0) { 
                sprintf(errmes,"Keyword #%d, %s: The scaling factor is 0.",
                kwds[k]->kindex,kwds[k]->kname);
                wrtwrn(out,errmes,0); 
        } 
    }

    /* search for excluded, non-indexed keywords */
    for (i = 0; i < nexlkeys; i++) {
        strcpy(temp,exlkeys[i]);
    	ptemp = temp;
    	key_match(tmpkwds,numusrkey,&ptemp,1,&k,&n);  
        if(k < 0) continue;
        for (j = k; j < k+n; j++) { 
            pkey = hduptr->kwds[j];
            sprintf( errmes, 
               "Keyword #%d, %s is not allowed in the array HDU.",
               pkey->kindex,pkey->kname);
            wrterr(out,errmes,1);
        } 
    }

    /* search for excluded, indexed keywords */
    for (i = 0; i < nexlnkeys; i++) {
        strcpy(temp,exlnkeys[i]);
    	ptemp = temp;
    	key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);  
        if(k < 0) continue;
        for (j = k; j < k+n; j++) { 
            pkey = hduptr->kwds[j];

	    p = kwds[j]->kname; 
	    p += strlen(temp);
            if(!isdigit((int)*p)) continue;

            sprintf( errmes, 
               "Keyword #%d, %s is not allowed in the array HDU.",
               pkey->kindex,pkey->kname);
            wrterr(out,errmes,1);
        } 
    }

    /* test datatype of reserved non-indexed floating point keywords */
    for (i = 0; i < nfltkeys; i++) {
        strcpy(temp,fltkeys[i]);
    	ptemp = temp;
    	key_match(tmpkwds,numusrkey,&ptemp,1,&k,&n);  
        if(k < 0) continue;
        for (j = k; j < k+n; j++) { 
            pkey = hduptr->kwds[j]; 
	    if (!check_flt(pkey,out)) continue;
        } 
    }
    
    /* test datatype of reserved non-indexed string keywords */
    for (i = 0; i < nstrkeys; i++) {
        strcpy(temp,strkeys[i]);
    	ptemp = temp;
    	key_match(tmpkwds,numusrkey,&ptemp,1,&k,&n);  
        if(k < 0) continue;
        for (j = k; j < k+n; j++) { 
            pkey = hduptr->kwds[j]; 
	    check_str(pkey,out);
        } 
    }

    return;
}

/*************************************************************
*
*      test_img_wcs 
*
*   Test the image WCS keywords
*
*	
*************************************************************/

/*
void test_img_wcs(fitsfile *infits, 	
	     FILE*     out,	
	     FitsHdu  *hduptr	
            )
{

    int nkeyrec, nreject, nwcs, status = 0;
    int *stat = 0, ii; 
    char *header;
    struct wcsprm *wcs;
*/

/* NOTE: WCSLIB currently doesn't provide very much diagnostic information
  about possible problems with the WCS keywords so for now, comment out this
  routine.
*/  

    /* use WCSLIB to look for inconsistencies in the WCS keywords */

    /* Read in the FITS header, excluding COMMENT and HISTORY keyrecords. */
/*
    if (fits_hdr2str(infits, 1, NULL, 0, &header, &nkeyrec, &status)) {
        sprintf(errmes,
           "test_img_ext failed to read header keywords into array %d", status);
        wrterr(out,errmes,1);
	return;
    }
*/
    /* Interpret the WCS keywords. */

/*
    if ((status = wcsbth(header, nkeyrec, WCSHDR_all, -2, 0, 0, &nreject, &nwcs,
                       &wcs))) {
        sprintf(errmes,
           "test_img_ext: wcsbth ERROR %d: %s.", status, wcshdr_errmsg[status]);
        wrterr(out,errmes,1);

        free(header);
	return;
    }

    free(header);

    if (wcs) {
        if (nwcs == 1) {
           sprintf(errmes,
               " Found 1 World Coordinate System (WCS).");
        } else {
           sprintf(errmes,
               " Found %d World Coordinate Systems (WCS).", nwcs);
        }
        wrtout(out,errmes);
    }
*/
    /* Translate non-standard WCS keyvalues and look for inconsistencies */

/* this doesn't provide any useful checks 
    stat = malloc(NWCSFIX * sizeof(int));

    if ((status = wcsfix(7, 0, wcs, stat))) {
      for (ii = 0; ii < NWCSFIX; ii++) {
        if (stat[ii] > 0) {
           sprintf(errmes, "wcsfix ERROR %d: %s.", stat[ii],
                   wcsfix_errmsg[stat[ii]]);
           wrtwrn(out,errmes,0);

        }
      }
    }

    if ((status = wcsset(wcs))) {
      sprintf(errmes,
         "wcsset ERROR %d %s.", status, wcs_errmsg[status]);
      wrtwrn(out,errmes,0);
    }
*/

/*
    status = wcsvfree(&nwcs, &wcs);

    return;
}
*/

/*************************************************************
*
*      test_tbl 
*
*   Test the table extension header and fill the tform, ttype,
*   tunit.
*
*	
*************************************************************/
void test_tbl(fitsfile *infits, 	/* input fits file   */ 
	     FILE*     out,	/* output ascii file */
	     FitsHdu  *hduptr	/* information about header */
            ) 

{
    FitsKey *pkey;
    FitsKey **kwds;
    char *p;
    char *q;
    int m,n,i,j,k; 
    long w,d,e;
    long lm;
    int mcol;

    /* excluded, non-index keywords (allowed in tile-compressed images) */
    char*  exlkey[] = {"BSCALE","BZERO", "BUNIT", "BLANK", "DATAMAX",
                       "DATAMIN" };
    int nexlkey = 6;
 
    /* floating WCS keywords  */
    char *cfltkeys[] = {"TCRVL","TCDLT","TCRPX","TCROT" };
    int ncfltkeys = 4;  

    /* string WCS keywords  */
    char *cstrkeys[] = {"TCTYP","TCUNI" };
    int ncstrkeys = 2
;  
    int numusrkey;

    numusrkey = hduptr->tkeys;
    mcol = hduptr->ncols;
    kwds = hduptr->kwds;
    
    if(mcol <= 0) goto OTHERKEY;
    /* set the ttype, ttform, tunit for tables */
    ttype =  (char **)calloc(mcol, sizeof(char *));
    tform =  (char **)calloc(mcol, sizeof(char *));
    tunit =  (char **)calloc(mcol, sizeof(char *));
    for (i=0; i< mcol; i++) {
       ttype[i] = snull;
       tform[i] = snull;
       tunit[i] = snull;
    }

    strcpy(temp,"TFIELDS");
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,1,&k,&n);  
    if( k >= 0) {
        pkey = hduptr->kwds[k];
        check_fixed_int(cards[pkey->kindex - 1], out);
    }

    strcpy(temp,"TTYPE");
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);
    for (j = k; j< k+n ; j++){
        pkey = hduptr->kwds[j];
        p = pkey->kname;
        p += 5;
        if(!isdigit((int)*p)) continue;

	check_str(pkey,out);
        i = (int) strtol(p,NULL,10) -1 ;
        if(i>= 0 && i < mcol) {
            ttype[i] = pkey->kvalue;
        } 
        else { 
            sprintf(errmes,
      "Keyword #%d, %s: invalid index %d (> TFIELD = %d).", 
            pkey->kindex,pkey->kname,i+1,mcol);
            wrterr(out,errmes,2);
        }     
    }

    strcpy(temp,"TFORM");
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);
    for (j = k; j< k + n ; j++){
        pkey = hduptr->kwds[j];
        p = pkey->kname;
        p += 5;
        if(!isdigit((int)*p)) continue;

	check_str(pkey,out);

/*  TFORMn keyword no longer required to be padded to at least 8 characters
        check_fixed_str(cards[pkey->kindex - 1], out);
*/

        if(*(pkey->kvalue) == ' ') {
            sprintf(errmes,"Keyword #%d, %s: TFORM=\"%s\" ",
            pkey->kindex,pkey->kname, pkey->kvalue);
            strcat(errmes,
                "should not have leading space.");
            wrterr(out,errmes,1);
        }

        i = (int) strtol(p,NULL,10) -1 ;
        if(i>= 0 && i < mcol) {
            tform[i] = pkey->kvalue;
        } 
        else { 
            sprintf(errmes,
      "Keyword #%d, %s: invalid index %d (> TFIELD = %d).", 
            pkey->kindex,pkey->kname,i+1,mcol);
            wrterr(out,errmes,2);
        }

        p = pkey->kvalue;
        while(*p != ' ' && *p != '\0') { 
            if( !isdigit((int)*p) && !isupper((int)*p) && *p != '.' && *p != ')'
                && *p != '(' ) { 
                sprintf(errmes,
"Keyword #%d, %s: The value %s has character %c which is not uppercase letter.",
                pkey->kindex,pkey->kname,pkey->kvalue,*p);
                wrterr(out,errmes,1);
            }

            p++;            
        }    
    }

    strcpy(temp,"TUNIT");
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);
    for (j = k; j< k + n ; j++){
        pkey = hduptr->kwds[j];
        p = pkey->kname;
        p += 5;
        if(!isdigit((int)*p)) continue;

	check_str(pkey,out);
        i = (int) strtol(p,NULL,10) -1 ;
        if(i>= 0 && i < mcol) {
            tunit[i] = pkey->kvalue;
        } 
        else { 
            sprintf(errmes,
      "Keyword #%d, %s: invalid index %d (> TFIELD = %d).", 
            pkey->kindex,pkey->kname,i+1,mcol);
            wrterr(out,errmes,1);
        }     
    }
   
    /*  Check TDISPn keywords */ 
    strcpy(temp,"TDISP");
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);  
    for (j = k; j< k + n ; j++){ 
	p = kwds[j]->kname; 
	p += 5;
        if(!isdigit((int)*p)) continue;

        if (*(kwds[j]->kvalue) == '\0') continue;  /* ignore blank string */
	check_str(kwds[j],out);
        if(*(kwds[j]->kvalue) == ' ') { 
            sprintf(errmes,"Keyword #%d, %s: TDISP=\"%s\" ", 
                kwds[j]->kindex,kwds[j]->kname,kwds[j]->kvalue); 
            strcat(errmes,
                    "should not have leading space.");
            wrterr(out,errmes,1); 
        }


	i = (int) strtol(p,NULL,10) -1 ; 
        if(i< 0 || i >= mcol ) {
            sprintf(errmes,
      "Keyword #%d, %s: invalid index %d (> TFIELD = %d).", 
            kwds[j]->kindex,kwds[j]->kname,i+1,mcol);
            wrterr(out,errmes,1);
            continue;
        }     
        p = kwds[j]->kvalue;
        switch (*p) { 
            case 'A':
                 p++;
		 w = 0;
                 w = strtol(p,NULL,10); 
                 if( !w || w == LONG_MAX || w == LONG_MIN) {
                     sprintf(errmes,
                       "Keyword #%d, %s: invalid format \"%s\".", 
                       kwds[j]->kindex,kwds[j]->kname, kwds[j]->kvalue); 
                     wrterr(out,errmes,1); 
                 } 
                 if(strchr(tform[i],'A') == NULL ){  
                     sprintf(errmes,
        "Keyword #%d, %s:  Format \"%s\" cannot be used for TFORM \"%s\".",
        kwds[j]->kindex,kwds[j]->kname, kwds[j]->kvalue, tform[i]); 
                     wrterr(out,errmes,1); 
                 } 
                 break;
            case 'L':
                 p++;
		 w = 0;
                 w = strtol(p,NULL,10); 
                 if(!w || w == LONG_MAX || w == LONG_MIN) {
                     sprintf(errmes,
                       "Keyword #%d, %s: invalid format \"%s\".", 
                       kwds[j]->kindex,kwds[j]->kname, kwds[j]->kvalue); 
                     wrterr(out,errmes,1); 
                 } 
                 if(strchr(tform[i],'L') == NULL ){  
                     sprintf(errmes,
        "Keyword #%d, %s:  Format %s cannot be used for TFORM \"%s\".",
        kwds[j]->kindex,kwds[j]->kname, kwds[j]->kvalue, tform[i]); 
                     wrterr(out,errmes,1); 
                 } 
                 break;
            case 'I': case 'B': case 'O': case 'Z':
                 p++;
		 w = 0;
                 w = strtol(p,NULL,10); 
                 if((q = strchr(p,'.')) != NULL) {
		     p = q;
                     p++;
                     lm = strtol(p,NULL,10); 
                 } 
                 else { 
                    lm = -1;    /* no minimum digit field */ 
                 }
                 if(!w || w == LONG_MAX || w == LONG_MIN  || 
                    lm == LONG_MAX || lm == LONG_MIN  || w < lm ) {
                     sprintf(errmes,
                       "Keyword #%d, %s: invalid format \"%s\".", 
                       kwds[j]->kindex,kwds[j]->kname, kwds[j]->kvalue); 
                     wrterr(out,errmes,1); 
                 } 
                 if(strchr(tform[i],'I') == NULL &&  
                    strchr(tform[i],'J') == NULL &&  
                    strchr(tform[i],'K') == NULL &&  
                    strchr(tform[i],'B') == NULL &&  
                    strchr(tform[i],'X') == NULL   ){ 
                     sprintf(errmes,
        "Keyword #%d, %s:  Format \"%s\" cannot be used for TFORM \"%s\".",
        kwds[j]->kindex,kwds[j]->kname, kwds[j]->kvalue, tform[i]); 
                     wrterr(out,errmes,1); 
                 } 
                 break;
            case 'F': 
                 p++;
		 d = -1;
		 w = 0;
                 w = strtol(p,NULL,10); 
                 if((q = strchr(p,'.')) != NULL) {
		     p = q;
                     p++;
                     d = strtol(p,NULL,10); 
                 } 
                 if(!w || w == LONG_MAX || w == LONG_MIN  || 
                    d == -1 || d == LONG_MAX || d == LONG_MIN  || w < d+1 ) {
                     sprintf(errmes,
                       "Keyword #%d, %s: invalid format \"%s\".", 
                       kwds[j]->kindex,kwds[j]->kname, kwds[j]->kvalue); 
                     wrterr(out,errmes,1); 
                 } 
                 if(strchr(tform[i],'E') == NULL &&  
                    strchr(tform[i],'F') == NULL &&  
                    strchr(tform[i],'C') == NULL &&   
                    strchr(tform[i],'D') == NULL &&  
                    strchr(tform[i],'M') == NULL &&   
                    strchr(tform[i],'I') == NULL &&  
                    strchr(tform[i],'J') == NULL &&  
                    strchr(tform[i],'B') == NULL &&  
                    strchr(tform[i],'X') == NULL  ){ 
                     sprintf(errmes,
        "Keyword #%d, %s:  Format \"%s\" cannot be used for TFORM \"%s\".",
        kwds[j]->kindex,kwds[j]->kname, kwds[j]->kvalue, tform[i]); 
                     wrterr(out,errmes,1); 
                 } 
                 break; 
            case 'E': case 'D':
                 p++; 
		 w = 0;
                 e = 0;
		 d = 0;
                 if(*p == 'N' || *p == 'S') { p++; e = 2;}
                 w = strtol(p,NULL,10); 
                 if((q = strchr(p,'.')) != NULL) {
		     p = q;
                     p++;
                     d = strtol(p,NULL,10); 
                 } 
                 if((q = strchr(p,'E')) != NULL) {
		     p = q;
                     p++;
                     e = strtol(p,NULL,10); 
                 } 
		 else {
                     e = 2;
                 }
                 if(!w || w == LONG_MAX || w == LONG_MIN  || 
                    !d || d == LONG_MAX || d == LONG_MIN  || 
                    !e || e == LONG_MAX || e == LONG_MIN  || 
                    w < d+e+3) {
                     sprintf(errmes,
                       "Keyword #%d, %s: invalid format \"%s\".", 
                       kwds[j]->kindex,kwds[j]->kname, kwds[j]->kvalue); 
                     wrterr(out,errmes,1); 
                 } 
                 if(strchr(tform[i],'E') == NULL &&  
                    strchr(tform[i],'F') == NULL &&  
                    strchr(tform[i],'C') == NULL &&   
                    strchr(tform[i],'D') == NULL &&  
                    strchr(tform[i],'M') == NULL &&   
                    strchr(tform[i],'I') == NULL &&  
                    strchr(tform[i],'J') == NULL &&  
                    strchr(tform[i],'B') == NULL &&  
                    strchr(tform[i],'X') == NULL  ){ 
                     sprintf(errmes,
        "Keyword #%d, %s:  Format \"%s\" cannot be used for TFORM \"%s\".",
        kwds[j]->kindex,kwds[j]->kname, kwds[j]->kvalue, tform[i]); 
                     wrterr(out,errmes,1); 
                 } 
                 break; 
            case 'G': 
                 p++; 
                 e = 0;
		 d = 0;
		 w = 0;
                 w = strtol(p,NULL,10); 
                 if((q = strchr(p,'.')) != NULL) {
		     p = q;
                     p++;
                     d = strtol(p,NULL,10); 
                 } 
                 if((q = strchr(p,'E')) != NULL) {
		     p = q;
                     p++;
                     e = strtol(p,NULL,10); 
                 } 
		 else {
                     e = 2;
                 }
                 if(!w || w == LONG_MAX || w == LONG_MIN  || 
                    !d || d == LONG_MAX || d == LONG_MIN  || 
                    !e || e == LONG_MAX || e == LONG_MIN  ){ 
                     sprintf(errmes,
                       "Keyword #%d, %s: invalid format \"%s\".", 
                       kwds[j]->kindex,kwds[j]->kname, kwds[j]->kvalue); 
                     wrterr(out,errmes,1); 
                 } 
                 break; 
             default: 
                 sprintf(errmes,
                   "Keyword #%d, %s: invalid format \"%s\".", 
                   kwds[j]->kindex,kwds[j]->kname, kwds[j]->kvalue); 
                 wrterr(out,errmes,1);
                 break; 
        }            
    }

OTHERKEY:
    if (!(hduptr->istilecompressed) ) { 
      /* tile compressed images can have these keywords */
      for (i = 0; i < nexlkey; i++) {
        strcpy(temp,exlkey[i]);
    	ptemp = temp;
    	key_match(tmpkwds,numusrkey,&ptemp,1,&k,&n);  
    	if(k > -1) {
            pkey = hduptr->kwds[k];
            sprintf( errmes, 
               "Keyword #%d, %s is not allowed in the Bin/ASCII table.",
               pkey->kindex,pkey->kname);
            wrterr(out,errmes,1);
        } 
      }

      /* search for excluded indexed keywords */

/* these WCS keywords are all allowed (changed July 2010) 
      for (i = 0; i < nexlkeys; i++) {
        strcpy(temp,exlkeys[i]);
    	ptemp = temp;
    	key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);  
        if(k < 0) continue;
        for (j = k; j < k+n; j++) { 
            pkey = hduptr->kwds[j];

	    p = kwds[j]->kname; 
	    p += strlen(temp);
            if(!isdigit((int)*p)) continue;

            sprintf( errmes, 
               "Keyword #%d, %s is not allowed in the Bin/ASCII table.",
               pkey->kindex,pkey->kname);
            wrterr(out,errmes,1);
        } 
      }
*/
    }

    /* test datatype of reserved indexed floating point WCS keywords */
    for (i = 0; i < ncfltkeys; i++) {
        strcpy(temp,cfltkeys[i]);
    	ptemp = temp;
    	key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);  
        if(k < 0) continue;

        for (j = k; j < k+n; j++) { 
            pkey = hduptr->kwds[j]; 

	    p = kwds[j]->kname; 
	    p += strlen(temp);
            if(!isdigit((int)*p)) continue;

	    if (!check_flt(pkey,out) )continue;

            m = (int)strtol(p,NULL,10);
            if (m < 1 || m > mcol) { 
                sprintf( errmes, 
            "Keyword #%d, %s: index %d is not in range 1-%d (TFIELD).",
                   pkey->kindex,pkey->kname,m,mcol);
                   wrterr(out,errmes,1); 
            }
        } 
    }

    /* test datatype of reserved indexed string WCS keywords */
    for (i = 0; i < ncstrkeys; i++) {
        strcpy(temp,cstrkeys[i]);
    	ptemp = temp;
    	key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);  
        if(k < 0) continue;

        for (j = k; j < k+n; j++) { 
            pkey = hduptr->kwds[j]; 

	    p = kwds[j]->kname; 
	    p += strlen(temp);
            if(!isdigit((int)*p)) continue;

	    if (!check_str(pkey,out) )continue;

            m = (int)strtol(p,NULL,10);
            if (m < 1 || m > mcol) { 
                sprintf( errmes, 
            "Keyword #%d, %s: index %d is not in range 1-%d (TFIELD).",
                   pkey->kindex,pkey->kname,m,mcol);
                   wrterr(out,errmes,1); 
            }

        } 
    }
    return;
}

/*************************************************************
*
*      test_asc_ext 
*
*   Test the ascii table extension header
*
*	
*************************************************************/
void test_asc_ext(fitsfile *infits, 	/* input fits file   */ 
	     FILE*     out,	/* output ascii file */
	     FitsHdu  *hduptr	/* information about header */
            )
{
    int numusrkey;
    FitsKey **kwds;
    FitsKey *pkey;
    char *p;
    int i,j,k; 
    int n;
    int mcol;
 
    numusrkey = hduptr->tkeys;
    kwds = hduptr->kwds;
    mcol = hduptr->ncols;

    /* The XTENSION, BITPIX, NAXIS, NAXISn, TFIELDS, PCOUNT, GCOUNT, TFORMn,  
       TBCOLn, TTYPEn keywords  have been checked in CFITSIO */ 

    /* General extension */ 
    test_ext(infits,out,hduptr);

    /* general table */ 
    test_tbl(infits,out,hduptr);

    /* Check TBCOLn */ 
    strcpy(temp,"TBCOL");
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);
    for (j = k; j< k + n ; j++){
        pkey = hduptr->kwds[j];
        p = pkey->kname;
        p += 5;
        if(!isdigit((int)*p)) continue;

        check_int(pkey,out);

        i = (int) strtol(p,NULL,10) ;
        if(i< 0 || i > mcol) {
            sprintf(errmes,
      "Keyword #%d, %s: invalid index %d (> TFIELD = %d).",
            pkey->kindex,pkey->kname,i,mcol);
            wrterr(out,errmes,1);
        }
        else {
            check_fixed_int(cards[pkey->kindex - 1], out);
        }
    }

    /*  Check TNULLn, TSCALn, and TZEORn keywords */ 
    strcpy(temp,"TNULL");
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);  
    for (j = k; j< k + n ; j++){ 
	p = kwds[j]->kname; 
	p += 5;
        if(!isdigit((int)*p)) continue;
        i = (int) strtol(p,NULL,10) -1;
        if(i< 0 || i >= mcol) {
            sprintf(errmes,
      "Keyword #%d, %s: invalid index %d (> TFIELD = %d).", 
           kwds[j]->kindex,kwds[j]->kname,i+1,mcol); 
            wrterr(out,errmes,1);
        }     
	check_str(kwds[j],out);
    } 
    
    strcpy(temp,"TSCAL");
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);  
    for (j = k; j< k + n ; j++){ 
	p = kwds[j]->kname; 
	p += 5;
        if(!isdigit((int)*p)) continue;
        i = (int) strtol(p,NULL,10) -1 ;
	if(check_flt(kwds[j],out)){
            if(strtod(kwds[j]->kvalue,NULL) == 0.0) { 
                sprintf(errmes,"Keyword #%d, %s: Scaling factor is zero.",
                kwds[j]->kindex,kwds[j]->kname);
                wrtwrn(out,errmes,0);  
            }
        }
        if(i< 0 || i >= mcol) {
            sprintf(errmes,
      "Keyword #%d, %s: invalid index %d (> TFIELD = %d).", 
            kwds[j]->kindex,kwds[j]->kname,i+1,mcol); 
            wrterr(out,errmes,1);
            continue;
        }     
        if(strchr(tform[i],'A') != NULL) { 
            sprintf(errmes,
              "Keyword #%d, %s may not be used for the A-format fields.", 
            kwds[j]->kindex,kwds[j]->kname); 
            wrterr(out,errmes,1); 
        }
    } 

    strcpy(temp,"TZERO");
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);  
    for (j = k; j< k + n ; j++){ 
	p = kwds[j]->kname; 
	p += 5;
        if(!isdigit((int)*p)) continue;
	check_flt(kwds[j],out);
	i = (int) strtol(p,NULL,10) -1 ;
        if(i< 0 || i >= mcol) {
            sprintf(errmes,
      "Keyword #%d, %s: invalid index %d (> TFIELD = %d).", 
            kwds[j]->kindex,kwds[j]->kname,i+1,mcol); 
            wrterr(out,errmes,1);
            continue;
        }     
        if(strchr(tform[i],'A') != NULL) { 
            sprintf(errmes,
              "Keyword #%d, %s may not be used for the A-format fields.", 
            kwds[j]->kindex,kwds[j]->kname); 
            wrterr(out,errmes,1); 
        }
    } 

    strcpy(temp,"TDIM");
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);  
    for (j = k; j< k + n ; j++){ 
	p = kwds[j]->kname; 
	p += 4;
        if(!isdigit((int)*p)) continue;

        pkey = hduptr->kwds[j];
        sprintf( errmes, 
           "Keyword #%d, %s is not allowed in the ASCII table.",
           pkey->kindex,pkey->kname);
        wrterr(out,errmes,1);
    } 

    strcpy(temp,"THEAP");
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,1,&k,&n);  
    if (k > -1) {
        pkey = hduptr->kwds[k];
        sprintf( errmes, 
           "Keyword #%d, %s is not allowed in the ASCII table.",
           pkey->kindex,pkey->kname);
        wrterr(out,errmes,1);
    }

    
    /* check whether the column name is unique  */
    test_colnam(out, hduptr);
    return ;
} 


/*************************************************************
*
*      test_bin_ext 
*
*   Test the binary table extension header
*
*	
*************************************************************/
void test_bin_ext(fitsfile *infits, 	/* input fits file   */ 
	     FILE*     out,	/* output ascii file */
	     FitsHdu  *hduptr	/* information about header */
            )
{
    FitsKey *pkey;
    int i,j,k,n;
    long l;
    int status = 0;
    char *p;

    int ntdim;
    long tdim[10];
    int repeat, width;
    FitsKey **kwds;
    int numusrkey;
    int mcol, vla, datatype;

    /* The indexed keywords excluded from ascii table */
    char *exlkeys[] = { "TBCOL"}; 
    int nexlkeys = 1;

    kwds = hduptr->kwds;
    numusrkey = hduptr->tkeys;
    mcol = hduptr->ncols;

    /* General extension */ 
    test_ext(infits,out,hduptr);

    /* General table */ 
    test_tbl(infits,out,hduptr);

    /* The XTENSION, BITPIX, NAXIS, NAXISn, TFIELDS, PCOUNT, GCOUNT, TFORMn,  
       TTYPEn keywords  have been checked in CFITSIO */
    
    /*  Check TNULLn keywords */ 
    strcpy(temp,"TNULL");
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);  
    for (j = k; j< k + n ; j++){ 
	p = kwds[j]->kname; 
	p += 5;
        if(!isdigit((int)*p)) continue;
	check_int(kwds[j],out);
	i = (int) strtol(p,NULL,10) -1 ;
        if(i< 0 || i >= mcol) {
            sprintf(errmes,
      "Keyword #%d, %s: invalid index %d (> TFIELD = %d).", 
           kwds[j]->kindex,kwds[j]->kname,i+1,mcol); 
            wrterr(out,errmes,1);
            continue;
        }     
        if(strchr(tform[i],'B') == NULL &&  
           strchr(tform[i],'I') == NULL &&  
           strchr(tform[i],'J') == NULL &&  
           strchr(tform[i],'K') == NULL ) { 
            sprintf(errmes,
     "Keyword #%d, %s is used for the column with format \"%s \".", 
            kwds[j]->kindex,kwds[j]->kname,tform[i]); 
            wrterr(out,errmes,2); 
        } 
        l = strtol(kwds[j]->kvalue,NULL,10); 
        if(strchr(tform[i],'B') != NULL && (
            l < 0 || l > 255) ) {
            sprintf(errmes,"Keyword #%d, %s: The value %ld", 
            kwds[j]->kindex,kwds[j]->kname, l); 
            strcat(errmes, " is not in the range of datatype B.");
            wrtwrn(out,errmes,0); 
        }
        l = strtol(kwds[j]->kvalue,NULL,10); 
        if(strchr(tform[i],'I') != NULL && (
            l < -32768 || l > 32767) ) {
            sprintf(errmes,"Keyword #%d, %s: The value %ld", 
            kwds[j]->kindex,kwds[j]->kname, l); 
            strcat(errmes, " is not in the range of datatype I ");
            wrtwrn(out,errmes,0); 
        }
    } 
    
    /*  Check TSCALn keywords */ 
    strcpy(temp,"TSCAL");
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);  
    for (j = k; j< k + n ; j++){ 
	p = kwds[j]->kname; 
	p += 5;
        if(!isdigit((int)*p)) continue;
	if (check_flt(kwds[j],out) && strtod(kwds[j]->kvalue,NULL) == 0.0) { 
            sprintf(errmes,"Keyword #%d, %s:",
            kwds[j]->kindex,kwds[j]->kname);
            strcat(errmes,
              "The scaling factor is zero.");
            wrtwrn(out,errmes,0); 
        }
	i = (int) strtol(p,NULL,10) -1 ;
        if(i< 0 || i >= mcol) {
            sprintf(errmes,
      "Keyword #%d, %s: invalid index %d (> TFIELD = %d).", 
            kwds[j]->kindex,kwds[j]->kname,i+1,mcol); 
            wrterr(out,errmes,1);
            continue;
        }     
        if(strchr(tform[i],'A') != NULL ||  
           strchr(tform[i],'L') != NULL ||  
           strchr(tform[i],'X') != NULL ) { 
            sprintf(errmes,
         "Keyword #%d, %s is used in A, L, or X column. ",
            kwds[j]->kindex,kwds[j]->kname); 
            wrterr(out,errmes,1); 
        }
    } 

    /*  Check TZEROn keywords */ 
    strcpy(temp,"TZERO");
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);  
    for (j = k; j< k + n ; j++){ 
	p = kwds[j]->kname; 
	p += 5;
        if(!isdigit((int)*p)) continue;
	check_flt(kwds[j],out);
	i = (int) strtol(p,NULL,10) -1 ;
        if(i< 0 || i >= mcol) {
            sprintf(errmes,
      "Keyword #%d, %s: invalid index %d (> TFIELD = %d).", 
            kwds[j]->kindex,kwds[j]->kname,i+1,mcol); 
            wrterr(out,errmes,1);
            continue;
        }     
        if(strchr(tform[i],'A') != NULL &&  
           strchr(tform[i],'L') != NULL &&  
           strchr(tform[i],'X') != NULL ) { 
            sprintf(errmes,
                "Keyword #%d, %s is used in A, L, or X column. ",
            kwds[j]->kindex,kwds[j]->kname); 
            wrterr(out,errmes,1); 
        }
    } 

    /* Check THEAP keyword */   
    hduptr->heap = (hduptr->naxes[0]) * (hduptr->naxes[1]);
    strcpy(temp,"THEAP");
    key_match(tmpkwds,numusrkey,&ptemp,1,&k,&n);  
    if(k > -1) { 
         if(check_int(kwds[k],out))
             hduptr->heap = (int) strtol(hduptr->kwds[k]->kvalue,NULL,10);
         if(!hduptr->pcount) { 
            sprintf( errmes, 
               "Pcount is zero, but keyword THEAP is present at record #%d). ",
	        kwds[k]->kindex);
                wrterr(out,errmes,1); 
         }
    }

    /* if PCOUNT != 0, test that there is at least 1 variable length array column */
    vla = 0;
    if(hduptr->pcount) {
        for (i=0; i< mcol; i++){ 
            if(fits_get_coltype(infits, i+1, &datatype, NULL, NULL, &status)){ 
               sprintf(errmes,"Column #%d: ",i);
 	       wrtferr(out,errmes, &status,2);
            }
            if (datatype < 0) {
	      vla = 1;
	      break;
	    }
	}

	if (vla == 0) {
	    sprintf(errmes,
	    "PCOUNT = %ld, but there are no variable-length array columns.",
	   (long) hduptr->pcount);
            wrtwrn(out,errmes,0);
	} 
    }
      
   
    /* Check TDIMn  keywords */ 
    strcpy(temp,"TDIM");
    key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);  
    for (j = k; j< k + n ; j++){ 
        pkey = kwds[j]; 
	p = pkey->kname; 
	p += 4;
        if(!isdigit((int)*p)) continue;
	check_str(kwds[j],out);
        if(*(pkey->kvalue) == ' ') { 
            sprintf(errmes,"Keyword #%d, %s: TDIM=\"%s\" ", 
                pkey->kindex,pkey->kname,pkey->kvalue); 
            strcat(errmes,
                    "should not have leading space.");
            wrterr(out,errmes,1); 
            continue;
        }
	i = (int) strtol(p,NULL,10) -1 ;
        if(i< 0 || i >= mcol) {
            sprintf(errmes,
      "Keyword #%d, %s: invalid index %d (> TFIELD = %d).", 
            kwds[j]->kindex,kwds[j]->kname,i+1,mcol); 
            wrterr(out,errmes,1);
            continue;
        }     
	if(fits_decode_tdim(infits,pkey->kvalue,i+1,10,&ntdim,tdim, &status)){ 
           sprintf(errmes,"Keyword #%d, %s: ", 
                kwds[j]->kindex,kwds[j]->kname);
	    wrtferr(out,errmes,&status,1);
        } 
    } 

    /* check the local convension "rAw"*/
    for (i = 0; i < hduptr->ncols; i++) { 
	if((p = strchr(tform[i],'A'))==NULL) continue; 
        repeat = (int) strtol(tform[i],NULL,10);
        p++;
	if(!isdigit((int)*p))continue;
	width = (int)strtol(p,NULL,10);
	if(repeat%width != 0)  { 
	    sprintf(errmes,
	 "TFORM %s of column %d: repeat %d is not the multiple of the width %d",
	    tform[i], i+1, repeat, width);
            wrtwrn(out,errmes,0);
        } 
    }
   
    for (i = 0; i < nexlkeys; i++) {
        strcpy(temp,exlkeys[i]);
    	ptemp = temp;
    	key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);  
        if(k < 0) continue;
        for (j = k; j < k+n; j++) { 
            pkey = hduptr->kwds[j];

	    p = kwds[j]->kname; 
	    p += strlen(temp);
            if(!isdigit((int)*p)) continue;

            sprintf( errmes, 
               "Keyword #%d, %s is not allowed in the Binary table.",
               pkey->kindex,pkey->kname);
            wrterr(out,errmes,1);
        } 
    }
    
    /* check whether the column name is unique */
    test_colnam(out, hduptr);
    return ;
} 

/*************************************************************
*
*      test_header
*
*   Test the general keywords that can be in any header
*
*	
*************************************************************/
void test_header(
	     fitsfile *infits, 	/* input fits file   */ 
	     FILE*     out,	/* output ascii file */
	     FitsHdu  *hduptr	/* information about header  */
)
{ 
    /* common mandatory  keywords */
    char *mandkey[] = {"SIMPLE", "BITPIX", "NAXIS",
                       "XTENSION",  "END"};  /* not including NAXIS */
    int nmandkey = 5;


    /* string keywords */ 
    char *strkey[] = {"EXTNAME", "ORIGIN", "AUTHOR","CREATOR","REFERENC","TELESCOP",
        "INSTRUME", "OBSERVER", "OBJECT"}; 
    int nstrkey = 9;

    /* int keywords  */
    char *intkey[] = {"EXTVER", "EXTLEVEL"};
    int nintkey = 2;

    /* floating keywords  */
    char *fltkey[] = {"EQUINOX", "MJD-OBS", "MJD-AVG"};
    int nfltkey = 3;

    FitsKey** kwds;		/* FitsKey structure array */ 
    int numusrkey;

    int i,j,k,n;
    long lv;
    char* pt;
    char **pp;
    int status = 0;
    int yr, mn, dy, hr, min;	/* time */
    double sec;
    int yy;

    kwds = hduptr->kwds;
    numusrkey = hduptr->tkeys;

/* Check the mandatory keywords */ 
    for (i = 0; i < nmandkey; i++) { 
	 pp = &(mandkey[i]);
         key_match(tmpkwds,numusrkey,pp,1,&k,&n);
         if(k > -1) { 
             for ( j = k; j < k + n; j++) { 
                sprintf(errmes,
      "Keyword #%d, %s is duplicated or out of order.", 
                kwds[j]->kindex,kwds[j]->kname);
             wrterr(out,errmes,1); 
             }
         }
    } 
    
    /* check the NAXIS index keyword */
    strcpy(temp,"NAXIS"); 
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);  
    for ( j = k; j < k + n; j++) {  
        pt = kwds[j]->kname+5; 
        lv = strtol(pt,NULL,10); 
        if(lv > 0 ){
            if(kwds[j]->kindex != 3 + lv) {
                sprintf(errmes,
                "Keyword #%d, %s is duplicated or out of order.", 
                kwds[j]->kindex,kwds[j]->kname);
                wrterr(out,errmes,1);  
            }
            if(lv > hduptr->naxis) {      
                sprintf(errmes,
                  "Keyword #%d, %s is not allowed (with n > NAXIS =%d).", 
                   kwds[j]->kindex,kwds[j]->kname,hduptr->naxis);
                wrterr(out,errmes,1); 
            }
        } 
    }

    /* Check the deprecated keywords */ 
    strcpy(temp,"EPOCH"); 
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,1,&k,&n);
    if(k > -1) { 
         sprintf(errmes,
            "Keyword #%d, %s is deprecated. Use EQUINOX instead.", 
             kwds[k]->kindex, kwds[k]->kname);
         wrtwrn(out,errmes,0);
	 check_flt(kwds[k],out);
    }

    
    /* Check the DATExxxx keyword */ 
    strcpy(temp,"DATE"); 
    ptemp = temp;
    key_match(tmpkwds,numusrkey,&ptemp,0,&k,&n);  
    for (j = k; j< n + k ; j++){
       check_str(kwds[j],out);
       if(fits_str2time(kwds[j]->kvalue, &yr, &mn, &dy, &hr, &min, 
          &sec, &status)){ 
           sprintf(errmes,"Keyword #%d, %s: ", kwds[j]->kindex,kwds[j]->kname);
           wrtserr(out,errmes,&status,1);
        } 
        if( (pt = strchr(kwds[j]->kvalue,'/'))!=NULL) { 
               pt +=4;
               yy = (int) strtol(pt,NULL,10); 
               if(0 <= yy && yy <=10) {
               sprintf(errmes,
                  "Keyword #%d, %s %s intends to mean year 20%-2.2d?",
                   kwds[j]->kindex, kwds[j]->kname, kwds[j]->kvalue, yy);
                   wrtwrn(out,errmes,0);
	       } 
        }
    }

    /* Check the reserved string keywords */
    for (i = 0; i < nstrkey; i++) {
        strcpy(temp,strkey[i]);
    	ptemp = temp;
    	key_match(tmpkwds,numusrkey,&ptemp,1,&k,&n);  
    	if(k > -1) check_str(kwds[k],out);
    }

    /* Check the reserved int keywords */
    for (i = 0; i < nintkey; i++) {
        strcpy(temp,intkey[i]);
    	ptemp = temp;
    	key_match(tmpkwds,numusrkey,&ptemp,1,&k,&n);  
    	if(k > -1) check_int(kwds[k],out);
    }

    /* Check  reserved floating  keywords */   
    for (i = 0; i < nfltkey; i++) {
        strcpy(temp,fltkey[i]);
        ptemp = temp;
        key_match(tmpkwds,numusrkey,&ptemp,1,&k,&n);  
    	if(k > -1) check_flt(kwds[k],out);
    }

    /* Check the duplication of the keywords */ 
    for (i = 0; i < numusrkey-1; i++) { 
       if(!strcmp(tmpkwds[i],tmpkwds[i+1])) { 
           sprintf(errmes,
     "Keyword %s is duplicated in card #%d and card #%d.", 
            kwds[i]->kname, kwds[i]->kindex, kwds[i+1]->kindex);
          wrtwrn(out,errmes,0);
       }
    }  

    /* check the long string convention */ 
    if (hduptr->use_longstr == 1) {  
        strcpy(temp,"LONGSTRN"); 
        ptemp = temp;
        key_match(tmpkwds,numusrkey,&ptemp,1,&k,&n);
        if(k <= -1) {  
            sprintf(errmes,
"The OGIP long string keyword convention is used without the recommended LONGSTRN keyword. ");
          wrtwrn(out,errmes,1);
        }
    }

/*  disabled this routine because it doesn't perform an useful tests 
    test_img_wcs(infits, out, hduptr);

*/
    return; 
}
	

/*************************************************************
*
*      key_match 
*
*   find the keywords whose name match the pattern. The keywords
*   name is stored in a sorted array. 
*
*	
*************************************************************/
void key_match(char **strs,  	/* fits keyname  array */
             int nstr,		/* total number of keys */
             char **pattern,	/* wanted pattern  */ 
             int exact,		/* exact matching or pattern matching  
                                   exact = 1: exact matching.
                                   exact = 0: pattern matching. 
                                   Any keywords with "patten"* is included
                                */
                                
             int *ikey,		/* The element number of first key 
                                   Return -99 if not found */ 
             int *mkey		/* total number of key matched 
                                   return -999 if not found */
            )  			
{ 
     char **p;
     char **pi;
     int i;
     int (*fnpt)(const void *, const void *); 
     *mkey = -999;
     *ikey = -99;
     if(exact)  
	 fnpt = compstre;
     else 
         fnpt = compstrp;
     p = (char **)bsearch(pattern, strs, nstr,sizeof(char *), fnpt); 
     if(p) {	    
        *mkey = 1; 
        *ikey = p - strs; 
	pi = p;
	i = *ikey - 1;
	p--;
        while(i > 0 && !fnpt(pattern, p)) {*mkey += 1; *ikey =i; i--; p--;}
	p = pi;
	i = *ikey + *mkey;
        p++;
        while(i < nstr && !fnpt(pattern, p) ) {*mkey += 1; i++; p++;}
     }
     return;
} 


/*************************************************************
*
*      test_colnam
*
*   Test the whether the column name is unique. 
*
*	
*************************************************************/
void test_colnam(FILE *out, 
		FitsHdu *hduptr)
{ 
    int i,n; 
    char *p, *q;
    ColName **cols; 
    char **ttypecopy;

    n = hduptr->ncols; 

    if(n <= 0) return;
    /* make a local working copy of ttype */
    ttypecopy = (char **)malloc(n*sizeof(char *));
    for (i = 0; i < n; i++) { 
        ttypecopy[i] = (char *)malloc(FLEN_VALUE*sizeof(char));
        strcpy(ttypecopy[i],ttype[i]);
    } 

    /* check whether there are any other non ASCII-text characters 
      (FITS standard R14). Also "uppercase" the working copies. */
    for (i = 0; i < n; i++) { 
        p = ttype[i];
        q = ttypecopy[i];
        if(!strlen(p)) { 
            sprintf(errmes,
            "Column #%d has no name (No TTYPE%d keyword).",i+1, i+1);
            wrtwrn(out,errmes,0);
            continue;
        }


/*      disable this check (it was only a warning) 
        if( (*p  > 'z' || *p < 'a') && (*p > 'Z' || *p <'A') 
                && (*p > '9' || *p < '0') ) { 
            sprintf(errmes,"Column #%d: Name \"%s\" does not begin with a letter or a digit.",i+1,ttype[i]);
            wrtwrn(out,errmes,1);
        }  
*/
        while(*p != '\0') { 
            if(    (*p > 'z' || *p < 'a') && (*p > 'Z' || *p < 'A')  
                && (*p > '9' || *p < '0') && (*p != '_')) { 
            sprintf(errmes, 
      "Column #%d: Name \"%s\" contains character \'%c\' other than letters, digits, and \"_\".",
            i+1,ttype[i],*p); 
            wrtwrn(out,errmes,0); 
            }
            if(*p <= 'z' || *p >= 'a') *q = toupper(*p); 
            p++; q++;
        }
    }
      
    cols = (ColName **)calloc(n, sizeof(ColName *));
    for (i=0; i < n; i++) { 
        cols[i] = (ColName *)malloc(sizeof(ColName));
	cols[i]->name = ttypecopy[i];
	cols[i]->index = i+1; 
    }
    
    /* sort the column name in the ascending order of name field*/ 
    qsort(cols, n, sizeof(ColName *), compcol); 

    /* Check the duplication of the column name */ 
    for (i = 0; i < n-1; i++) { 
        if(!strlen(cols[i]->name)) continue;

/*      disable this warning
        if(!strncmp(cols[i]->name,cols[i+1]->name,16)) { 
            sprintf(errmes,
     "Columns #%d, %s and #%d, %s are not unique within first 16 characters(case insensitive).", 
            cols[i]->index,   ttype[(cols[i]->index-1)], 
            cols[i+1]->index, ttype[(cols[i+1]->index-1)]);
          wrtwrn(out,errmes,1);
        }
*/

        if(!strcmp(cols[i]->name,cols[i+1]->name)) { 
            sprintf(errmes,
     "Columns #%d, %s and #%d, %s are not unique (case insensitive).", 
            cols[i]->index,   ttype[(cols[i]->index-1)], 
            cols[i+1]->index, ttype[(cols[i+1]->index-1)]);
          wrtwrn(out,errmes,0);
        }
    }  
    for (i = 0; i < n; i++) { free(cols[i]); free(ttypecopy[i]);}
    free(cols); free(ttypecopy);
    return;
}
	
/*************************************************************
*
*     parse_vtform 
*
*   Parse the tform of the variable length vector. 
*
*	
*************************************************************/
void   parse_vtform(fitsfile *infits,
		FILE *out, 
                FitsHdu *hduptr, 
		int colnum,		/* column number */
		int* datacode,		/* data code */
		long* maxlen		/* maximum length of the vector */
               ) 
{ 
    int i = 0; 
    int status = 0;
    char *p;
    

    *maxlen = -1;
    strcpy(temp,tform[colnum-1]); 
    p = temp; 

    if(isdigit((int)*p)) sscanf(ptemp,"%d",&i); 
    if(i > 1) { 
        sprintf(errmes,"Illegal repeat value for value %s of TFORM%d.", 
	   tform[colnum-1], colnum);
        wrterr(out,errmes,1);
    } 
    while(isdigit((int)*p))p++;

    if( (*p != 'P') && (*p != 'Q') ) { 
        sprintf(errmes,
	  "TFORM%d is not for the variable length array: %s.", 
        colnum, tform[colnum-1]); 
        wrterr(out,errmes,1);
    } 

    fits_get_coltype(infits,colnum, datacode, NULL, NULL, &status); 
    status = 0;
    p += 2;
    if(*p != '(') return;
    p++; 
    if(!isdigit((int)*p)) { 
       sprintf(errmes, "Bad value of TFORM%d: %s.",colnum,tform[colnum-1]);
       wrterr(out,errmes,1);
    } 
    sscanf(p,"%ld",maxlen);
    while(isdigit((int)*p))p++;
    if(*p != ')') { 
       sprintf(errmes, "Bad value of TFORM%d: %s.",colnum,tform[colnum-1]);
       wrterr(out,errmes,1);
    } 
    return;
}

/*************************************************************
*
*      print_title 
*
*  Print the title of the HDU. 
*  when verbose < 2, called by wrterr and wrtwrn.
*	
*************************************************************/
void print_title(FILE* out, int hdunum, int hdutype)
{
    static char hdutitle[64];
    static int oldhdu = 0;

    /* print out the title */
    curhdu = hdunum;
    curtype = hdutype;

    if(oldhdu == curhdu) return; /* Do not print it twice */
    if(curhdu == 1){
	        sprintf(hdutitle," HDU %d: Primary Array ", curhdu);
    }
    else { 
        switch (curtype) { 
	    case IMAGE_HDU:  
		sprintf(hdutitle," HDU %d: Image Exten. ", curhdu);
                break;
	    case ASCII_TBL:  
		sprintf(hdutitle," HDU %d: ASCII Table ", curhdu);
                break;
	    case BINARY_TBL:  
		sprintf(hdutitle," HDU %d: BINARY Table ", curhdu);
                break;
            default: 
		sprintf(hdutitle," HDU %d: Unknown Ext. ", curhdu);
                break;
        }
    } 
    wrtsep(out,'=',hdutitle,60);
    wrtout(out," ");
    oldhdu = curhdu;
    if(curhdu == totalhdu) oldhdu = 0;  /* reset the old hdu at the last hdu */
    return; 
}

/*************************************************************
*
*      print_header 
*
*  Print the header of the HDU. 
*	
*************************************************************/
void print_header(FILE* out)
{ 
    char htemp[100];
    int i;
    for (i=1; i <= ncards; i++)  { 
        sprintf(htemp,"%4d | %s",i,cards[i-1]);    	
	wrtout(out, htemp);
    }
    wrtout(out," ");
    return; 
}  

/*************************************************************
*
*      print_summary 
*
*  Print out the summary of this hdu.
*	
**************************************************************/
void print_summary(fitsfile *infits, 	/* input fits file   */ 
	     FILE*	out,		/* output ascii file */
             FitsHdu *hduptr 
            ) 
{ 
   
    int i = 0;
    char extver[10];
    char extnv[FLEN_VALUE];
    long npix;
    int hdutype;

    /* get the error number and wrn number */ 
    set_hduerr(hduptr->hdunum);

    hdutype = hduptr->hdutype;
    sprintf(comm," %d header keywords", hduptr->nkeys);
    wrtout(out,comm);
    wrtout(out," ");
    if(hdutype == ASCII_TBL || hdutype== BINARY_TBL) {  
        sprintf(extnv, "%s",hduptr->extname);
        if (hduptr->extver!=-999) {
            sprintf(extver,"(%d)",hduptr->extver);
            strcat(extnv,extver);
        }

#if (USE_LL_SUFFIX == 1)
        sprintf(comm," %s  (%d columns x %lld rows)", extnv, hduptr->ncols,
           hduptr->naxes[1]);
#else
        sprintf(comm," %s  (%d columns x %ld rows)", extnv, hduptr->ncols,
           hduptr->naxes[1]);
#endif
        wrtout(out,comm);
        if(hduptr->ncols) {
           wrtout(out," ");
           sprintf(comm, " Col# Name (Units)       Format"); 
           wrtout(out,comm);
	}    
        for ( i = 0; i < hduptr->ncols; i++) {
           if(strlen(tunit[i]))  
               sprintf(extnv,"%s (%s)",ttype[i],tunit[i]); 
           else 
               sprintf(extnv,"%s",ttype[i]); 
 	   sprintf(comm," %3d %-20.20s %-10.10s",
              i+1, extnv, tform[i]);
	   wrtout(out,comm);
        }  
    }      
    else if(hdutype == IMAGE_HDU && hduptr->isgroup) { 

            sprintf(comm, " %d Random Groups, ",hduptr->gcount);

            switch(hduptr->bitpix) { 
            	case BYTE_IMG: 
               	    strcpy(temp," 8-bit integer pixels, ");
               	    break; 
                case SHORT_IMG: 
                   strcpy(temp," 16-bit integer pixels, ");
                   break; 
                case USHORT_IMG: 
                   strcpy(temp," 16-bit unsigned integer pixels, ");
                   break; 
                case LONG_IMG: 
                   strcpy(temp," 32-bit integer pixels, ");
                   break; 
                case LONGLONG_IMG: 
                   strcpy(temp," 64-bit long integer pixels, ");
                   break; 
                case ULONG_IMG: 
                   strcpy(temp," 32-bit unsigned integer pixels, ");
                   break; 
                case FLOAT_IMG: 
                   strcpy(temp," 32-bit floating point pixels, ");
                   break; 
                case DOUBLE_IMG: 
                   strcpy(temp," 64-bit double precision pixels, ");
                   break; 
                default: 
                   strcpy(temp," unknown datatype, ");
                   break; 
            }
	    strcat(comm,temp);

            sprintf(temp," %d axes ",hduptr->naxis);
	    strcat(comm,temp);

#if (USE_LL_SUFFIX == 1)
	    sprintf(temp, "(%lld",hduptr->naxes[0]);
#else
	    sprintf(temp, "(%ld",hduptr->naxes[0]);
#endif
	    strcat(comm,temp);

	    npix = hduptr->naxes[0];
	    for ( i = 1; i < hduptr->naxis; i++){ 
	       npix *= hduptr->naxes[i];
#if (USE_LL_SUFFIX == 1)
	       sprintf(temp, " x %lld",hduptr->naxes[i]);
#else
	       sprintf(temp, " x %ld",hduptr->naxes[i]);
#endif
	       strcat(comm,temp);
            }  
	    strcat(comm,"), ");
            wrtout(out,comm);
    }
    else if(hdutype == IMAGE_HDU) { 
        if(hduptr->naxis > 0) {

	    if(hduptr->hdunum == 1) {  
                strcpy(extnv,"");
            } else { 
                sprintf(extnv, "%s",hduptr->extname);
                if (hduptr->extver!=-999) {
                   sprintf(extver," (%d)",hduptr->extver);
                   strcat(extnv,extver);
                }
            }
	    strcpy(comm,extnv);

            switch(hduptr->bitpix) { 
            	case BYTE_IMG: 
               	    strcpy(temp," 8-bit integer pixels, ");
               	    break; 
                case SHORT_IMG: 
                   strcpy(temp," 16-bit integer pixels, ");
                   break; 
                case USHORT_IMG: 
                   strcpy(temp," 16-bit unsigned integer pixels, ");
                   break; 
                case LONG_IMG: 
                   strcpy(temp," 32-bit integer pixels, ");
                   break; 
                case LONGLONG_IMG: 
                   strcpy(temp," 64-bit long integer pixels, ");
                   break; 
                case ULONG_IMG: 
                   strcpy(temp," 32-bit unsigned integer pixels, ");
                   break; 
                case FLOAT_IMG: 
                   strcpy(temp," 32-bit floating point pixels, ");
                   break; 
                case DOUBLE_IMG: 
                   strcpy(temp," 64-bit double precision pixels, ");
                   break; 
                default: 
                   strcpy(temp," unknown datatype, ");
                   break; 
            }
	    strcat(comm,temp);

            sprintf(temp," %d axes ",hduptr->naxis);
	    strcat(comm,temp);

#if (USE_LL_SUFFIX == 1)
	    sprintf(temp, "(%lld",hduptr->naxes[0]);
#else
	    sprintf(temp, "(%ld",hduptr->naxes[0]);
#endif
	    strcat(comm,temp);

	    npix = hduptr->naxes[0];
	    for ( i = 1; i < hduptr->naxis; i++){ 
	       npix *= hduptr->naxes[i];
#if (USE_LL_SUFFIX == 1)
	       sprintf(temp, " x %lld",hduptr->naxes[i]);
#else
	       sprintf(temp, " x %ld",hduptr->naxes[i]);
#endif
	       strcat(comm,temp);
            }  
	    strcat(comm,"), ");
            wrtout(out,comm);
        }
        else{ 
            sprintf(comm," Null data array; NAXIS = 0 ");
            wrtout(out,comm);
        }       
    }
    wrtout(out," ");
    return;
} 

/*************************************************************
*
*      close_hdu 
*
*  Free the memory allocated to the FitsHdu structure and 
*  other temporary  spaces.
*	
**************************************************************/
void close_hdu( FitsHdu *hduptr ) 
{    
    int i;
    int n;
    /* free  memories */ 
    for (i=0; i <  ncards; i++)  free(cards[i]);

    n = hduptr->nkeys - 4 - hduptr->naxis ;   /* excluding the SIMPLE, 
						 BITPIX, NAXIS, NAXISn  
						 and END */ 
    for (i=0; i <  n; i++)  free(hduptr->kwds[i]);

    for (i=0; i <  hduptr->ncols; i++) { 
	free(hduptr->datamin[i]);
	free(hduptr->datamax[i]);
	free(hduptr->tnull[i]);
    }
    if(hduptr->hdutype == ASCII_TBL && hduptr->hdutype == BINARY_TBL){
	if(hduptr->ncols > 0)free(ttype);
	if(hduptr->ncols > 0)free(tunit);
	if(hduptr->ncols > 0)free(tform);
    }
    if(hduptr->naxis) free(hduptr->naxes);
    if(hduptr->ncols > 0)free(hduptr->datamax);
    if(hduptr->ncols > 0)free(hduptr->datamin);
    if(hduptr->ncols > 0)free(hduptr->tnull);
    free(hduptr->kwds);
    free(cards);
    free(tmpkwds);
    return;
}
