/*
  This file is part of "fitsverify" and was imported from:
    http://heasarc.gsfc.nasa.gov/docs/software/ftools/fitsverify/
 */
#include "fverify.h"
typedef struct { 
   int nnum;
   int ncmp;
   int *indatatyp;
   double *datamax;
   double *datamin;
   double *tnull;
   unsigned char *mask;      /* for bit X column only */
   int ntxt;
   FitsHdu *hduptr;
   FILE* out;
}UserIter;

/*************************************************************
*
*      test_data 
*
*   Test the HDU data  
*
*  This routine reads every row and column of ASCII tables to
*  verify that the values have the correct format.
*
*  This routine checks the following types of columns in binary tables:
*
*    Logical L - value must be T, F or zero
*    Bit    nX - if n != a multiple of 8, then check that fill bits = 0
*    String  A - must contain ascii text, or zero
*
*   It is impossible to write an invalid value to the other types of
*   columns in binary tables (B, I, J, K, E, D, C and M) so these
*   columns are not read.
*
*  Since it is impossible to write an invalid value in a FITS image,
*  this routine does not read the image pixels.
*	
*************************************************************/
void test_data(fitsfile *infits, 	/* input fits file   */ 
	      FILE	*out,		/* output ascii file */
	      FitsHdu    *hduptr	/* fits hdu pointer  */
            )

{  
    iteratorCol *iter_col=0;
    int  ncols; 

    int nnum = 0;
    int *numlist;	/* the list of the column  whose data
                           type is numerical(scalar and complex) */
    int ncmp = 0;
    int *cmplist;	/* the list of the column  whose data
                           type is numerical(scalar and complex) */
    int ntxt = 0;		 
    int *txtlist;	/* the list of column  whose data type is
			   string, logical, bit or complex */
    int niter = 0;	/* total columns read into  the literator function */

    int ndesc = 0;		 
    int *desclist;	/* the list of column which is the descriptor of
			   the variable length array. */

    long rows_per_loop = 0, offset;
    UserIter usrdata;

    int datatype;
    long repeat; 

    long totalrows;
    long heap_offset;
    long length;
    long toffset;
    long *maxlen;
    int icol;
    char *cdata;
    double *ndata;
    int *idata;
    int *maxminflag;
    int *dflag; 
    int nread;
    char lnull = 2;
    int anynul;

    long nelem;
    long rlength;
    long bytelength;
    long maxmax;

    int i = 0;
    int j = 0;
    long jl = 0;
    long k = 0;
    int status = 0;
    char errtmp[80];
    int*  perbyte;

    int find_badchar = 0; 
    int find_badlog = 0; 

    if(testcsum)
        test_checksum(infits,out); 

    if(testfill) { 
        test_agap(infits,out,hduptr);     /* test the bytes between the
                                                   ascii table columns. */
        if(ffcdfl(infits, &status)) { 
            wrtferr(out,"checking data fill: ", &status, 1);
            status = 0;
        }
    }

    if(hduptr->hdutype != ASCII_TBL &&  
       hduptr->hdutype != BINARY_TBL ) return;

    ncols = hduptr->ncols;
    if(ncols <= 0) return;

    /* separate the numerical, complex, text and 
      the variable length vector columns */
    numlist =(int*)malloc(ncols * sizeof(int)); 
    cmplist =(int*)malloc(ncols * sizeof(int)); 
    txtlist =(int*)malloc(ncols * sizeof(int)); 
    desclist =(int*)malloc(ncols * sizeof(int)); 

    if(hduptr->hdutype == ASCII_TBL) {

        /*read every column of an ASCII table */
	rows_per_loop = 0;
        for (i=0; i< ncols; i++){ 
            if(fits_get_coltype(infits, i+1, &datatype, NULL, NULL, &status)){ 
               sprintf(errmes,"Column #%d: ",i);
 	       wrtferr(out,errmes, &status,2);
            }
            if ( datatype != TSTRING ) { 
	           numlist[nnum] = i+1; 
	           nnum++;

            } else { 
 	       txtlist[ntxt] = i+1;
	       ntxt++;
            }
        }

    } else if (hduptr->hdutype == BINARY_TBL) { 

        /* only check Bit, Logical and String columns in Binary tables */
	rows_per_loop = 0;
        for (i=0; i< ncols; i++){ 
            if(fits_get_coltype(infits, i+1, &datatype, &repeat, NULL, 
               &status)){ 
               sprintf(errmes,"Column #%d: ",i);
 	       wrtferr(out,errmes, &status,2);
            }

	    if(datatype < 0) {    /* variable length column */
	       desclist[ndesc] = i+1;
	       ndesc++;

            } else if(datatype == TBIT && (repeat%8) ) 
                {  /* bit column that does not have a multiple of 8 bits */
	           numlist[nnum] = i+1; 
	           nnum++;

            } else if( (datatype == TLOGICAL) ||
                       (datatype == TSTRING ) )  {
	           txtlist[ntxt] = i+1; 
	           ntxt++;
            }
            /* ignore all other types of columns (B I J K E D C and M ) */
        } 
    }


    /*  Use Iterator to read the columns that are not variable length arrays */ 
    /* columns from  1 to nnum are scalar numerical columns. 
       columns from  nnum+1 to  nnum+ncmp are complex columns.
       columns from  nnum+ncmp are text columns */
    niter = nnum + ncmp + ntxt;

    if(niter)iter_col = (iteratorCol *) malloc (sizeof(iteratorCol)*niter);

    for (i=0; i< nnum; i++){  
	fits_iter_set_by_num(&iter_col[i], infits, numlist[i], TDOUBLE, 
	   InputCol);
    }
    for (i=0; i< ncmp; i++){ 
	j = nnum + i;
	fits_iter_set_by_num(&iter_col[j], infits, cmplist[i], TDBLCOMPLEX, 
	   InputCol);
    }	  
    for (i=0; i< ntxt; i++){ 
	j = nnum + ncmp + i;
	fits_iter_set_by_num(&iter_col[j], infits, txtlist[i], 0, 
	   InputCol);
    }	  


    offset = 0;
    usrdata.nnum = nnum;
    usrdata.ncmp = ncmp;
    if (nnum > 0 || ncmp > 0) {
        usrdata.datamax  = (double *)calloc((nnum+ncmp), sizeof(double));
        usrdata.datamin  = (double *)calloc((nnum+ncmp), sizeof(double));
    }
    usrdata.tnull = (double *)calloc(ncols, sizeof(double));
    usrdata.ntxt = ntxt;
    usrdata.hduptr = hduptr;
    usrdata.out = out;
   
    /* get the mask for the bit X column  
        for column other than the X, it always 255 
        for Column nX, it will be 000...111, where # of 0 is n%8, 
        # of 1 is 8 - n%8. 
    */

    if(nnum > 0) usrdata.mask = 
            (unsigned char *)calloc(nnum,sizeof( unsigned char));
    if(nnum > 0) usrdata.indatatyp = 
            (int *)calloc(nnum,sizeof( int));
    for (i=0; i< nnum; i++){
        j = fits_iter_get_colnum(&(iter_col[i]));  
        if(fits_get_coltype(infits, j, &datatype, &repeat, NULL, &status)){ 
           sprintf(errmes,"Column #%d: ",i);
 	   wrtferr(out,errmes, &status,2);
        } 
        usrdata.indatatyp[i] = datatype;
        usrdata.mask[i] = 255;
        if(datatype == TBIT) { 
            repeat = repeat%8;
            usrdata.mask[i] = (usrdata.mask[i])>>repeat;
            if(!repeat) usrdata.mask[i] = 0;
        }
    }       
     

    if(niter > 0) {  
	if(fits_iterate_data(niter, iter_col, offset,rows_per_loop, iterdata, 
            &usrdata,&status)){ 
            wrtserr(out,"When Reading data, ",&status,2);
        }
    }
      
    if(niter>0) free(iter_col);
    free(numlist);
    free(cmplist);
    free(txtlist);
    if(nnum > 0) free(usrdata.mask);
    if(nnum > 0) free(usrdata.indatatyp);
    if(nnum > 0 || ncmp > 0) {
          free(usrdata.datamax); 
          free(usrdata.datamin);
    }
    free(usrdata.tnull);
    if(!ndesc ) { 
	goto data_end; 
    } 

    /* ------------read the variable length vectors -------------------*/ 
    usrdata.datamax  = (double *)calloc(ndesc, sizeof(double));
    usrdata.datamin  = (double *)calloc(ndesc, sizeof(double));
    usrdata.tnull  = (double *)calloc(ndesc, sizeof(double));
    maxminflag     = (int *) calloc(ndesc , sizeof(int));
    maxlen         = (long *) calloc(ndesc , sizeof(long));
    dflag          = (int *) calloc(ndesc , sizeof(int));
    perbyte        = (int *) calloc(ndesc , sizeof(int));
    fits_get_num_rows(infits,&totalrows,&status);
    status = 0;
    heap_offset = hduptr->heap - hduptr->naxes[0] * hduptr->naxes[1];

  /* this routine now only reads and test BIT, LOGICAL, and STRING columns */
  /* There is no point in reading the other columns because the other datatypes */
  /* have no possible invalid values.  */

    for (i = 0; i < ndesc; i++) { 
        icol = desclist[i]; 
        parse_vtform(infits,out,hduptr,icol,&datatype,&maxlen[i]);
	dflag[i] = 4; 
        switch (datatype) { 
          case -TBIT:
              dflag[i] = 1;
              perbyte[i] = -8;
              break; 
          case -TBYTE:
              perbyte[i] = 1;
              break; 
          case -TLOGICAL:
              dflag[i] = 3;
              perbyte[i] = 1;
              break; 
          case -TSTRING:
              dflag[i] = 0;
              perbyte[i] = 1;
              break; 
          case -TSHORT: 
              perbyte[i] = 2;
              break; 
          case -TLONG: 
              perbyte[i] = 4;
              break; 
          case -TFLOAT: 
              perbyte[i] = 4;
              break; 
          case -TDOUBLE: 
              perbyte[i] = 8;
              break; 
          case -TCOMPLEX: 
              dflag[i] = 2;
              perbyte[i] = 8;
              break; 
          case -TDBLCOMPLEX: 
              dflag[i] = 2;
              perbyte[i] = 16;
              break; 
          default:
              break;
        }
    }

    maxmax = maxlen[0]; 
    for (i = 1; i < ndesc; i++) { 
	if(maxmax < maxlen[i]) maxmax = maxlen[i];
    } 
    if(maxmax < 0) maxmax = 100;
    ndata = (double *)malloc(2*maxmax*sizeof(double)); 
    cdata = (char *)malloc((maxmax+1) *sizeof(char));
    idata = (int *)malloc(maxmax *sizeof(int));

	    
    for (jl = 1; jl <= totalrows; jl++) { 
        for (i = 0; i < ndesc; i++) { 
            icol = desclist[i]; 

            /* read and check the descriptor length and offset values */
            if(fits_read_descript(infits, icol ,jl,&length,
		   &toffset, &status)){ 
                
                sprintf(errtmp,"Row #%ld Col.#%d: ",jl,icol);
	        wrtferr(out,errtmp,&status,2);
            } 
	    if(length > maxlen[i] && maxlen[i] > -1 ) { 
	        sprintf(errmes, "Descriptor of Column #%d at Row %ld: ", 
                     icol, jl);
                sprintf(errtmp,"nelem(%ld) > maxlen(%ld) given by TFORM%d.",
                    length,maxlen[i],icol);
                strcat(errmes,errtmp);
                wrterr(out,errmes,1); 
            } 

            if( perbyte[i] < 0)  
                 bytelength = length/8; 
            else 
                 bytelength = length*perbyte[i]; 

            if(heap_offset + toffset + bytelength > hduptr->pcount ) { 
	        sprintf(errmes, "Descriptor of Column #%d at Row %ld: ", 
                     icol, jl);
	        sprintf(errtmp, 
                    " offset of first element(%ld) + nelem(%ld)", 
                     toffset,length); 
                strcat(errmes,errtmp);
                if(perbyte[i] < 0) 
	            sprintf(errtmp, "/8 >  total heap area  = %ld.",
		       hduptr->pcount-heap_offset); 
                else 
	            sprintf(errtmp, "*%d >  total heap area  = %ld.",
		       perbyte[i],hduptr->pcount-heap_offset); 
                strcat(errmes,errtmp);
                wrterr(out,errmes,2);
            }

            if(!length) continue;  /* skip the 0 length array */

            /* now check the values in BIT, LOGICAL, and String columns */
	    rlength = length;
	    if(length > maxmax) rlength = maxmax;
	    nread = 0;

            if(dflag[i] == 1) { /* read BIT column */
	        nelem = rlength*dflag[i];
		anynul = 0;

/*  NOT YET IMPLEMENTED:  This code should test that the fill bits that
    pad out the last byte are all zero.  Currently this test is applied
    to fixed length logical arrays, but has not yet been done for 
    the variable length logical array case.  It is probably safe to assume
    that not many FITS files will contain variable length Logical columns,
    to adding this test is not a high priority.

	        if(fits_read_col(infits, TDOUBLE, icol , jl, 1, 
	            nelem, &nullval, ndata, &anynul, &status)) { 
	       	    wrtferr(out,"",&status,2);
                }
*/
            }

            else if(dflag[i] == 0) { /* read String column */
	        if(fits_read_col(infits, TSTRING, icol, jl, 1, 
		    rlength, NULL, &cdata, &anynul, &status)) { 
                    sprintf(errtmp,"Row #%ld Col.#%d: ",jl,icol);
	            wrtferr(out,errtmp,&status,2);
                } 
                else {
                  j = 0;
                  while (cdata[j] != 0) {

                    if ((cdata[j] > 126) || (cdata[j] < 32) ) {
                      sprintf(errmes, 
                      "String in row #%ld, and column #%d contains non-ASCII text.", jl,icol); 
                      wrterr(out,errmes,1);
                        strcpy(errmes,
            "             (This error is reported only once; other rows may have errors).");
                      print_fmt(out,errmes,13);
                      find_badchar = 1;
                      break;
                    }
                    j++;
                  }
                }
            }
            else if(dflag[i] == 3) { /* read Logical column */
	        if(fits_read_col(infits, TLOGICAL, icol, jl, 1, 
		    rlength, &lnull, cdata, &anynul, &status)) { 
                    sprintf(errtmp,"Row #%ld Col.#%d: ",jl,icol);
	            wrtferr(out,errtmp,&status,2);
                }
                else {
		  for (k = 0; k < rlength; k++) {
                    if (cdata[k] > 2) {
                      sprintf(errmes, 
                      "Logical value in row #%ld, and column #%d has illegal value = %d", 
                         jl, icol, (int) cdata[k]); 
                       wrterr(out,errmes,1);
                       strcpy(errmes,
           "             (This error is reported only once; other rows may have errors).");
                       print_fmt(out,errmes,13);
                       find_badlog = 1;
                       break; 
                    }
                  }
                }
	    }
        }
    }
    free(ndata);
    free(cdata);
    free(idata);

    free(usrdata.datamax); 
    free(usrdata.datamin);
    free(usrdata.tnull);
    free(maxminflag);
    free(maxlen);
    free(dflag);
    free(perbyte);

data_end: 
    free(desclist);
    for ( i = 0; i< ncols; i++) {
	(hduptr->datamax[i])[12] = '\0';
	(hduptr->datamin[i])[12] = '\0';
	(hduptr->tnull[i])[11] = '\0';
    }

    return;
} 

/***********************************************************************/
/* iterator work function */

    int iterdata(long totaln, 
		 long offset, 
		 long firstn,
		 long nrows,
		 int narray,
		 iteratorCol *iter_col,
		 void *usrdata
		 )
{ 
    static UserIter *usrpt;
    static FitsHdu  *hdupt;
    static int nnum;
    static int ntxt;
    static int ncmp;
    static int *flag_minmax = 0;			/* define the initial min and max value */
    static long *repeat;
    static int *datatype;
    static int find_badbit = 0; 
    static int find_badchar = 0; 
    static int find_badlog = 0; 

    double  *data;
    unsigned char *ldata;
    char **cdata;
    unsigned char *ucdata;

    /* bit column working space */
    static unsigned char bdata;

    int i; 
    long j,k,l;
    long nelem;

    int flag_cmpnull = 0;

    if(firstn == 1 ) {  /* first time for this table, so initialize */
        usrpt = (UserIter *)usrdata;
	hdupt= usrpt->hduptr;
        nnum = usrpt->nnum;
        ncmp = usrpt->ncmp;
        ntxt = usrpt->ntxt;
	flag_minmax = (int *)calloc(nnum+ncmp, sizeof(int));
	repeat   = (long *)calloc(narray,sizeof(long));
	datatype = (int *)calloc(narray,sizeof(int));
        for (i=0; i < narray; i++) {  
	    repeat[i] = fits_iter_get_repeat(&(iter_col[i]));
	    datatype[i] = fits_iter_get_datatype(&(iter_col[i])); 
        }
        find_badbit = 0; 
        find_badchar = 0; 
        find_badlog = 0; 
    }

    /* columns from  1 to nnum are scalar numerical columns. 
       columns from  nnum+1 to  nnum+ncmp are complex columns. (not used any more)
       columns from  nnum+ncmp are text columns */

    /* deal with the numerical column */
    for (i=0; i < nnum+ncmp; i++) { 
	data = (double *) fits_iter_get_array(&(iter_col[i]));
	j = 1;
	flag_cmpnull = 0;
	nelem = nrows * repeat[i];
	if(i >= nnum) nelem = 2 * nrows *repeat[i];
	if(nelem == 0) continue;

        /* check for the bit jurisfication  */
        if(!find_badbit && usrpt->indatatyp[i] == TBIT ) { 
            for (k = 0; k < nrows; k++) {
               j = (k+1)*repeat[i];
               bdata = (unsigned char)data[j]; 
               if( bdata & usrpt->mask[i] ) { 
                  sprintf(errmes, 
                    "Row #%ld, and Column #%d: X vector ", firstn+k, 
                      fits_iter_get_colnum(&(iter_col[i]))); 
                  for (l = 1; l<= repeat[i]; l++) {
                     sprintf(comm, "0x%02x ", (unsigned char) data[k*repeat[i]+l]);
                     strcat(errmes,comm); 
                  }
                  strcat(errmes,"is not left justified."); 
                  wrterr(usrpt->out,errmes,2);
                  strcpy(errmes,
          "             (This error is reported only once; other rows may have errors).");
                  print_fmt(usrpt->out,errmes,13);
                  find_badbit = 1;
                  break;
               }
            }
        }  
    }

    /* deal with character and logical columns */
    for (i = nnum + ncmp; i < narray; i++) { 
        if(datatype[i] == TSTRING ) {	/* character */
            nelem = nrows;
	    if(nelem == 0) continue;
	    cdata = (char **) fits_iter_get_array(&(iter_col[i]));

            /* test for illegal ASCII text characters > 126  or < 32 */
            if (!find_badchar) {
              for (k = 0; k < nrows; k++) {
                ucdata = (unsigned char *)cdata[k+1];
                j = 0;
                while (ucdata[j] != 0) {

                  if ((ucdata[j] > 126) || (ucdata[j] < 32)) {
                    sprintf(errmes, 
                    "String in row #%ld, and column #%d contains non-ASCII text.", firstn+k, 
                      fits_iter_get_colnum(&(iter_col[i]))); 
                      wrterr(usrpt->out,errmes,1);
                      strcpy(errmes,
          "             (This error is reported only once; other rows may have errors).");
                      print_fmt(usrpt->out,errmes,13);
                    find_badchar = 1;
                    break;
                  }
                  j++;
                }
              }
            }
        }

	else {  			/* logical value */
            nelem = nrows * repeat[i];
	    if(nelem == 0) continue;
	    ldata = (unsigned char *) fits_iter_get_array(&(iter_col[i]));

            /* test for illegal logical column values */
            /* The first element in the array gives the value that is used to represent nulls */
            if (!find_badlog) {
                for(j = 1; j <= nrows * repeat[i]; j++) {
                  if (ldata[j] > 2) {
                    sprintf(errmes, 
                    "Logical value in row #%ld, and column #%d has illegal value = %d", 
                       (firstn+j - 2)/repeat[i] +1, 
                       fits_iter_get_colnum(&(iter_col[i])), (int) ldata[j]); 
                       wrterr(usrpt->out,errmes,1);
                       strcpy(errmes,
         "             (This error is reported only once; other rows may have errors).");
                       print_fmt(usrpt->out,errmes,13);
                       find_badlog = 1;
                       break; 
                  }
                }
            }
        }
    }

    if(firstn + nrows - 1 == totaln) { 
	free(flag_minmax);
	free(datatype);
	free(repeat); 
    }
    return 0; 
}

/*************************************************************
*
*      test_agap 
*
*   Test the bytes between the ASCII table column. 
*
*	
*************************************************************/
void test_agap(fitsfile *infits, 	/* input fits file   */ 
	      FILE	*out,		/* output ascii file */
	      FitsHdu    *hduptr	/* fits hdu pointer  */
            )
{ 
    int ncols;
    long nrows;
    long irows;
    long rowlen;
    unsigned char *data;
    int *temp;
    unsigned char *p;
    long i, j;
    int k, m, t;
    long firstrow = 1;
    long ntodo;
    long nerr = 0;
    int status = 0;
    char keyname[9];
    char tform[FLEN_VALUE], comment[256];
    int typecode, decimals;
    long width, tbcol;
    nerr = 0;

    if(hduptr->hdutype != ASCII_TBL) return;
    ncols = hduptr->ncols;
    fits_get_num_rows(infits,&nrows,&status); 
    status = 0; 

    fits_get_rowsize(infits, &irows, &status);
    status = 0; 
    rowlen = hduptr->naxes[0];  
    data = (unsigned char*)malloc(rowlen*sizeof(unsigned char)*irows);

    /* Create a template row with data fields filled with 1s.
       Used below - different ASCII rules apply within data columns
       vs. between data columns. */

    temp = (int*)malloc(rowlen * sizeof(int));
    for (m = 0; m<rowlen; m++ ) temp[m]=0;
    for (k = 1; k<=ncols; k++ ) { 
	sprintf(keyname, "TFORM%d",k);
	fits_read_key_str(infits, keyname, tform, comment, &status);
	if (fits_ascii_tform(tform, &typecode, &width, &decimals, &status))
	    wrtferr(out,"",&status,1);
	sprintf(keyname, "TBCOL%d",k);
	fits_read_key_lng(infits, keyname, &tbcol, comment, &status);
	for (t = tbcol; t < tbcol+width; t++) temp[t-1]=1;
    }

    i = nrows; 
    while( i > 0) { 
	if( i > irows)  
	    ntodo = irows; 
        else
	    ntodo = i; 
        
        p = data;
        if(fits_read_tblbytes(infits,firstrow,1, rowlen*ntodo, 
	    data, &status)){  
	    wrtferr(out,"",&status,1);
        } 
        for (j = 0; j<rowlen*ntodo; j++ ) { 
            if(!isascii(*p))  {
	        if(!nerr) { 
		     sprintf(errmes, 
			"row %ld contains non-ASCII characters.",
			j/rowlen+1); 
                     wrterr(out,errmes,1);
                } 
                nerr++;
            } else if(isascii(*p) && !isprint(*p))  {
	        if(temp[j%rowlen]) { 
	             if(!nerr) { 
		          sprintf(errmes, 
			     "row %ld data contains non-ASCII-text characters.",
			     j/rowlen+1); 
                          wrterr(out,errmes,1);
                     } 
                     nerr++;
                } 
            } 
	    p++;
        }
	firstrow += ntodo;
	i -=ntodo;
    } 
    if(nerr) { 
	sprintf(errmes,
	    "This ASCII table contains %ld non-ASCII-text characters",nerr);
        wrterr(out,errmes,1);
    }
    free(data);
    free(temp);
    return;
}
	    

/*************************************************************
*
*      test_checksum 
*
*   Test the checksum of the hdu 
*
*	
*************************************************************/
void test_checksum(fitsfile *infits, 	/* input fits file   */ 
	      FILE	*out		/* output ascii file */
            )
{ 
    int status = 0; 
    int dataok, hduok;

    if (fits_verify_chksum(infits, &dataok, &hduok, &status))
    {
        wrtferr(out,"verifying checksums: ",&status,2);
        return;
    }

    if(dataok == -1)  
	wrtwrn(out,
        "Data checksum is not consistent with  the DATASUM keyword",0);

    if(hduok == -1 )  { 
	if(dataok == 1) { 
	   wrtwrn(out,
  "Invalid CHECKSUM means header has been modified. (DATASUM is OK) ",0);
        } 
	else {
	   wrtwrn(out, "HDU checksum is not in agreement with CHECKSUM.",0);
        }
    }
    return;
}
