/*
  This file is part of "fitsverify" and was imported from:
    http://heasarc.gsfc.nasa.gov/docs/software/ftools/fitsverify/
 */
/******************************************************************************
* Function
*      wrtout: print messages in the streams of stdout and out.  
*      wrterr: print erro messages  in the streams of stderr and out.  
*      wrtferr: print cfitsio erro messages in the streams of stderr and out.  
*      wrtwrn: print warning messages in the streams of stdout and out.  
*      wrtsep: print seperators.  
*      num_err_wrn: Return the number of errors and  warnings.
*
*******************************************************************************/
#include "fverify.h"
static int nwrns = 0;
static int nerrs = 0;
static char temp[512];

void num_err_wrn(int *num_err, int *num_wrn) 
{ 
    *num_wrn = nwrns;
    *num_err = nerrs;
    return;
}

void reset_err_wrn() 
{ 
    nwrns = 0; 
    nerrs = 0;
    return; 
}

int wrtout(FILE *out, char *mess)
{
    if(out != NULL )fprintf(out,"%s\n",mess);
    if(out == stdout) fflush(stdout);
    return 0;
}

int wrtwrn(FILE *out, char *mess, int isheasarc)
{
    if(err_report) return 0;           /* Don't print the warnings */    
    if(!heasarc_conv && isheasarc) return 0;  /* heasarc warnings  but with
                                                 heasarc convention turns off */
    nwrns++;
    strcpy(temp,"*** Warning: ");
    strcat(temp,mess);
    if(isheasarc) strcat(temp," (HEASARC Convention)");
    print_fmt(out,temp,13);
/*    if(nwrns > MAXWRNS ) { 
	 fprintf(stderr,"??? Too many Warnings! I give up...\n");
          
    }  */
    return nwrns;
}

int wrterr(FILE *out, char *mess, int severity )
{

    if(severity < err_report) { 
        fits_clear_errmsg();
        return 0; 
    }
    nerrs++;

    strcpy(temp,"*** Error:   ");
    strcat(temp,mess);
    if(out != NULL) {
         if ((out!=stdout) && (out!=stderr)) print_fmt(out,temp,13);
/*
   if ERR2OUT is defined, then error messages will be sent to the
   stdout stream rather than to stderr
*/
#ifdef ERR2OUT
         print_fmt(stdout,temp,13);
#else
         print_fmt(stderr,temp,13);
#endif
    }

    if(nerrs > MAXERRORS ) { 

#ifdef ERR2OUT
	 fprintf(stdout,"??? Too many Errors! I give up...\n");
#else
	 fprintf(stderr,"??? Too many Errors! I give up...\n");
#endif
         close_report(out);
         exit(1);
    }
    fits_clear_errmsg();
    return nerrs;
}

int wrtferr(FILE *out, char* mess, int *status, int severity)
/* construct an error message: mess + cfitsio error */
{
    char ttemp[255];

    if(severity < err_report) { 
        fits_clear_errmsg();
        return 0; 
    }
    nerrs++;

    strcpy(temp,"*** Error:   ");
    strcat(temp,mess);
    fits_get_errstatus(*status, ttemp);
    strcat(temp,ttemp);
    if(out != NULL ) {
        if ((out!=stdout) && (out!=stderr)) print_fmt(out,temp,13);
/*
   if ERR2OUT is defined, then error messages will be sent to the
   stdout stream rather than to stderr
*/
#ifdef ERR2OUT
         print_fmt(stdout,temp,13);
#else
         print_fmt(stderr,temp,13);
#endif
    }

    *status = 0;
    fits_clear_errmsg();
    if(nerrs > MAXERRORS ) { 
#ifdef ERR2OUT
	 fprintf(stdout,"??? Too many Errors! I give up...\n");
#else
	 fprintf(stderr,"??? Too many Errors! I give up...\n");
#endif
         close_report(out);
         exit(1);
    }
    return nerrs;
} 

int wrtserr(FILE *out, char* mess, int *status, int severity)
/* dump the cfitsio stack */
{
    char* errfmt = "             %.67s\n";
    int i;
    char tmp[20][80];
    int nstack = 0;

    if(severity < err_report) { 
        fits_clear_errmsg();
        return 0; 
    }
    nerrs++;

    strcpy(temp,"*** Error:   ");
    strcat(temp,mess);
    strcat(temp,"(from CFITSIO error stack:)");
    while(nstack < 20) {
        tmp[nstack][0] = '\0';
        i = fits_read_errmsg(tmp[nstack]);
        if(!i && tmp[nstack][0]=='\0') break;
        nstack++;
    }

    if(out !=NULL) {
        if ((out!=stdout) && (out!=stderr)) { 
           print_fmt(out,temp,13);
           for(i=0; i<=nstack; i++) fprintf(out,errfmt,tmp[i]);
         }

#ifdef ERR2OUT
           print_fmt(stdout,temp,13);
           for(i=0; i<=nstack; i++) fprintf(stdout,errfmt,tmp[i]);
#else
           print_fmt(stderr,temp,13);
           for(i=0; i<=nstack; i++) fprintf(stderr,errfmt,tmp[i]);
#endif
    }

    *status = 0;
    fits_clear_errmsg();
    if(nerrs > MAXERRORS ) { 
#ifdef ERR2OUT
	 fprintf(stdout,"??? Too many Errors! I give up...\n");
#else
	 fprintf(stderr,"??? Too many Errors! I give up...\n");
#endif
         close_report(out);
         exit(1);
    }
    return nerrs;
}

void print_fmt(FILE *out, char *temp, int nprompt)
/* Print output of messages in a 80 character record.  
    Continue lines are aligned. */ 
{ 
     
    char *p;
    int i,j;  
    int clen;
    char tmp[81]; 
    static char cont_fmt[80];
    static int save_nprompt = 0;

    if (out == NULL) return;

    if(nprompt != save_nprompt) { 
        for (i = 0; i < nprompt; i++) cont_fmt[i] = ' ';
        strcat(cont_fmt,"%.67s\n"); 
        save_nprompt = nprompt;
    }

    i = strlen(temp) - 80;
    if(i <= 0) {  
        fprintf(out,"%.80s\n",temp); 
    } 
    else{ 
        p = temp; 
        clen = 80 -nprompt;
        strncpy(tmp,p,80); 
        tmp[80] = '\0';
        if(isprint((int)*(p+79)) && isprint((int)*(p+80)) && *(p+80) != '\0') { 
           j = 79; 
           while(*(p+j) != ' ' && j > 0) j--; 
           p += j;
           while( *p == ' ')p++;
           tmp[j] = '\0';  
        } 
        else if( *(p+80) == ' ') { 
             j = 80; 
             while( *(p+j) == ' ') j++; 
             p +=  j; 
        } 
        else {
             p += 80;
        }
        fprintf(out,"%.80s\n",tmp); 
        while(*p != '\0' && i > 0) { 
            strncpy(tmp,p,clen); 
            tmp[clen] = '\0';
            i = strlen(p)- clen;
            if(i > 0 && isprint((int)*(p+clen-1)) 
                     && isprint((int)*(p+clen)) 
                     && *(p+clen) != '\0') {  
                j = clen; 
                while(*(p+j)!= ' ' && j > 0) j--; 
                p += j;
                while( *p == ' ')p++;
                tmp[j] = '\0';  
            }
            else if(i> 0 &&  *(p+clen) == ' ') { 
                 j = clen; 
                 while( *(p+j) == ' ') j++; 
                 p += j; 
            }
            else if(i> 0)  {
                 p+= clen;
            }
            fprintf(out,cont_fmt,tmp);
        } 
    } 
    if(out==stdout) fflush(stdout);
    return;
}
void wrtsep(FILE *out,char fill, char *title, int nchar)
/* print a line of char fill with string title in the middle */
{
    int ntitle; 
    char *line; 
    char *p;
    int first_end;
    int i = 0;

    ntitle = strlen(title); 
    if(ntitle > nchar) nchar = ntitle;
    if(nchar <= 0) return; 
    line = (char *)malloc((nchar+1)*sizeof(char)); 
    p = line;
    if(ntitle < 1) { 
        for (i=0; i < nchar; i++) {*p = fill; p++;}	
	*p = '\0';
    }
    else { 
	first_end = ( nchar - ntitle)/2; 
	for (i = 0; i < first_end; i++) { *p = fill; p++;}
	*p = '\0';
	strcat(line, title);
	p += ntitle;
        for( i = first_end + ntitle; i < nchar; i++) {*p = fill; p++;}
	*p = '\0';
    }
    if(out != NULL )fprintf(out,"%s\n",line);
    if(out == stdout )fflush(out);
    free (line);
    return ;
}
	

/* comparison function for the FitsKey structure array */
   int compkey (const void *key1, const void *key2)
{
       char *name1;
       char *name2;
       name1 = (*(FitsKey **)key1)->kname;
       name2 = (*(FitsKey **)key2)->kname;
       return strncmp(name1,name2,FLEN_KEYWORD);
}

/* comparison function for the colname structure array */
   int compcol (const void *col1, const void *col2)
{
       char *name1;
       char *name2;
       name1 = (*(ColName **)col1)->name;
       name2 = (*(ColName **)col2)->name;
       return strcmp(name1,name2);
}
/* comparison function for the string pattern maching*/
   int compstrp (const void *str1, const void *str2)
{
   char *p;
   char *q;
   p = (char *)(*(char**) str1);
   q = (char *)(*(char**) str2);
   while( *q == *p && *q != '\0') {
       p++;
       q++;
       if(*p == '\0') return 0;   /* str2 is longer than str1, but
                                 matched */
   }
   return (*p - *q);
}

/* comparison function for the string exact maching*/
   int compstre (const void *str1, const void *str2)
{
   char *p;
   char *q;
   p = (char *)(*(char**) str1);
   q = (char *)(*(char**) str2);
   return strcmp( p, q);
}

