/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "boilerplate.h"
#include "gsl/gsl_vector.h"
#include "gsl/gsl_multiroots.h"
#include "anwcs.h"
#include "errors.h"
#include "log.h"

const char* OPTIONS = "vh1:2:";
static int loglvl = LOG_MSG;

static anwcs_t* wcs1 = NULL;
static anwcs_t* wcs2 = NULL;

void print_help(char* progname) {
    BOILERPLATE_HELP_HEADER(stdout);
    printf("\nUsage: %s\n"
           "   -1 <WCS first input file>\n"
           "   -2 <WCS second input file>\n"
           "   -h\n"
           "  [-v]: +verbose\n"
           "\nPrints x and y coordinates of invariant point"
           "\n", progname);
}

int fvec(const gsl_vector *x, void *params, gsl_vector *f)
{
    double ra,dec,xp,yp;
    double xi = gsl_vector_get(x,0);
    double yi = gsl_vector_get(x,1);
    anwcs_pixelxy2radec(wcs1, xi, yi, &ra, &dec);
    anwcs_radec2pixelxy(wcs2, ra, dec, &xp, &yp);
    xp = xp - xi;
    yp = yp - yi;
    gsl_vector_set(f,0,xp);
    gsl_vector_set(f,1,yp);
    return GSL_SUCCESS;
}
int print_state (size_t iter, gsl_multiroot_fsolver * s)
{
    if (loglvl > LOG_MSG) {
        if (iter==0) {
            fprintf(stderr,"Solving...\n");
            fprintf(stderr,"Iteration           X                Y\n");
            fprintf(stderr,"%8ld: %16.6f %16.6f\n",(long)iter,
                    gsl_vector_get(s->x,0),
                    gsl_vector_get(s->x,1));
        } else fprintf(stderr,"%8ld: %16.6f %16.6f\n",(long)iter,
                       gsl_vector_get(s->x,0),
                       gsl_vector_get(s->x,1));
    }
    return 1;
}

int main(int argc, char** args) {
    int ext = 0,c;
    double ra,dec;
    double sol[2];
    const gsl_multiroot_fsolver_type *T;
    gsl_multiroot_fsolver *s;
    int status;
    size_t iter=0;
    const size_t n=2;
    gsl_multiroot_function f={&fvec,n,NULL};
    gsl_vector *x = gsl_vector_alloc(n);
    char *wcsfn1=NULL, *wcsfn2=NULL;
  
    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch(c) {
        case 'v':
            loglvl++;
            break;
        case 'h':
            print_help(args[0]);
            exit(0);
        case '1':
            wcsfn1 = optarg;
            break;
        case '2':
            wcsfn2 = optarg;
            break;
        }
    }
    log_init(loglvl);
    if (optind != argc) {
        print_help(args[0]);
        exit(-1);
    }
    if (!(wcsfn1) || !(wcsfn2)) {
        print_help(args[0]);
        exit(-1);
    }
    /* open the two wcs systems */
    wcs1 = anwcs_open(wcsfn1, ext);
    if (!wcs1) {
        ERROR("Failed to read WCS file");
        exit(-1);
    }
    logverb("Read WCS:\n");
    if (log_get_level() >= LOG_VERB) {
        anwcs_print(wcs1, log_get_fid());
    }
    wcs2 = anwcs_open(wcsfn2, ext);
    if (!wcs2) {
        ERROR("Failed to read WCS file");
        exit(-1);
    }
    logverb("Read WCS:\n");
    if (log_get_level() >= LOG_VERB) {
        anwcs_print(wcs2, log_get_fid());
    }
  
    /* setup the solver, start in the middle */

    gsl_vector_set(x,0,anwcs_imagew(wcs1)/2.0);
    gsl_vector_set(x,1,anwcs_imageh(wcs1)/2.0);
    T = gsl_multiroot_fsolver_hybrids;
    s = gsl_multiroot_fsolver_alloc (T,2);
    gsl_multiroot_fsolver_set(s,&f,x);
    print_state(iter,s);
    do {
        iter++;
        status = gsl_multiroot_fsolver_iterate(s);
        print_state(iter,s);
        if (status) break;
        status = gsl_multiroot_test_residual(s->f,1e-7);
    } while (status == GSL_CONTINUE && iter < 1000);
    sol[0]=gsl_vector_get(s->x,0);
    sol[1]=gsl_vector_get(s->x,1);


    /* write some diagnostics on stderr */
    /* transform to ra/dec */
    anwcs_pixelxy2radec(wcs1, sol[0], sol[1], &ra, &dec);
    if (loglvl > LOG_MSG)
        fprintf(stderr,"Pixel (%.10f, %.10f) -> RA,Dec (%.10f, %.10f)\n", 
                sol[0], sol[1], ra, dec);
    /* transform to x/y with second wcs 
     center of rotation should stay the same x/y */
    anwcs_radec2pixelxy(wcs2, ra, dec, &sol[0], &sol[1]);
    if (loglvl > LOG_MSG)
        fprintf(stderr,"RA,Dec (%.10f, %.10f) -> Pixel (%.10f, %.10f) \n", 
                ra, dec, sol[0], sol[1]);

    /* write the solution */
    fprintf(stdout,"%f\n",sol[0]); 
    fprintf(stdout,"%f\n",sol[1]);
  
    return(0);
}
