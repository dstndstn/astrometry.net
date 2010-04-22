/*
* lsqr.c
* This is a C version of LSQR derived by James W. Howse <jhowse@lanl.gov>
* from the Fortran 77 implementation of C. C. Paige and M. A. Saunders.
*
* This file defines functions for data type allocation and deallocation,
* lsqr itself (the main algorithm),
* and functions that scale, copy, and compute the Euclidean norm of a vector.
*
*
* The following history comments are maintained by Michael Saunders.
*
* 08 Sep 1999: First version of lsqr.c (this file) provided by
*              James W. Howse <jhowse@lanl.gov>.
*
* 16 Aug 2005: Forrester H Cole <fcole@Princeton.edu> reports "too many"
"*              iterations" when a good solution x0 is supplied in
*              input->sol_vec.  Note that the f77 and Matlab versions of LSQR
*              intentionally don't allow x0 to be input.  They require
*              the user to define r0 = b - A*x0 and solve A dx = r0,
*              then add x = x0 + dx.  Note that nothing is gained unless
*              larger stopping tolerances are set in the solve for dx
*              (because dx doesn't need to be computed accurately if
*              x0 is very good).  The same is true here.  BEWARE!
*
* 14 Feb 2006: Marc Grunberg <marc@renass.u-strasbg.fr> reports successful
*              use on a sparse system of size 1422805 x 246588 with
*              76993294 nonzero entries.  Also reports that the solution
*              vector isn't initialized(!).  Added
*                 output->sol_vec->elements[indx] = 0.0;
*              to the first for loop.
*
*              Presumably you are supposed to provide x0 even if it is zero.
*              Again, BEWARE!  I feel it's another reason not to allow x0.
*
* 18 Mar 2007: Dima Sorkin <dima.sorkin@gmail.com> reports long strings
*              split across lines (fixed later by Krishna Mohan Gundu's
*              script -- see 30 Aug 2007 below).
*              Also recommends changing %i to %li for long ints.
*
* 05 Jul 2007: Joel Erickson <Joel.Erickson@twosigma.com> reports bug in
*              resid_tol and resid_tol_mach used in the stopping rules:
*                 output->mat_resid_norm   should be
*                 output->frob_mat_norm    in both.
*
* 30 Aug 2007: Krishna Mohan Gundu <gkmohan@gmail.com> also reports
*              long strings split across lines, and provides perl script
*              fix_newlines.pl to change to the form "..." "...".
*              The script is included in the sol/software/lsqr/c/ directory.
*              It has been used to create this version of lsqr.c.
*
*              Krishna also questioned the fact that the test program
*              seems to fail on the last few rectangular cases.
*              Don't worry!  ||A'r|| is in fact quite small.
*              atol and btol have been reduced from 1e-10 to 1e-15
*              to allow more iterations and hence "success".
*
* 02 Sep 2007  Macro "sqr" now defined to be "lsqr_sqr".
*/

#include "lsqr.h"

/*-------------------------------------------------------------------------*/
/*                                                                         */
/*  Define the data type allocation and deallocation functions.            */
/*                                                                         */
/*-------------------------------------------------------------------------*/

     /*---------------------------------------------------------------*/
     /*                                                               */
     /*  Define the error handling function for LSQR and its          */
     /*  associated functions.                                        */ 
     /*                                                               */
     /*---------------------------------------------------------------*/

void lsqr_error( char  *msg,
                 int   code  )
{
  fprintf(stderr, "\t%s\n", msg);
  exit(code);
}

     /*---------------------------------------------------------------*/
     /*                                                               */
     /*  Define the allocation function for a long vector with        */ 
     /*  subscript range x[0..n-1].                                   */ 
     /*                                                               */
     /*---------------------------------------------------------------*/

lvec *alloc_lvec( long  lvec_size )
{
  lvec *lng_vec;

  lng_vec = (lvec *) malloc( sizeof(lvec) );
  if (!lng_vec) lsqr_error("lsqr: vector allocation failure in function "
"alloc_lvec()", -1);

  lng_vec->elements = (long *) malloc( (lvec_size) * sizeof(long) );
  if (!lng_vec->elements) lsqr_error("lsqr: element allocation failure in "
"function alloc_lvec()", -1);

  lng_vec->length = lvec_size;

  return lng_vec;
}

     /*---------------------------------------------------------------*/
     /*                                                               */
     /*  Define the deallocation function for the vector              */ 
     /*  created with the function 'alloc_lvec()'.                    */ 
     /*                                                               */
     /*---------------------------------------------------------------*/

void free_lvec( lvec  *lng_vec )
{
  free((double *) (lng_vec->elements));
  free((lvec *) (lng_vec));
  return;
}

     /*---------------------------------------------------------------*/
     /*                                                               */
     /*  Define the allocation function for a double vector with      */ 
     /*  subscript range x[0..n-1].                                   */ 
     /*                                                               */
     /*---------------------------------------------------------------*/

dvec *alloc_dvec( long  dvec_size )
{
  dvec *dbl_vec;

  dbl_vec = (dvec *) malloc( sizeof(dvec) );
  if (!dbl_vec) lsqr_error("lsqr: vector allocation failure in function "
"alloc_dvec()", -1);

  dbl_vec->elements = (double *) malloc( (dvec_size) * sizeof(double) );
  if (!dbl_vec->elements) lsqr_error("lsqr: element allocation failure in "
"function alloc_dvec()", -1);

  dbl_vec->length = dvec_size;

  return dbl_vec;
}

     /*---------------------------------------------------------------*/
     /*                                                               */
     /*  Define the deallocation function for the vector              */ 
     /*  created with the function 'alloc_dvec()'.                    */ 
     /*                                                               */
     /*---------------------------------------------------------------*/

void free_dvec( dvec  *dbl_vec )
{
  free((double *) (dbl_vec->elements));
  free((dvec *) (dbl_vec));
  return;
}

     /*---------------------------------------------------------------*/
     /*                                                               */
     /*  Define the allocation function for all of the structures     */ 
     /*  used by the function 'lsqr()'.                               */ 
     /*                                                               */
     /*---------------------------------------------------------------*/

void alloc_lsqr_mem( lsqr_input   **in_struct,
                     lsqr_output  **out_struct,
                     lsqr_work    **wrk_struct,
                     lsqr_func    **fnc_struct,
                     long         max_num_rows,
                     long         max_num_cols )
{
  *in_struct = (lsqr_input *) alloc_lsqr_in( max_num_rows, max_num_cols );
  if (!in_struct) lsqr_error("lsqr: input structure allocation failure in "
"function alloc_lsqr_in()", -1);

  *out_struct = (lsqr_output *) alloc_lsqr_out( max_num_rows, max_num_cols );
  if (!out_struct) lsqr_error("lsqr: output structure allocation failure in "
"function alloc_lsqr_out()", -1); 

  (*out_struct)->sol_vec = (dvec *) (*in_struct)->sol_vec;

  *wrk_struct = (lsqr_work *) alloc_lsqr_wrk( max_num_rows, max_num_cols ); 
  if (!wrk_struct) lsqr_error("lsqr: work structure allocation failure in "
"function alloc_lsqr_wrk()", -1);

  *fnc_struct = (lsqr_func *) alloc_lsqr_fnc( );
  if (!fnc_struct) lsqr_error("lsqr: work structure allocation failure in "
"function alloc_lsqr_fnc()", -1);

  return;
}

     /*---------------------------------------------------------------*/
     /*                                                               */
     /*  Define the deallocation function for the structure           */ 
     /*  created with the function 'alloc_lsqr_mem()'.                */ 
     /*                                                               */
     /*---------------------------------------------------------------*/

void free_lsqr_mem( lsqr_input   *in_struct,
                    lsqr_output  *out_struct,
                    lsqr_work    *wrk_struct,
                    lsqr_func    *fnc_struct  )
{
  free_lsqr_in(in_struct);
  free_lsqr_out(out_struct);
  free_lsqr_wrk(wrk_struct);
  free_lsqr_fnc(fnc_struct);
  return;
}

     /*---------------------------------------------------------------*/
     /*                                                               */
     /*  Define the allocation function for the structure of          */ 
     /*  type 'lsqr_input'.                                           */ 
     /*                                                               */
     /*---------------------------------------------------------------*/

lsqr_input *alloc_lsqr_in( long  max_num_rows,
                           long  max_num_cols  )
{
  lsqr_input *in_struct;

  in_struct = (lsqr_input *) malloc( sizeof(lsqr_input) );
  if (!in_struct) lsqr_error("lsqr: input structure allocation failure in "
"function alloc_lsqr_in()", -1);

  in_struct->rhs_vec = (dvec *) alloc_dvec( max_num_rows );
  if (!in_struct->rhs_vec) lsqr_error("lsqr: right hand side vector \'b\' "
"allocation failure in function alloc_lsqr_in()", -1);

  in_struct->sol_vec = (dvec *) alloc_dvec( max_num_cols );
  if (!in_struct->sol_vec) lsqr_error("lsqr: solution vector \'x\' allocation "
"failure in function alloc_lsqr_in()", -1);

  return in_struct;
}

     /*---------------------------------------------------------------*/
     /*                                                               */
     /*  Define the deallocation function for the structure           */ 
     /*  created with the function 'alloc_lsqr_in()'.                 */ 
     /*                                                               */
     /*---------------------------------------------------------------*/

void free_lsqr_in( lsqr_input  *in_struct )
{
  free_dvec(in_struct->rhs_vec);
  free_dvec(in_struct->sol_vec);
  free((lsqr_input *) (in_struct));
  return;
}

     /*---------------------------------------------------------------*/
     /*                                                               */
     /*  Define the allocation function for the structure of          */ 
     /*  type 'lsqr_output'.                                          */ 
     /*                                                               */
     /*---------------------------------------------------------------*/

lsqr_output *alloc_lsqr_out( long  max_num_rows,
                             long  max_num_cols  )
{
  lsqr_output *out_struct;

  out_struct = (lsqr_output *) malloc( sizeof(lsqr_output) );
  if (!out_struct) lsqr_error("lsqr: output structure allocation failure in "
"function alloc_lsqr_out()", -1);

  out_struct->std_err_vec = (dvec *) alloc_dvec( max_num_cols );
  if (!out_struct->std_err_vec) lsqr_error("lsqr: standard error vector \'e\' "
"allocation failure in function alloc_lsqr_out()", -1);

  return out_struct;
}

     /*---------------------------------------------------------------*/
     /*                                                               */
     /*  Define the deallocation function for the structure           */ 
     /*  created with the function 'alloc_lsqr_out()'.                */ 
     /*                                                               */
     /*---------------------------------------------------------------*/

void free_lsqr_out( lsqr_output  *out_struct )
{
  free_dvec(out_struct->std_err_vec);
  free((lsqr_output *) (out_struct));
  return;
}

     /*---------------------------------------------------------------*/
     /*                                                               */
     /*  Define the allocation function for the structure of          */ 
     /*  type 'lsqr_work'.                                            */ 
     /*                                                               */
     /*---------------------------------------------------------------*/

lsqr_work *alloc_lsqr_wrk( long  max_num_rows,
                           long  max_num_cols )
{
  lsqr_work *wrk_struct;

  wrk_struct = (lsqr_work *) malloc( sizeof(lsqr_work) );
  if (!wrk_struct) lsqr_error("lsqr: work structure allocation failure in "
"function alloc_lsqr_wrk()", -1);

  wrk_struct->bidiag_wrk_vec = (dvec *) alloc_dvec( max_num_cols );
  if (!wrk_struct->bidiag_wrk_vec) lsqr_error("lsqr: bidiagonalization work "
"vector \'v\' allocation failure in function alloc_lsqr_wrk()", -1);

  wrk_struct->srch_dir_vec = (dvec *) alloc_dvec( max_num_cols );
  if (!wrk_struct->srch_dir_vec)
     lsqr_error("lsqr: search direction vector \'w\' "
"allocation failure in function alloc_lsqr_wrk()", -1);

  return wrk_struct;
}

     /*---------------------------------------------------------------*/
     /*                                                               */
     /*  Define the deallocation function for the structure           */ 
     /*  created with the function 'alloc_lsqr_wrk()'.                */ 
     /*                                                               */
     /*---------------------------------------------------------------*/

void free_lsqr_wrk( lsqr_work  *wrk_struct )
{
  free_dvec(wrk_struct->bidiag_wrk_vec);
  free_dvec(wrk_struct->srch_dir_vec);
  free((lsqr_work *) (wrk_struct));
  return;
}

     /*---------------------------------------------------------------*/
     /*                                                               */
     /*  Define the allocation function for the structure of          */ 
     /*  type 'lsqr_func'.                                            */ 
     /*                                                               */
     /*---------------------------------------------------------------*/

lsqr_func *alloc_lsqr_fnc( )
{
  lsqr_func *fnc_struct;

  fnc_struct = (lsqr_func *) malloc( sizeof(lsqr_func) );
  if (!fnc_struct) lsqr_error("lsqr: function structure allocation failure in "
"function alloc_lsqr_fnc()", -1);

  return fnc_struct;
}

     /*---------------------------------------------------------------*/
     /*                                                               */
     /*  Define the deallocation function for the structure           */ 
     /*  created with the function 'alloc_lsqr_fnc()'.                */ 
     /*                                                               */
     /*---------------------------------------------------------------*/

void free_lsqr_fnc( lsqr_func  *fnc_struct )
{
  free((lsqr_func *) (fnc_struct));
  return;
}

/*-------------------------------------------------------------------------*/
/*                                                                         */
/*  Define the LSQR function.                                              */
/*                                                                         */
/*-------------------------------------------------------------------------*/

/*
*------------------------------------------------------------------------------
*
*     LSQR  finds a solution x to the following problems:
*
*     1. Unsymmetric equations --    solve  A*x = b
*
*     2. Linear least squares  --    solve  A*x = b
*                                    in the least-squares sense
*
*     3. Damped least squares  --    solve  (   A    )*x = ( b )
*                                           ( damp*I )     ( 0 )
*                                    in the least-squares sense
*
*     where 'A' is a matrix with 'm' rows and 'n' columns, 'b' is an
*     'm'-vector, and 'damp' is a scalar.  (All quantities are real.)
*     The matrix 'A' is intended to be large and sparse.  
*
*
*     Notation
*     --------
*
*     The following quantities are used in discussing the subroutine
*     parameters:
*
*     'Abar'   =  (   A    ),          'bbar'  =  ( b )
*                 ( damp*I )                      ( 0 )
*
*     'r'      =  b  -  A*x,           'rbar'  =  bbar  -  Abar*x
*
*     'rnorm'  =  sqrt( norm(r)**2  +  damp**2 * norm(x)**2 )
*              =  norm( rbar )
*
*     'rel_prec'  =  the relative precision of floating-point arithmetic
*                    on the machine being used.  Typically 2.22e-16
*                    with 64-bit arithmetic.
*
*     LSQR  minimizes the function 'rnorm' with respect to 'x'.
*
*
*     References
*     ----------
*
*     C.C. Paige and M.A. Saunders,  LSQR: An algorithm for sparse
*          linear equations and sparse least squares,
*          ACM Transactions on Mathematical Software 8, 1 (March 1982),
*          pp. 43-71.
*
*     C.C. Paige and M.A. Saunders,  Algorithm 583, LSQR: Sparse
*          linear equations and least-squares problems,
*          ACM Transactions on Mathematical Software 8, 2 (June 1982),
*          pp. 195-209.
*
*     C.L. Lawson, R.J. Hanson, D.R. Kincaid and F.T. Krogh,
*          Basic linear algebra subprograms for Fortran usage,
*          ACM Transactions on Mathematical Software 5, 3 (Sept 1979),
*          pp. 308-323 and 324-325.
*
*------------------------------------------------------------------------------
*/
void lsqr( lsqr_input *input, lsqr_output *output, lsqr_work *work,
           lsqr_func *func,
           void *prod,
		   int (*per_iteration_callback)(lsqr_input*, lsqr_output*, void* token),
		   void* token)
{
  double  dvec_norm2( dvec * );

  long    indx,
          term_iter,
          term_iter_max;
  
  double  alpha, 
          beta,
          rhobar,
          phibar,
          bnorm,
          bbnorm,
          cs1,
          sn1,
          psi,
          rho,
          cs,
          sn,
          theta,
          phi,
          tau,
          ddnorm,
          delta,
          gammabar,
          zetabar,
          gamma,
          cs2,
          sn2,
          zeta,
          xxnorm,
          res,
          resid_tol,
          cond_tol,
          resid_tol_mach,
          temp,
          stop_crit_1,
          stop_crit_2,
          stop_crit_3;

  static char term_msg[8][80] = 
  {
    "The exact solution is x = x0",
    "The residual Ax - b is small enough, given ATOL and BTOL",
    "The least squares error is small enough, given ATOL",
    "The estimated condition number has exceeded CONLIM",
    "The residual Ax - b is small enough, given machine precision",
    "The least squares error is small enough, given machine precision",
    "The estimated condition number has exceeded machine precision",
    "The iteration limit has been reached"
  };

  if( input->lsqr_fp_out != NULL )
    fprintf( input->lsqr_fp_out, "  Least Squares Solution of A*x = b\n"
"        The matrix A has %7li rows and %7li columns\n"
"        The damping parameter is\tDAMP = %10.2e\n"
"        ATOL = %10.2e\t\tCONDLIM = %10.2e\n"
"        BTOL = %10.2e\t\tITERLIM = %10li\n\n",
        input->num_rows, input->num_cols, input->damp_val, input->rel_mat_err,
        input->cond_lim, input->rel_rhs_err, input->max_iter );
  
  output->term_flag = 0;
  term_iter = 0;

  output->num_iters = 0;

  output->frob_mat_norm = 0.0;
  output->mat_cond_num = 0.0;
  output->sol_norm = 0.0;
  
  for(indx = 0; indx < input->num_cols; indx++)
    {
      work->bidiag_wrk_vec->elements[indx] = 0.0;
      work->srch_dir_vec->elements[indx] = 0.0;
      output->std_err_vec->elements[indx] = 0.0;
      output->sol_vec->elements[indx] = 0.0;
    }

  bbnorm = 0.0;
  ddnorm = 0.0;
  xxnorm = 0.0;

  cs2 = -1.0;
  sn2 = 0.0;
  zeta = 0.0;
  res = 0.0;

  if( input->cond_lim > 0.0 )
    cond_tol = 1.0 / input->cond_lim;
  else
    cond_tol = DBL_EPSILON;

  alpha = 0.0;
  beta = 0.0;
/*
*  Set up the initial vectors u and v for bidiagonalization.  These satisfy 
*  the relations
*             BETA*u = b - A*x0 
*             ALPHA*v = A^T*u
*/
    /* Compute b - A*x0 and store in vector u which initially held vector b */
  dvec_scale( (-1.0), input->rhs_vec );
  func->mat_vec_prod( 0, input->sol_vec, input->rhs_vec, prod );
  dvec_scale( (-1.0), input->rhs_vec );
  
    /* compute Euclidean length of u and store as BETA */
  beta = dvec_norm2( input->rhs_vec );
  
  if( beta > 0.0 )
    {
        /* scale vector u by the inverse of BETA */
      dvec_scale( (1.0 / beta), input->rhs_vec );

        /* Compute matrix-vector product A^T*u and store it in vector v */
      func->mat_vec_prod( 1, work->bidiag_wrk_vec, input->rhs_vec, prod );
      
        /* compute Euclidean length of v and store as ALPHA */
      alpha = dvec_norm2( work->bidiag_wrk_vec );      
    }
  
  if( alpha > 0.0 )
    {
        /* scale vector v by the inverse of ALPHA */
      dvec_scale( (1.0 / alpha), work->bidiag_wrk_vec );

        /* copy vector v to vector w */
      dvec_copy( work->bidiag_wrk_vec, work->srch_dir_vec );
    }    

  output->mat_resid_norm = alpha * beta;
  output->resid_norm = beta;
  bnorm = beta;
/*
*  If the norm || A^T r || is zero, then the initial guess is the exact
*  solution.  Exit and report this.
*/
  if( (output->mat_resid_norm == 0.0) && (input->lsqr_fp_out != NULL) )
    {
      fprintf( input->lsqr_fp_out, "\tISTOP = %3li\t\t\tITER = %9li\n"
"        || A ||_F = %13.5e\tcond( A ) = %13.5e\n"
"        || r ||_2 = %13.5e\t|| A^T r ||_2 = %13.5e\n"
"        || b ||_2 = %13.5e\t|| x - x0 ||_2 = %13.5e\n\n", 
        output->term_flag, output->num_iters, output->frob_mat_norm, 
        output->mat_cond_num, output->resid_norm, output->mat_resid_norm,
        bnorm, output->sol_norm );
      
      fprintf( input->lsqr_fp_out, "  %s\n\n", term_msg[output->term_flag]);
      
      return;
    }

  rhobar = alpha;
  phibar = beta;
/*
*  If statistics are printed at each iteration, print a header and the initial
*  values for each quantity.
*/
  if( input->lsqr_fp_out != NULL )
    {
      fprintf( input->lsqr_fp_out,
"  ITER     || r ||    Compatible  "
"||A^T r|| / ||A|| ||r||  || A ||    cond( A )\n\n" );

      stop_crit_1 = 1.0;
      stop_crit_2 = alpha / beta;

      fprintf( input->lsqr_fp_out,
      "%6li %13.5e %10.2e \t%10.2e \t%10.2e  %10.2e\n",
               output->num_iters, output->resid_norm, stop_crit_1, stop_crit_2,
               output->frob_mat_norm, output->mat_cond_num);
    }
 
/*
*  The main iteration loop is continued as long as no stopping criteria
*  are satisfied and the number of total iterations is less than some upper
*  bound.
*/
  while( output->term_flag == 0 )
    {
      output->num_iters++;
/*      
*     Perform the next step of the bidiagonalization to obtain
*     the next vectors u and v, and the scalars ALPHA and BETA.
*     These satisfy the relations
*                BETA*u  =  A*v  -  ALPHA*u,
*                ALFA*v  =  A^T*u  -  BETA*v.
*/      
         /* scale vector u by the negative of ALPHA */
      dvec_scale( (-alpha), input->rhs_vec );

        /* compute A*v - ALPHA*u and store in vector u */
      func->mat_vec_prod( 0, work->bidiag_wrk_vec, input->rhs_vec, prod );

        /* compute Euclidean length of u and store as BETA */
      beta = dvec_norm2( input->rhs_vec );

        /* accumulate this quantity to estimate Frobenius norm of matrix A */
   /* bbnorm += sqr(alpha) + sqr(beta) + sqr(input->damp_val);*/
      bbnorm += alpha*alpha + beta*beta
                + input->damp_val*input->damp_val;

      if( beta > 0.0 )
        {
            /* scale vector u by the inverse of BETA */
          dvec_scale( (1.0 / beta), input->rhs_vec );

            /* scale vector v by the negative of BETA */
          dvec_scale( (-beta), work->bidiag_wrk_vec );

            /* compute A^T*u - BETA*v and store in vector v */
          func->mat_vec_prod( 1, work->bidiag_wrk_vec, input->rhs_vec, prod );
      
            /* compute Euclidean length of v and store as ALPHA */
          alpha = dvec_norm2( work->bidiag_wrk_vec );

          if( alpha > 0.0 )
              /* scale vector v by the inverse of ALPHA */
            dvec_scale( (1.0 / alpha), work->bidiag_wrk_vec );
        }
/*
*     Use a plane rotation to eliminate the damping parameter.
*     This alters the diagonal (RHOBAR) of the lower-bidiagonal matrix.
*/
      cs1 = rhobar / sqrt( lsqr_sqr(rhobar) + lsqr_sqr(input->damp_val) );
      sn1 = input->damp_val
                   / sqrt( lsqr_sqr(rhobar) + lsqr_sqr(input->damp_val) );
      
      psi = sn1 * phibar;
      phibar = cs1 * phibar;
/*      
*     Use a plane rotation to eliminate the subdiagonal element (BETA)
*     of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
*/
      rho = sqrt( lsqr_sqr(rhobar) + lsqr_sqr(input->damp_val)
                                   + lsqr_sqr(beta) );
      cs  = sqrt( lsqr_sqr(rhobar) + lsqr_sqr(input->damp_val) ) / rho;
      sn  = beta / rho;

      theta = sn * alpha;
      rhobar = -cs * alpha;
      phi = cs * phibar;
      phibar = sn * phibar;
      tau = sn * phi;
/*
*     Update the solution vector x, the search direction vector w, and the 
*     standard error estimates vector se.
*/     
      for(indx = 0; indx < input->num_cols; indx++)
        {
            /* update the solution vector x */
          output->sol_vec->elements[indx] += (phi / rho) * 
            work->srch_dir_vec->elements[indx];

            /* update the standard error estimates vector se */
          output->std_err_vec->elements[indx] += lsqr_sqr( (1.0 / rho) * 
            work->srch_dir_vec->elements[indx] );

            /* accumulate this quantity to estimate condition number of A 
*/
          ddnorm += lsqr_sqr( (1.0 / rho) *
            work->srch_dir_vec->elements[indx] );

            /* update the search direction vector w */
          work->srch_dir_vec->elements[indx] = 
          work->bidiag_wrk_vec->elements[indx] -
            (theta / rho) * work->srch_dir_vec->elements[indx];
        }
/*
*     Use a plane rotation on the right to eliminate the super-diagonal element
*     (THETA) of the upper-bidiagonal matrix.  Then use the result to estimate 
*     the solution norm || x ||.
*/
      delta = sn2 * rho;
      gammabar = -cs2 * rho;
      zetabar = (phi - delta * zeta) / gammabar;

        /* compute an estimate of the solution norm || x || */
      output->sol_norm = sqrt( xxnorm + lsqr_sqr(zetabar) );

      gamma = sqrt( lsqr_sqr(gammabar) + lsqr_sqr(theta) );
      cs2 = gammabar / gamma;
      sn2 = theta / gamma;
      zeta = (phi - delta * zeta) / gamma;

        /* accumulate this quantity to estimate solution norm || x || */
      xxnorm += lsqr_sqr(zeta);
/*
*     Estimate the Frobenius norm and condition of the matrix A, and the 
*     Euclidean norms of the vectors r and A^T*r.
*/
      output->frob_mat_norm = sqrt( bbnorm );
      output->mat_cond_num = output->frob_mat_norm * sqrt( ddnorm );

      res += lsqr_sqr(psi);
      output->resid_norm = sqrt( lsqr_sqr(phibar) + res );

      output->mat_resid_norm = alpha * fabs( tau );
/*
*     Use these norms to estimate the values of the three stopping criteria.
*/
      stop_crit_1 = output->resid_norm / bnorm;

      stop_crit_2 = 0.0;
      if( output->resid_norm > 0.0 )
        stop_crit_2 = output->mat_resid_norm / ( output->frob_mat_norm * 
          output->resid_norm );

      stop_crit_3 = 1.0 / output->mat_cond_num;

/*    05 Jul 2007: Bug reported by Joel Erickson <Joel.Erickson@twosigma.com>.
*/
      resid_tol = input->rel_rhs_err + input->rel_mat_err * 
        output->frob_mat_norm *         /* (not output->mat_resid_norm *) */
        output->sol_norm / bnorm;

      resid_tol_mach = DBL_EPSILON + DBL_EPSILON *
        output->frob_mat_norm *         /* (not output->mat_resid_norm *) */
        output->sol_norm / bnorm;
/*
*     Check to see if any of the stopping criteria are satisfied.
*     First compare the computed criteria to the machine precision.
*     Second compare the computed criteria to the the user specified precision.
*/
        /* iteration limit reached */
      if( output->num_iters >= input->max_iter )
        output->term_flag = 7;

        /* condition number greater than machine precision */
      if( stop_crit_3 <= DBL_EPSILON )
        output->term_flag = 6;
        /* least squares error less than machine precision */
      if( stop_crit_2 <= DBL_EPSILON )
        output->term_flag = 5;
        /* residual less than a function of machine precision */
      if( stop_crit_1 <= resid_tol_mach )
        output->term_flag = 4;

        /* condition number greater than CONLIM */
      if( stop_crit_3 <= cond_tol )
        output->term_flag = 3;
        /* least squares error less than ATOL */
      if( stop_crit_2 <= input->rel_mat_err )
        output->term_flag = 2;
        /* residual less than a function of ATOL and BTOL */
      if( stop_crit_1 <= resid_tol )
        output->term_flag = 1;
/*
*  If statistics are printed at each iteration, print a header and the initial
*  values for each quantity.
*/
      if( input->lsqr_fp_out != NULL )
        {
          fprintf( input->lsqr_fp_out,
          "%6li %13.5e %10.2e \t%10.2e \t%10.2e %10.2e\n",
                   output->num_iters, output->resid_norm, stop_crit_1, 
                   stop_crit_2,
                   output->frob_mat_norm, output->mat_cond_num);
        }
/*
*     The convergence criteria are required to be met on NCONV consecutive 
*     iterations, where NCONV is set below.  Suggested values are 1, 2, or 3.
*/
      if( output->term_flag == 0 )
        term_iter = -1;

      term_iter_max = 1;
      term_iter++;

      if( (term_iter < term_iter_max) &&
          (output->num_iters < input->max_iter) )
        output->term_flag = 0;

	  if (per_iteration_callback)
		  per_iteration_callback(input, output, token);

    } /* end while loop */
/*
*  Finish computing the standard error estimates vector se.
*/
  temp = 1.0;

  if( input->num_rows > input->num_cols )
    temp = ( double ) ( input->num_rows - input->num_cols );

  if( lsqr_sqr(input->damp_val) > 0.0 )
    temp = ( double ) ( input->num_rows );
  
  temp = output->resid_norm / sqrt( temp );
  
  for(indx = 0; indx < input->num_cols; indx++)
      /* update the standard error estimates vector se */
    output->std_err_vec->elements[indx] = temp * 
      sqrt( output->std_err_vec->elements[indx] );
/*
*  If statistics are printed at each iteration, print the statistics for the
*  stopping condition.
*/
  if( input->lsqr_fp_out != NULL )
    {
      fprintf( input->lsqr_fp_out, "\n\tISTOP = %3li\t\t\tITER = %9li\n"
"        || A ||_F = %13.5e\tcond( A ) = %13.5e\n"
"        || r ||_2 = %13.5e\t|| A^T r ||_2 = %13.5e\n"
"        || b ||_2 = %13.5e\t|| x - x0 ||_2 = %13.5e\n\n", 
        output->term_flag, output->num_iters, output->frob_mat_norm, 
        output->mat_cond_num, output->resid_norm, output->mat_resid_norm,
        bnorm, output->sol_norm );

      fprintf( input->lsqr_fp_out, "  %s\n\n", term_msg[output->term_flag]);
      
    }

  return;
}

/*-------------------------------------------------------------------------*/
/*                                                                         */
/*  Define the function 'dvec_norm2()'.  This function takes a vector      */
/*  arguement and computes the Euclidean or L2 norm of this vector.  Note  */
/*  that this is a version of the BLAS function 'dnrm2()' rewritten to     */
/*  use the current data structures.                                       */
/*                                                                         */
/*-------------------------------------------------------------------------*/

double dvec_norm2(dvec *vec)
{
  long   indx;
  double norm;
  
  norm = 0.0;

  for(indx = 0; indx < vec->length; indx++)
    norm += lsqr_sqr(vec->elements[indx]);

  return sqrt(norm);
}


/*-------------------------------------------------------------------------*/
/*                                                                         */
/*  Define the function 'dvec_scale()'.  This function takes a vector      */
/*  and a scalar as arguments and multiplies each element of the vector    */
/*  by the scalar.  Note  that this is a version of the BLAS function      */
/*  'dscal()' rewritten to use the current data structures.                */
/*                                                                         */
/*-------------------------------------------------------------------------*/

void dvec_scale(double scal, dvec *vec)
{
  long   indx;

  for(indx = 0; indx < vec->length; indx++)
    vec->elements[indx] *= scal;

  return;
}

/*-------------------------------------------------------------------------*/
/*                                                                         */
/*  Define the function 'dvec_copy()'.  This function takes two vectors    */
/*  as arguements and copies the contents of the first into the second.    */
/*  Note  that this is a version of the BLAS function 'dcopy()' rewritten  */
/*  to use the current data structures.                                    */
/*                                                                         */
/*-------------------------------------------------------------------------*/

void dvec_copy(dvec *orig, dvec *copy)
{
  long   indx;

  for(indx = 0; indx < orig->length; indx++)
    copy->elements[indx] = orig->elements[indx];

  return;
}
