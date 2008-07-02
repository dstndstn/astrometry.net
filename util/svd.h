/*
  Please see the file "svd.c" for copyright information.
 */

#ifndef SVD_H
#define SVD_H

int svd(int m,int n,int withu,int withv,double eps,double tol,
		double **a,double *q,double **u,double **v);

#endif
