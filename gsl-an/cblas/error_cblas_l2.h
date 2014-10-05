/* cblas/error_cblas_l2.h
 *
 * Copyright (C) 2010 José Luis García Pallero
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

#ifndef __ERROR_CBLAS_L2_H__
#define __ERROR_CBLAS_L2_H__
#include <gsl/gsl_math.h>
#include "error_cblas.h"
/*
 * =============================================================================
 * Prototypes for level 2 BLAS
 * =============================================================================
 */

/*
 * Routines with standard 4 prefixes (S, D, C, Z)
 */

/* cblas_xgemv() */
#define CBLAS_ERROR_GEMV(pos,order,TransA,M,N,alpha,A,lda,X,incX,beta,Y,incY) \
CHECK_ORDER(pos,1,order); \
CHECK_TRANSPOSE(pos,2,TransA); \
CHECK_DIM(pos,3,M); \
CHECK_DIM(pos,4,N); \
if((order)==CblasRowMajor) { \
    if((lda)<GSL_MAX(1,N)) { \
        pos = 7; \
    } \
} else if((order)==CblasColMajor) { \
    if((lda)<GSL_MAX(1,M)) { \
        pos = 7; \
    } \
};                       \
CHECK_STRIDE(pos,9,incX); \
CHECK_STRIDE(pos,12,incY);

/* cblas_xgbmv() */
#define CBLAS_ERROR_GBMV(pos,order,TransA,M,N,KL,KU,alpha,A,lda,X,incX,beta,Y,incY) \
CHECK_ORDER(pos,1,order); \
CHECK_TRANSPOSE(pos,2,TransA); \
CHECK_DIM(pos,3,M); \
CHECK_DIM(pos,4,N); \
CHECK_DIM(pos,5,KL); \
CHECK_DIM(pos,6,KU); \
if((lda)<GSL_MAX(1,(KL+KU+1))) { \
    pos = 9; \
};                        \
CHECK_STRIDE(pos,11,incX); \
CHECK_STRIDE(pos,14,incY);

/* cblas_xtrmv() */
#define CBLAS_ERROR_TRMV(pos,order,Uplo,TransA,Diag,N,A,lda,X,incX) \
CHECK_ORDER(pos,1,order); \
CHECK_UPLO(pos,2,Uplo); \
CHECK_TRANSPOSE(pos,3,TransA); \
CHECK_DIAG(pos,4,Diag); \
CHECK_DIM(pos,5,N); \
if((lda)<GSL_MAX(1,N)) \
    pos = 7; \
CHECK_STRIDE(pos,9,incX);

/* cblas_xtbmv() */
#define CBLAS_ERROR_TBMV(pos,order,Uplo,TransA,Diag,N,K,A,lda,X,incX) \
CHECK_ORDER(pos,1,order); \
CHECK_UPLO(pos,2,Uplo); \
CHECK_TRANSPOSE(pos,3,TransA); \
CHECK_DIAG(pos,4,Diag); \
CHECK_DIM(pos,5,N); \
CHECK_DIM(pos,6,K); \
if((lda)<GSL_MAX(1,(K+1))) { \
    pos = 8; \
}; \
CHECK_STRIDE(pos,10,incX);

/* cblas_xtpmv() */
#define CBLAS_ERROR_TPMV(pos,order,Uplo,TransA,Diag,N,Ap,X,incX) \
CHECK_ORDER(pos,1,order); \
CHECK_UPLO(pos,2,Uplo); \
CHECK_TRANSPOSE(pos,3,TransA); \
CHECK_DIAG(pos,4,Diag); \
CHECK_DIM(pos,5,N); \
CHECK_STRIDE(pos,8,incX);

/* cblas_xtrsv() */
#define CBLAS_ERROR_TRSV(pos,order,Uplo,TransA,Diag,N,A,lda,X,incX) \
CBLAS_ERROR_TRMV(pos,order,Uplo,TransA,Diag,N,A,lda,X,incX)

/* cblas_xtbsv() */
#define CBLAS_ERROR_TBSV(pos,order,Uplo,TransA,Diag,N,K,A,lda,X,incX) \
CBLAS_ERROR_TBMV(pos,order,Uplo,TransA,Diag,N,K,A,lda,X,incX)

/* cblas_xtpsv() */
#define CBLAS_ERROR_TPSV(pos,order,Uplo,TransA,Diag,N,Ap,X,incX) \
CBLAS_ERROR_TPMV(pos,order,Uplo,TransA,Diag,N,Ap,X,incX)

/*
 * Routines with S and D prefixes only
 */

/* cblas_xsymv() */
#define CBLAS_ERROR_SD_SYMV(pos,order,Uplo,N,alpha,A,lda,X,incX,beta,Y,incY) \
CHECK_ORDER(pos,1,order); \
CHECK_UPLO(pos,2,Uplo); \
CHECK_DIM(pos,3,N); \
if((lda)<GSL_MAX(1,N)) { \
    pos = 6; \
};                       \
CHECK_STRIDE(pos,8,incX); \
CHECK_STRIDE(pos,11,incY);

/* cblas_xsbmv() */
#define CBLAS_ERROR_SD_SBMV(pos,order,Uplo,N,K,alpha,A,lda,X,incX,beta,Y,incY) \
CHECK_ORDER(pos,1,order); \
CHECK_UPLO(pos,2,Uplo); \
CHECK_DIM(pos,3,N); \
CHECK_DIM(pos,4,K); \
if((lda)<GSL_MAX(1,K+1)) { \
    pos = 7; \
};                       \
CHECK_STRIDE(pos,9,incX); \
CHECK_STRIDE(pos,12,incY);

/* cblas_xspmv() */
#define CBLAS_ERROR_SD_SPMV(pos,order,Uplo,N,alpha,Ap,X,incX,beta,Y,incY) \
CHECK_ORDER(pos,1,order); \
CHECK_UPLO(pos,2,Uplo); \
CHECK_DIM(pos,3,N); \
CHECK_STRIDE(pos,7,incX); \
CHECK_STRIDE(pos,10,incY);

/* cblas_xger() */
#define CBLAS_ERROR_SD_GER(pos,order,M,N,alpha,X,incX,Y,incY,A,lda) \
CHECK_ORDER(pos,1,order); \
CHECK_DIM(pos,2,M); \
CHECK_DIM(pos,3,N); \
CHECK_STRIDE(pos,6,incX); \
CHECK_STRIDE(pos,8,incY); \
if((order)==CblasRowMajor) { \
    if((lda)<GSL_MAX(1,N)) { \
        pos = 10; \
    } \
} else if((order)==CblasColMajor) { \
    if((lda)<GSL_MAX(1,M)) { \
        pos = 10; \
    } \
};

/* cblas_xsyr() */
#define CBLAS_ERROR_SD_SYR(pos,order,Uplo,N,alpha,X,incX,A,lda) \
CHECK_ORDER(pos,1,order); \
CHECK_UPLO(pos,2,Uplo); \
CHECK_DIM(pos,3,N); \
CHECK_STRIDE(pos,6,incX); \
if((lda)<GSL_MAX(1,N)) { \
    pos = 8; \
};

/* cblas_xspr() */
#define CBLAS_ERROR_SD_SPR(pos,order,Uplo,N,alpha,X,incX,Ap) \
CHECK_ORDER(pos,1,order); \
CHECK_UPLO(pos,2,Uplo); \
CHECK_DIM(pos,3,N); \
CHECK_STRIDE(pos,6,incX);

/* cblas_xsyr2() */
#define CBLAS_ERROR_SD_SYR2(pos,order,Uplo,N,alpha,X,incX,Y,incY,A,lda) \
CHECK_ORDER(pos,1,order); \
CHECK_UPLO(pos,2,Uplo); \
CHECK_DIM(pos,3,N); \
CHECK_STRIDE(pos,6,incX); \
CHECK_STRIDE(pos,8,incY); \
if((lda)<GSL_MAX(1,N)) { \
    pos = 10; \
};

/* cblas_xspr2() */
#define CBLAS_ERROR_SD_SPR2(pos,order,Uplo,N,alpha,X,incX,Y,incY,A) \
CHECK_ORDER(pos,1,order); \
CHECK_UPLO(pos,2,Uplo); \
CHECK_DIM(pos,3,N); \
CHECK_STRIDE(pos,6,incX); \
CHECK_STRIDE(pos,8,incY);

/*
 * Routines with C and Z prefixes only
 */

/* cblas_xhemv() */
#define CBLAS_ERROR_CZ_HEMV(pos,order,Uplo,N,alpha,A,lda,X,incX,beta,Y,incY) \
CBLAS_ERROR_SD_SYMV(pos,order,Uplo,N,alpha,A,lda,X,incX,beta,Y,incY)

/* cblas_xhbmv() */
#define CBLAS_ERROR_CZ_HBMV(pos,order,Uplo,N,K,alpha,A,lda,X,incX,beta,Y,incY) \
CBLAS_ERROR_SD_SBMV(pos,order,Uplo,N,K,alpha,A,lda,X,incX,beta,Y,incY)

/* cblas_xhpmv() */
#define CBLAS_ERROR_CZ_HPMV(pos,order,Uplo,N,alpha,Ap,X,incX,beta,Y,incY) \
CBLAS_ERROR_SD_SPMV(pos,order,Uplo,N,alpha,Ap,X,incX,beta,Y,incY)

/* cblas_xgeru() */
#define CBLAS_ERROR_CZ_GERU(pos,order,M,N,alpha,X,incX,Y,incY,A,lda) \
CBLAS_ERROR_SD_GER(pos,order,M,N,alpha,X,incX,Y,incY,A,lda)

/* cblas_xgerc() */
#define CBLAS_ERROR_CZ_GERC(pos,order,M,N,alpha,X,incX,Y,incY,A,lda) \
CBLAS_ERROR_SD_GER(pos,order,M,N,alpha,X,incX,Y,incY,A,lda)

/* cblas_xher() */
#define CBLAS_ERROR_CZ_HER(pos,order,Uplo,N,alpha,X,incX,A,lda) \
CBLAS_ERROR_SD_SYR(pos,order,Uplo,N,alpha,X,incX,A,lda)

/* cblas_xhpr() */
#define CBLAS_ERROR_CZ_HPR(pos,order,Uplo,N,alpha,X,incX,A) \
CBLAS_ERROR_SD_SPR(pos,order,Uplo,N,alpha,X,incX,A)

/* cblas_xher2() */
#define CBLAS_ERROR_CZ_HER2(pos,order,Uplo,N,alpha,X,incX,Y,incY,A,lda) \
CBLAS_ERROR_SD_SYR2(pos,order,Uplo,N,alpha,X,incX,Y,incY,A,lda)

/* cblas_xhpr2() */
#define CBLAS_ERROR_CZ_HPR2(pos,order,Uplo,N,alpha,X,incX,Y,incY,Ap) \
CBLAS_ERROR_SD_SPR2(pos,order,Uplo,N,alpha,X,incX,Y,incY,Ap)

#endif /* __ERROR_CBLAS_L2_H__ */
