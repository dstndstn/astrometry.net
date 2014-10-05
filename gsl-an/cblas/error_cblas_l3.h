/* cblas/error_cblas_l3.h
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

#ifndef __ERROR_CBLAS_L3_H__
#define __ERROR_CBLAS_L3_H__
#include <gsl/gsl_math.h>
#include "error_cblas.h"

/*
 * =============================================================================
 * Prototypes for level 3 BLAS
 * =============================================================================
 */

/*
 * Routines with standard 4 prefixes (S, D, C, Z)
 */

/* cblas_xgemm() */
#define CBLAS_ERROR_GEMM(pos,Order,TransA,TransB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc) \
{ \
    enum CBLAS_TRANSPOSE __transF=CblasNoTrans,__transG=CblasNoTrans; \
    if((Order)==CblasRowMajor) { \
        __transF = ((TransA)!=CblasConjTrans) ? (TransA) : CblasTrans; \
        __transG = ((TransB)!=CblasConjTrans) ? (TransB) : CblasTrans; \
    } else { \
        __transF = ((TransB)!=CblasConjTrans) ? (TransB) : CblasTrans; \
        __transG = ((TransA)!=CblasConjTrans) ? (TransA) : CblasTrans; \
    } \
    CHECK_ORDER(pos,1,Order); \
    CHECK_TRANSPOSE(pos,2,TransA); \
    CHECK_TRANSPOSE(pos,3,TransB); \
    CHECK_DIM(pos,4,M); \
    CHECK_DIM(pos,5,N); \
    CHECK_DIM(pos,6,K); \
    if((Order)==CblasRowMajor) { \
        if(__transF==CblasNoTrans) { \
            if((lda)<GSL_MAX(1,(K))) { \
                (pos) = 9; \
            } \
        } else { \
            if((lda)<GSL_MAX(1,(M))) { \
                (pos) = 9; \
            } \
        } \
        if(__transG==CblasNoTrans) { \
            if((ldb)<GSL_MAX(1,(N))) { \
                (pos) = 11; \
            } \
        } else { \
            if((ldb)<GSL_MAX(1,(K))) { \
                (pos) = 11; \
            } \
        } \
        if((ldc)<GSL_MAX(1,(N))) { \
            (pos) = 14; \
        } \
    } else if((Order)==CblasColMajor) { \
        if(__transF==CblasNoTrans) { \
            if((ldb)<GSL_MAX(1,(K))) { \
                (pos) = 11; \
            } \
        } else { \
            if((ldb)<GSL_MAX(1,(N))) { \
                (pos) = 11; \
            } \
        } \
        if(__transG==CblasNoTrans) { \
            if((lda)<GSL_MAX(1,(M))) { \
                (pos) = 9; \
            } \
        } else { \
            if((lda)<GSL_MAX(1,(K))) { \
                (pos) = 9; \
            } \
        } \
        if((ldc)<GSL_MAX(1,(M))) { \
            (pos) = 14; \
        } \
    } \
}

/* cblas_xsymm() */
#define CBLAS_ERROR_SYMM(pos,Order,Side,Uplo,M,N,alpha,A,lda,B,ldb,beta,C,ldc) \
{ \
    int __dimA=0; \
    if((Side)==CblasLeft) { \
        __dimA = (M); \
    } else { \
        __dimA = (N); \
    } \
    CHECK_ORDER(pos,1,Order); \
    CHECK_SIDE(pos,2,Side) \
    CHECK_UPLO(pos,3,Uplo); \
    CHECK_DIM(pos,4,M); \
    CHECK_DIM(pos,5,N); \
    if((lda)<GSL_MAX(1,__dimA)) { \
        (pos) = 8; \
    } \
    if((Order)==CblasRowMajor) { \
        if((ldb)<GSL_MAX(1,(N))) { \
                (pos) = 10; \
        } \
        if((ldc)<GSL_MAX(1,(N))) { \
                (pos) = 13; \
        } \
    } else if((Order)==CblasColMajor) { \
        if((ldb)<GSL_MAX(1,(M))) { \
                (pos) = 10; \
        } \
        if((ldc)<GSL_MAX(1,(M))) { \
                (pos) = 13; \
        } \
    } \
}

/* cblas_xsyrk() */
#define CBLAS_ERROR_SYRK(pos,Order,Uplo,Trans,N,K,alpha,A,lda,beta,C,ldc) \
{ \
    int __dimA=0; \
    if((Order)==CblasRowMajor) { \
        if((Trans)==CblasNoTrans) { \
            __dimA = (K); \
        } else { \
            __dimA = (N); \
        } \
    } else { \
        if((Trans)==CblasNoTrans) { \
            __dimA = (N); \
        } else { \
            __dimA = (K); \
        } \
    } \
    CHECK_ORDER(pos,1,Order); \
    CHECK_UPLO(pos,2,Uplo); \
    CHECK_TRANSPOSE(pos,3,Trans); \
    CHECK_DIM(pos,4,N); \
    CHECK_DIM(pos,5,K); \
    if((lda)<GSL_MAX(1,__dimA)) { \
        (pos) = 8; \
    } \
    if((ldc)<GSL_MAX(1,(N))) { \
        (pos) = 11; \
    } \
}

/* cblas_xsyr2k() */
#define CBLAS_ERROR_SYR2K(pos,Order,Uplo,Trans,N,K,alpha,A,lda,B,ldb,beta,C,ldc) \
{ \
    int __dim=0; \
    if((Order)==CblasRowMajor) { \
        if((Trans)==CblasNoTrans) { \
            __dim = (K); \
        } else { \
            __dim = (N); \
        } \
    } else { \
        if((Trans)==CblasNoTrans) { \
            __dim = (N); \
        } else { \
            __dim = (K); \
        } \
    } \
    CHECK_ORDER(pos,1,Order); \
    CHECK_UPLO(pos,2,Uplo); \
    CHECK_TRANSPOSE(pos,3,Trans); \
    CHECK_DIM(pos,4,N); \
    CHECK_DIM(pos,5,K); \
    if((lda)<GSL_MAX(1,__dim)) { \
        (pos) = 8; \
    } \
    if((ldb)<GSL_MAX(1,__dim)) { \
        (pos) = 11; \
    } \
    if((ldc)<GSL_MAX(1,(N))) { \
        (pos) = 14; \
    } \
}

/* cblas_xtrmm() */
#define CBLAS_ERROR_TRMM(pos,Order,Side,Uplo,TransA,Diag,M,N,alpha,A,lda,B,ldb) \
{ \
    int __dim=0; \
    if((Side)==CblasLeft) { \
        __dim = (M); \
    } else { \
        __dim = (N); \
    } \
    CHECK_ORDER(pos,1,Order); \
    CHECK_SIDE(pos,2,Side); \
    CHECK_UPLO(pos,3,Uplo); \
    CHECK_TRANSPOSE(pos,4,TransA); \
    CHECK_DIAG(pos,5,Diag); \
    CHECK_DIM(pos,6,M); \
    CHECK_DIM(pos,7,N); \
    if((lda)<GSL_MAX(1,__dim)) { \
        (pos) = 10; \
    } \
    if((Order)==CblasRowMajor) { \
        if((ldb)<GSL_MAX(1,(N))) { \
            (pos) = 12; \
        } \
    } else { \
        if((ldb)<GSL_MAX(1,(M))) { \
            (pos) = 12; \
        } \
    } \
}

/* cblas_xtrsm() */
#define CBLAS_ERROR_TRSM(pos,Order,Side,Uplo,TransA,Diag,M,N,alpha,A,lda,B,ldb) \
CBLAS_ERROR_TRMM(pos,Order,Side,Uplo,TransA,Diag,M,N,alpha,A,lda,B,ldb)

/*
 * Routines with prefixes C and Z only
 */

/* cblas_xhemm() */
#define CBLAS_ERROR_HEMM(pos,Order,Side,Uplo,M,N,alpha,A,lda,B,ldb,beta,C,ldc) \
CBLAS_ERROR_SYMM(pos,Order,Side,Uplo,M,N,alpha,A,lda,B,ldb,beta,C,ldc)

/* cblas_xherk() */
#define CBLAS_ERROR_HERK(pos,Order,Uplo,Trans,N,K,alpha,A,lda,beta,C,ldc) \
CBLAS_ERROR_SYRK(pos,Order,Uplo,Trans,N,K,alpha,A,lda,beta,C,ldc)

/* cblas_xher2k() */
#define CBLAS_ERROR_HER2K(pos,Order,Uplo,Trans,N,K,alpha,A,lda,B,ldb,beta,C,ldc) \
CBLAS_ERROR_SYR2K(pos,Order,Uplo,Trans,N,K,alpha,A,lda,B,ldb,beta,C,ldc)

#endif /* __ERROR_CBLAS_L3_H__ */
