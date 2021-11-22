/* $Id: qfits_float.h,v 1.3 2006/02/17 10:24:52 yjung Exp $
 *
 * This file is part of the ESO QFITS Library
 * Copyright (C) 2001-2004 European Southern Observatory
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

/*
 * $Author: yjung $
 * $Date: 2006/02/17 10:24:52 $
 * $Revision: 1.3 $
 * $Name: qfits-6_2_0 $
 */

#ifndef QFITS_FLOAT_H
#define QFITS_FLOAT_H

/*-----------------------------------------------------------------------------
                                   Macros
 -----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/**
  @brief    Test a floating-point variable for NaN value.
  @param    n   Number to test (float or double)
  @return   1 if n is NaN, 0 else.

  This macro is needed to support both float and double variables
  as input parameter. It checks on the size of the input variable
  to branch to the float or double version.

  Portability is an issue for this function which is present on
  most Unixes but not all, under various libraries (C lib on BSD,
  Math lib on Linux, sunmath on Solaris, ...). Integrating the
  code for this function makes qfits independent from any math
  library.
 */
/*----------------------------------------------------------------------------*/
#define qfits_isnan(n) ((sizeof(n)==sizeof(float)) ? _qfits_isnanf(n) : \
                        (sizeof(n)==sizeof(double)) ? _qfits_isnand(n) : -1)

/*----------------------------------------------------------------------------*/
/**
  @brief    Test a floating-point variable for Inf value.
  @param    n   Number to test (float or double)
  @return   1 if n is Inf or -Inf, 0 else.

  This macro is needed to support both float and double variables
  as input parameter. It checks on the size of the input variable
  to branch to the float or double version.

  Portability is an issue for this function which is missing on most
  Unixes. Most of the time, another function called finite() is
  offered to perform the opposite task, but it is not consistent
  among platforms and found in various libraries. Integrating the
  code for this function makes qfits independent from any math
  library.
 */
/*----------------------------------------------------------------------------*/
#define qfits_isinf(n) ((sizeof(n)==sizeof(float)) ? _qfits_isinff(n) : \
                        (sizeof(n)==sizeof(double)) ? _qfits_isinfd(n) : -1)

/*-----------------------------------------------------------------------------
                               Function prototypes
 -----------------------------------------------------------------------------*/

int _qfits_isnanf(float);
int _qfits_isinff(float);
int _qfits_isnand(double);
int _qfits_isinfd(double);

#endif
