/* Note: this file has been modified from its original form by the
   Astrometry.net team.  For details see http://astrometry.net */

/* $Id: qfits_error.c,v 1.6 2006/02/17 10:24:52 yjung Exp $
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
 * $Revision: 1.6 $
 * $Name: qfits-6_2_0 $
 */

/*-----------------------------------------------------------------------------
                                   Includes
 -----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdarg.h>

#include "qfits_error.h"

/*-----------------------------------------------------------------------------
                                   Defines
 -----------------------------------------------------------------------------*/

/* Max number of error handling functions registered */
#define QFITS_ERR_MAXERRDISP        8
/* Max size of an error message */
#define QFITS_ERR_MSGSIZE            1024

/*-----------------------------------------------------------------------------
                               Static declarations
 -----------------------------------------------------------------------------*/

/* Type of a display function only defined for legibility here */
typedef void (*qfits_err_dispfunc)(char *);

/* Default display function prints out msg to stderr */
static void qfits_err_display_stderr(char * s)
{ fprintf(stderr, "qfits: %s\n", s); }

/* Static control structure, completely private */
static struct {
    qfits_err_dispfunc     disp[QFITS_ERR_MAXERRDISP];
    int                 n;
    int                    active;
} qfits_err_control = {{qfits_err_display_stderr}, 1, 0};

static void qfits_err_main_display(char *);

/*----------------------------------------------------------------------------*/
/**
 * @defgroup    qfits_error     Messaging functionalities 
 *
 *   This module is responsible for error message display. It allows
 *  to re-direct all messages to a given set of functions for display.
 *
 */
/*----------------------------------------------------------------------------*/
/**@{*/

/*-----------------------------------------------------------------------------
                              Function codes
 -----------------------------------------------------------------------------*/

void qfits_err_remove_all(void) {
    qfits_err_control.n = 0;
}

/* Public warning/error functions */
void qfits_warning(const char *fmt, ...)
{
    char msg[QFITS_ERR_MSGSIZE];
    char all[QFITS_ERR_MSGSIZE];
    va_list ap;

    /* Check if display is activated */
    if (qfits_err_control.active==0) {
        return;
    }
    va_start(ap, fmt);
    vsprintf(msg, fmt, ap);
    va_end(ap);

    sprintf(all, "*** %s", msg);
    qfits_err_main_display(all);
    return;
}
void qfits_error(const char *fmt, ...)
{
    char msg[QFITS_ERR_MSGSIZE];
    char all[QFITS_ERR_MSGSIZE];
    va_list ap;

    /* Check if display is activated */
    if (qfits_err_control.active==0) {
        return;
    }
    va_start(ap, fmt);
    vsprintf(msg, fmt, ap);
    va_end(ap);

    sprintf(all, "error: %s", msg);
    qfits_err_main_display(all);
    return;
}
/**@}*/

/*----------------------------------------------------------------------------*/
/**
  @brief    used for message display
  @param    msg     message
  @return   nothing
  It calls registered display functions one after another.
 */
/*----------------------------------------------------------------------------*/
static void qfits_err_main_display(char * msg)
{
    int    i;

    /* Check if there is a message in input */
    if (msg==NULL)
        return;

    /* Loop on all registered functions and call them */
    for (i=0; i<qfits_err_control.n; i++) {
        if (qfits_err_control.disp[i]) {
            qfits_err_control.disp[i](msg);
        }
    }
    return;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Get the current status of error display.
  @return    int 1 if error display is active, 0 if not.

  This function returns the current error display status. If it returns 1,
  it means that all calls to qfits_error/qfits_warning will display
  messages using the registered functions, otherwise they do nothing.
 */
/*----------------------------------------------------------------------------*/
int qfits_err_statget(void)
{
    return qfits_err_control.active;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Set the current status of error display.
  @param    sta        New status to be set.
  @return    int giving the previous display status.

  This function sets the current error display status to the required
  value, and returns the previous value. It is useful to store the
  previous value, in view of restoring it afterwards, e.g. to make a
  function silent on all its errors. Example:

  @code
  int prev_stat = qfits_err_statset(0);
  function_call();
  qfits_err_statset(prev_stat);
  @endcode
 */
/*----------------------------------------------------------------------------*/
int qfits_err_statset(int sta)
{
    int prev;
    prev = qfits_err_control.active;
    qfits_err_control.active=sta;
    return prev;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Register a function to display error/warning messages.
  @param    dispfn    Display function (see doc below).
  @return    int 0 if function was registered, -1 if not.

  This function registers a display function into the error-handling
  module. Display functions have the following prototype:

  @code
  void display_function(char * msg);
  @endcode

  They are simple functions that expect a ready-made error message
  and return void. They can do whatever they want with the message
  (log it to a file, send it to a GUI, to the syslog, ...). The
  message is built using a printf-like statement in qfits_error and
  qfits_warning, then passed to all registered display functions.

  A maximum of QFITS_ERR_MAXERRDISP can be registered (see source code).
  If the limit has been reached, this function will signal it by
  returning -1.
 */
/*----------------------------------------------------------------------------*/
int qfits_err_register(qfits_err_dispfunc dispfn)
{
    if (qfits_err_control.n==QFITS_ERR_MAXERRDISP) {
        /* Cannot register any more function */
        return -1;
    }
    qfits_err_control.disp[qfits_err_control.n] = dispfn;
    qfits_err_control.n ++;
    return 0;
}

