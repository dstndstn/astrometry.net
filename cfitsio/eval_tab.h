/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton interface for Bison's Yacc-like parsers in C

   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* Tokens.  */
#ifndef FFTOKENTYPE
# define FFTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum fftokentype {
     BOOLEAN = 258,
     LONG = 259,
     DOUBLE = 260,
     STRING = 261,
     BITSTR = 262,
     FUNCTION = 263,
     BFUNCTION = 264,
     GTIFILTER = 265,
     REGFILTER = 266,
     COLUMN = 267,
     BCOLUMN = 268,
     SCOLUMN = 269,
     BITCOL = 270,
     ROWREF = 271,
     NULLREF = 272,
     SNULLREF = 273,
     OR = 274,
     AND = 275,
     NE = 276,
     EQ = 277,
     GTE = 278,
     LTE = 279,
     LT = 280,
     GT = 281,
     POWER = 282,
     NOT = 283,
     FLTCAST = 284,
     INTCAST = 285,
     UMINUS = 286,
     DIFF = 287,
     ACCUM = 288
   };
#endif
/* Tokens.  */
#define BOOLEAN 258
#define LONG 259
#define DOUBLE 260
#define STRING 261
#define BITSTR 262
#define FUNCTION 263
#define BFUNCTION 264
#define GTIFILTER 265
#define REGFILTER 266
#define COLUMN 267
#define BCOLUMN 268
#define SCOLUMN 269
#define BITCOL 270
#define ROWREF 271
#define NULLREF 272
#define SNULLREF 273
#define OR 274
#define AND 275
#define NE 276
#define EQ 277
#define GTE 278
#define LTE 279
#define LT 280
#define GT 281
#define POWER 282
#define NOT 283
#define FLTCAST 284
#define INTCAST 285
#define UMINUS 286
#define DIFF 287
#define ACCUM 288




#if ! defined FFSTYPE && ! defined FFSTYPE_IS_DECLARED
typedef union FFSTYPE
#line 181 "eval.y"
{
    int    Node;        /* Index of Node */
    double dbl;         /* real value    */
    long   lng;         /* integer value */
    char   log;         /* logical value */
    char   str[256];    /* string value  */
}
/* Line 1489 of yacc.c.  */
#line 123 "y.tab.h"
	FFSTYPE;
# define ffstype FFSTYPE /* obsolescent; will be withdrawn */
# define FFSTYPE_IS_DECLARED 1
# define FFSTYPE_IS_TRIVIAL 1
#endif

extern FFSTYPE fflval;

