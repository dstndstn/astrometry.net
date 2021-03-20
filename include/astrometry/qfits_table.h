/* Note: this file has been modified from its original form by the
   Astrometry.net team.  For details see http://astrometry.net */

/* $Id: qfits_table.h,v 1.9 2006/02/20 09:45:25 yjung Exp $
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
 * $Date: 2006/02/20 09:45:25 $
 * $Revision: 1.9 $
 * $Name: qfits-6_2_0 $
 */

#ifndef QFITS_TABLE_H
#define QFITS_TABLE_H

/*-----------------------------------------------------------------------------
                                   Includes
 -----------------------------------------------------------------------------*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
    
#include "astrometry/qfits_header.h"

/*-----------------------------------------------------------------------------
                                   Defines
 -----------------------------------------------------------------------------*/

/* The following defines the maximum acceptable size for a FITS value */
#define FITSVALSZ                    60

#define QFITS_INVALIDTABLE            0
#define QFITS_BINTABLE                1
#define QFITS_ASCIITABLE            2

/*-----------------------------------------------------------------------------
                                   New types
 -----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/**
  @brief    Column data type
 */ 
/*----------------------------------------------------------------------------*/
typedef enum _TFITS_DATA_TYPE_ {
    TFITS_ASCII_TYPE_A,
    TFITS_ASCII_TYPE_D,
    TFITS_ASCII_TYPE_E,
    TFITS_ASCII_TYPE_F,
    TFITS_ASCII_TYPE_I,
    TFITS_BIN_TYPE_A,
    TFITS_BIN_TYPE_B,
    TFITS_BIN_TYPE_C,
    TFITS_BIN_TYPE_D,
    TFITS_BIN_TYPE_E,
    TFITS_BIN_TYPE_I,
    TFITS_BIN_TYPE_J,
    TFITS_BIN_TYPE_K,
    TFITS_BIN_TYPE_L,
    TFITS_BIN_TYPE_M,
    TFITS_BIN_TYPE_P,
    TFITS_BIN_TYPE_X,
    TFITS_BIN_TYPE_UNKNOWN
} tfits_type;

/*----------------------------------------------------------------------------*/
/**
  @brief    Column object

  This structure contains all information needed to read a column in a table.
  These informations come from the header. 
  The qfits_table object contains a list of qfits_col objects.

  This structure has to be created from scratch and filled if one want to 
  generate a FITS table.
 */
/*----------------------------------------------------------------------------*/
typedef struct qfits_col
{
    /** 
      Number of atoms in one field.
     In ASCII tables, it is the number of characters in the field as defined
     in TFORM%d keyword.
     In BIN tables, it is the number of atoms in each field. For type 'A', 
     it is the number of characters. A field with two complex object will
     have atom_nb = 4.
    */
    int            atom_nb;

    /**
     Number of decimals in a ASCII field. 
     This value is always 0 for BIN tables
    */
    int         atom_dec_nb;

    /** 
      Size of one element in bytes. In ASCII tables, atom_size is the size
      of the element once it has been converted in its 'destination' type.
      For example, if "123" is contained in an ASCII table in a column 
      defined as I type, atom_nb=3, atom_size=4.
      In ASCII tables:
       - type 'A' : atom_size = atom_nb = number of chars
       - type 'I', 'F' or 'E' : atom_size = 4
       - type 'D' : atom_size = 8
      In BIN tables :
       - type 'A', 'L', 'X', 'B': atom_size = 1 
       - type 'I' : atom_size = 2
       - type 'E', 'J', 'C', 'P' : atom_size = 4
       - type 'D', 'M' : atom_size = 8
      In ASCII table, there is one element per field. The size in bytes and 
      in number of characters is atom_nb, and the size in bytes after 
      conversion of the field is atom_size.
      In BIN tables, the size in bytes of a field is always atom_nb*atom_size.
     */
    int            atom_size;    
    
    /** 
      Type of data in the column as specified in TFORM keyword 
      In ASCII tables : TFITS_ASCII_TYPE_* with *=A, I, F, E or D 
      In BIN tables : TFITS_BIN_TYPE_* with *=L, X, B, I, J, A, E, D, C, M or P 
    */
    tfits_type    atom_type;

    /** Label of the column */
    char        tlabel[FITSVALSZ];

    /** Unit of the data */
    char        tunit[FITSVALSZ];
    
    /** Null value */
    char        nullval[FITSVALSZ];

    /** Display format */
    char        tdisp[FITSVALSZ];
    
    /** 
      zero and scale are used when the quantity in the field does not     
      represent a true physical quantity. Basically, thez should be used
      when they are present: physical_value = zero + scale * field_value 
      They are read from TZERO and TSCAL in the header
     */
    int            zero_present;    
    float        zero;        
    int            scale_present;
    float        scale;   

    /** Offset between the beg. of the table and the beg. of the column.
     NOTE, THIS IS NOT THE OFFSET FROM THE BEGINNING OF THE *ROW*! */
    int            off_beg;
    
    /** Flag to know if the column is readable. An empty col is not readable */
    int            readable;
} qfits_col;


/*----------------------------------------------------------------------------*/
/**
  @brief    Table object

  This structure contains all information needed to read a FITS table.
  These information come from the header. The object is created by 
  qfits_open().
 
  To read a FITS table, here is a code example:
  @code
  int main(int argc, char* argv[])
  {
      qfits_table     *   table;
     int                    n_ext;
    int                    i;

    // Query the number of extensions
    n_ext = qfits_query_n_ext(argv[1]);
    
    // For each extension
    for (i=0; i<n_ext; i++) {
        // Read all the infos about the current table 
        table = qfits_table_open(argv[1], i+1);
        // Display the current table 
        dump_extension(table, stdout, '|', 1, 1);
    }
    return;
  }
  @endcode
 */
/*----------------------------------------------------------------------------*/
typedef struct qfits_table
{
    /**
        Name of the file the table comes from or it is intended to end to
     */
    char            filename[512];
    /** 
        Table type. 
        Possible values: QFITS_INVALIDTABLE, QFITS_BINTABLE, QFITS_ASCIITABLE
     */
    int                tab_t;
    /** Width in bytes of the table */
    int                tab_w;            
    /** Number of columns */
    int                nc;            
    /** Number of rows */
    int                nr;
    /** Array of qfits_col objects */
    qfits_col    *    col;            
} qfits_table;

/*-----------------------------------------------------------------------------
                               Function prototypes
 -----------------------------------------------------------------------------*/

qfits_table* qfits_table_copy(const qfits_table* t);

qfits_header * qfits_table_prim_header_default(void);
qfits_header * qfits_table_ext_header_default(const qfits_table *);
qfits_table * qfits_table_new(const char *, int, int, int, int);
int qfits_col_fill(qfits_col *, int, int, int, tfits_type, const char *, 
        const char *, const char *, const char *, int, float, int, float, int);

qfits_table * qfits_table_open2(const qfits_header* hdr, off_t offset_beg, size_t data_size,
								const char* filename, int xtnum);

void qfits_table_close(qfits_table *);
unsigned char * qfits_query_column(const qfits_table *, int, const int *);
unsigned char * qfits_query_column_seq(const qfits_table *, int, int, int);
void * qfits_query_column_data(const qfits_table *, int, const int *, 
        const void *);
void * qfits_query_column_seq_data(const qfits_table *, int, int, int, 
        const void *);

int qfits_query_column_seq_to_array_inds(const qfits_table	    *   th,
										 int                 colnum,
										 const int* indices,
										 int Ninds,
										 unsigned char*      destination,
										 int                 dest_stride);

int qfits_query_column_seq_to_array(const qfits_table	    *   th,
									int                 colnum,
									int                 start_ind,
									int                 nb_rows,
									unsigned char*      destination,
									int                 dest_stride);

int qfits_query_column_seq_to_array_no_endian_swap(const qfits_table	    *   th,
												   int                 colnum,
												   int                 start_ind,
												   int                 nb_rows,
												   unsigned char*      destination,
												   int                 dest_stride);

int * qfits_query_column_nulls(const qfits_table *, int, const int *, int *, 
        int *);

/**
  @brief    Compute the table width in bytes from the columns infos 
  @param    th      Allocated qfits_table
  @return   the width (-1 in error case)
 */
/*----------------------------------------------------------------------------*/
int qfits_compute_table_width(const qfits_table * th);

int qfits_table_append_xtension(FILE *, const qfits_table *, const void **);
int qfits_table_append_xtension_hdr(FILE *, const qfits_table *, const void **,
        const qfits_header *);
char * qfits_table_field_to_string(const qfits_table *, int, int, int);

const qfits_col* qfits_table_get_col(const qfits_table* t, int i);

int qfits_table_interpret_type(
        const char  *   str,
        int         *   nb,
        int         *   dec_nb,
        tfits_type  *   type,
        int             table_type);
// thread-safe.
int qfits_is_table_header(const qfits_header* hdr);


#endif
