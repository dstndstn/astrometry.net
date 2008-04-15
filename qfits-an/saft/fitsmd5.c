/* $Id: fitsmd5.c,v 1.7 2006/02/17 10:26:07 yjung Exp $
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
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

/*
 * $Author: yjung $
 * $Date: 2006/02/17 10:26:07 $
 * $Revision: 1.7 $
 * $Name: qfits-6_2_0 $
 */

/*-----------------------------------------------------------------------------
                                Includes
 -----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>

/*-----------------------------------------------------------------------------
                Support for gzipped files if linked against zlib.
 -----------------------------------------------------------------------------*/

#if HAVE_ZLIB
#include "zlib.h"

#define FILE            gzFile
#define fopen           gzopen
#define fclose          gzclose
#define fread(b,s,n,f)  gzread(f,b,n*s)

#define GZIP_MAGIC1     0x1f
#define GZIP_MAGIC2     0x8b

static is_gzipped(char * filename)
{
    FILE * in ;
    unsigned char b1, b2 ;
    int r1, r2 ;

    if ((in=fopen(filename, "r"))==NULL) {
        return -1 ;
    }
    r1 = fread(&b1, 1, 1, in);
    r2 = fread(&b2, 1, 1, in);
    fclose(in);
    if (r1!=1 || r2!=1)
        return 0 ;
    if ((int)b1!=GZIP_MAGIC1 || (int)b2!=GZIP_MAGIC2) {
        return 0 ;
    }
    return 1 ;
}
#endif 

/*-----------------------------------------------------------------------------
                                Define
 -----------------------------------------------------------------------------*/

/* Definitions related to FITS */
#define FITSLINESZ        80    /* a FITS line is 80 chars     */
#define FITSCARDS        36    /* 36 cards per block        */
#define FITSBLOCKSZ        (FITSLINESZ*FITSCARDS)    /* FITS block size=2880 */

/* Definitions related to MD5 */
#define MD5HASHSZ        32 /* an MD5 key length is 32 bytes = 128 bits */

/* FITS keyword used to store MD5 key */
#define FITSMD5KEY        "DATAMD5 "
 
/*-----------------------------------------------------------------------------
                                New types
 -----------------------------------------------------------------------------*/

/* The following types defined for MD5 computation only */
typedef unsigned int word32 ;

struct MD5Context {
    word32 buf[4];
    word32 bits[2];
    unsigned char in[64];
};

/*-----------------------------------------------------------------------------
                        Private function prototypes
 -----------------------------------------------------------------------------*/

static void MD5Init(struct MD5Context *);
static void MD5Update(struct MD5Context *, unsigned char *, unsigned);
static void MD5Final(unsigned char *, struct MD5Context *);
static void MD5Transform(word32 *, word32 *);
static void byteReverse(unsigned char *, unsigned);

static int    fits_md5_check(char *, int);
static char * fits_pretty_string(char *);
static char * fits_getvalue(char *);

static void usage(void);

/*-----------------------------------------------------------------------------
                            Global variables
 -----------------------------------------------------------------------------*/

static char * pname = NULL ;
static char prog_desc[] = "Compute/Update the DATAMD5 keyword/value" ;
static int silent_process=0 ;

/*-----------------------------------------------------------------------------
                            MD5 function code
 -----------------------------------------------------------------------------*/

/* Reverse bytes in a 32-bit word. Harmless on little endian machines */
static void byteReverse(unsigned char *buf, unsigned longs)
{
    word32 t;
    do {
        t = (word32) ((unsigned) buf[3] << 8 | buf[2]) << 16 |
            ((unsigned) buf[1] << 8 | buf[0]);
        *(word32 *) buf = t;
        buf += 4;
    } while (--longs);
}

/* Start MD5 accumulation. Set bit count to 0 and buffer to MD5 init. const. */
static void MD5Init(struct MD5Context *ctx)
{
    ctx->buf[0] = 0x67452301;
    ctx->buf[1] = 0xefcdab89;
    ctx->buf[2] = 0x98badcfe;
    ctx->buf[3] = 0x10325476;
    ctx->bits[0] = 0;
    ctx->bits[1] = 0;
}

/* Update to reflect the concatenation of another buffer full of bytes. */
static void MD5Update(struct MD5Context *ctx, unsigned char *buf, unsigned len)
{
    register word32 t;

    /* Update bitcount */
    t = ctx->bits[0];
    if ((ctx->bits[0] = t + ((word32) len << 3)) < t)
    ctx->bits[1]++;        /* Carry from low to high */
    ctx->bits[1] += len >> 29;

    t = (t >> 3) & 0x3f;    /* Bytes already in shsInfo->data */

    /* Handle any leading odd-sized chunks */
    if (t) {
        unsigned char *p = (unsigned char *) ctx->in + t;
        t = 64 - t;
        if (len < t) {
            memmove(p, buf, len);
            return;
        }
        memmove(p, buf, t);
        byteReverse(ctx->in, 16);
        MD5Transform(ctx->buf, (word32 *) ctx->in);
        buf += t;
        len -= t;
    }
    /* Process data in 64-byte chunks */
    while (len >= 64) {
        memmove(ctx->in, buf, 64);
        byteReverse(ctx->in, 16);
        MD5Transform(ctx->buf, (word32 *) ctx->in);
        buf += 64;
        len -= 64;
    }

    /* Handle any remaining bytes of data. */
    memmove(ctx->in, buf, len);
}

/* Final wrapup - pad to 64-byte boundary with the bit pattern 1 0*  */
    /* (64-bit count of bits processed, MSB-first) */
static void MD5Final(unsigned char digest[16], struct MD5Context *ctx)
{
    unsigned int        count ;
    unsigned char   *   p ;

    /* Compute number of bytes mod 64 */
    count = (ctx->bits[0] >> 3) & 0x3F;

    /* Set the first char of padding to 0x80.  This is safe since there is
       always at least one byte free */
    p = ctx->in + count;
    *p++ = 0x80;

    /* Bytes of padding needed to make 64 bytes */
    count = 64 - 1 - count;

    /* Pad out to 56 mod 64 */
    if (count < 8) {
        /* Two lots of padding:  Pad the first block to 64 bytes */
        memset(p, 0, count);
        byteReverse(ctx->in, 16);
        MD5Transform(ctx->buf, (word32 *) ctx->in);

        /* Now fill the next block with 56 bytes */
        memset(ctx->in, 0, 56);
    } else {
        /* Pad block to 56 bytes */
        memset(p, 0, count - 8);
    }
    byteReverse(ctx->in, 14);

    /* Append length in bits and transform */
    ((word32 *) ctx->in)[14] = ctx->bits[0];
    ((word32 *) ctx->in)[15] = ctx->bits[1];

    MD5Transform(ctx->buf, (word32 *) ctx->in);
    byteReverse((unsigned char *) ctx->buf, 4);
    memmove(digest, ctx->buf, 16);
    memset(ctx, 0, sizeof(ctx));    /* In case it's sensitive */
}

/* The four core functions - F1 is optimized somewhat */

/* #define F1(x, y, z) (x & y | ~x & z) */
#define F1(x, y, z) (z ^ (x & (y ^ z)))
#define F2(x, y, z) F1(z, x, y)
#define F3(x, y, z) (x ^ y ^ z)
#define F4(x, y, z) (y ^ (x | ~z))

/* This is the central step in the MD5 algorithm. */
#define MD5STEP(f, w, x, y, z, data, s) \
    ( w += f(x, y, z) + data,  w = w<<s | w>>(32-s),  w += x )

/*
 * The core of the MD5 algorithm, this alters an existing MD5 hash to
 * reflect the addition of 16 longwords of new data.  MD5Update blocks
 * the data and converts bytes into longwords for this routine.
 */
static void MD5Transform(word32 buf[4], word32 in[16])
{
    register word32 a, b, c, d;

    a = buf[0];
    b = buf[1];
    c = buf[2];
    d = buf[3];

    MD5STEP(F1, a, b, c, d, in[0] + 0xd76aa478, 7);
    MD5STEP(F1, d, a, b, c, in[1] + 0xe8c7b756, 12);
    MD5STEP(F1, c, d, a, b, in[2] + 0x242070db, 17);
    MD5STEP(F1, b, c, d, a, in[3] + 0xc1bdceee, 22);
    MD5STEP(F1, a, b, c, d, in[4] + 0xf57c0faf, 7);
    MD5STEP(F1, d, a, b, c, in[5] + 0x4787c62a, 12);
    MD5STEP(F1, c, d, a, b, in[6] + 0xa8304613, 17);
    MD5STEP(F1, b, c, d, a, in[7] + 0xfd469501, 22);
    MD5STEP(F1, a, b, c, d, in[8] + 0x698098d8, 7);
    MD5STEP(F1, d, a, b, c, in[9] + 0x8b44f7af, 12);
    MD5STEP(F1, c, d, a, b, in[10] + 0xffff5bb1, 17);
    MD5STEP(F1, b, c, d, a, in[11] + 0x895cd7be, 22);
    MD5STEP(F1, a, b, c, d, in[12] + 0x6b901122, 7);
    MD5STEP(F1, d, a, b, c, in[13] + 0xfd987193, 12);
    MD5STEP(F1, c, d, a, b, in[14] + 0xa679438e, 17);
    MD5STEP(F1, b, c, d, a, in[15] + 0x49b40821, 22);

    MD5STEP(F2, a, b, c, d, in[1] + 0xf61e2562, 5);
    MD5STEP(F2, d, a, b, c, in[6] + 0xc040b340, 9);
    MD5STEP(F2, c, d, a, b, in[11] + 0x265e5a51, 14);
    MD5STEP(F2, b, c, d, a, in[0] + 0xe9b6c7aa, 20);
    MD5STEP(F2, a, b, c, d, in[5] + 0xd62f105d, 5);
    MD5STEP(F2, d, a, b, c, in[10] + 0x02441453, 9);
    MD5STEP(F2, c, d, a, b, in[15] + 0xd8a1e681, 14);
    MD5STEP(F2, b, c, d, a, in[4] + 0xe7d3fbc8, 20);
    MD5STEP(F2, a, b, c, d, in[9] + 0x21e1cde6, 5);
    MD5STEP(F2, d, a, b, c, in[14] + 0xc33707d6, 9);
    MD5STEP(F2, c, d, a, b, in[3] + 0xf4d50d87, 14);
    MD5STEP(F2, b, c, d, a, in[8] + 0x455a14ed, 20);
    MD5STEP(F2, a, b, c, d, in[13] + 0xa9e3e905, 5);
    MD5STEP(F2, d, a, b, c, in[2] + 0xfcefa3f8, 9);
    MD5STEP(F2, c, d, a, b, in[7] + 0x676f02d9, 14);
    MD5STEP(F2, b, c, d, a, in[12] + 0x8d2a4c8a, 20);

    MD5STEP(F3, a, b, c, d, in[5] + 0xfffa3942, 4);
    MD5STEP(F3, d, a, b, c, in[8] + 0x8771f681, 11);
    MD5STEP(F3, c, d, a, b, in[11] + 0x6d9d6122, 16);
    MD5STEP(F3, b, c, d, a, in[14] + 0xfde5380c, 23);
    MD5STEP(F3, a, b, c, d, in[1] + 0xa4beea44, 4);
    MD5STEP(F3, d, a, b, c, in[4] + 0x4bdecfa9, 11);
    MD5STEP(F3, c, d, a, b, in[7] + 0xf6bb4b60, 16);
    MD5STEP(F3, b, c, d, a, in[10] + 0xbebfbc70, 23);
    MD5STEP(F3, a, b, c, d, in[13] + 0x289b7ec6, 4);
    MD5STEP(F3, d, a, b, c, in[0] + 0xeaa127fa, 11);
    MD5STEP(F3, c, d, a, b, in[3] + 0xd4ef3085, 16);
    MD5STEP(F3, b, c, d, a, in[6] + 0x04881d05, 23);
    MD5STEP(F3, a, b, c, d, in[9] + 0xd9d4d039, 4);
    MD5STEP(F3, d, a, b, c, in[12] + 0xe6db99e5, 11);
    MD5STEP(F3, c, d, a, b, in[15] + 0x1fa27cf8, 16);
    MD5STEP(F3, b, c, d, a, in[2] + 0xc4ac5665, 23);

    MD5STEP(F4, a, b, c, d, in[0] + 0xf4292244, 6);
    MD5STEP(F4, d, a, b, c, in[7] + 0x432aff97, 10);
    MD5STEP(F4, c, d, a, b, in[14] + 0xab9423a7, 15);
    MD5STEP(F4, b, c, d, a, in[5] + 0xfc93a039, 21);
    MD5STEP(F4, a, b, c, d, in[12] + 0x655b59c3, 6);
    MD5STEP(F4, d, a, b, c, in[3] + 0x8f0ccc92, 10);
    MD5STEP(F4, c, d, a, b, in[10] + 0xffeff47d, 15);
    MD5STEP(F4, b, c, d, a, in[1] + 0x85845dd1, 21);
    MD5STEP(F4, a, b, c, d, in[8] + 0x6fa87e4f, 6);
    MD5STEP(F4, d, a, b, c, in[15] + 0xfe2ce6e0, 10);
    MD5STEP(F4, c, d, a, b, in[6] + 0xa3014314, 15);
    MD5STEP(F4, b, c, d, a, in[13] + 0x4e0811a1, 21);
    MD5STEP(F4, a, b, c, d, in[4] + 0xf7537e82, 6);
    MD5STEP(F4, d, a, b, c, in[11] + 0xbd3af235, 10);
    MD5STEP(F4, c, d, a, b, in[2] + 0x2ad7d2bb, 15);
    MD5STEP(F4, b, c, d, a, in[9] + 0xeb86d391, 21);

    buf[0] += a;
    buf[1] += b;
    buf[2] += c;
    buf[3] += d;
}


/*-----------------------------------------------------------------------------
                            FITS-related functions
 -----------------------------------------------------------------------------*/

/* Pretty-print a FITS string value */
static char * fits_pretty_string(char * s)
{
    static char     pretty[FITSLINESZ+1] ;
    int             i,j ;

    if (s==NULL) return NULL ;

    pretty[0] = (char)0 ;
    if (s[0]!='\'') return s ;

    /* skip first quote */
    i=1 ;
    j=0 ;
    /* trim left-side blanks */
    while (s[i]==' ') {
        if (i==(int)strlen(s)) break ;
        i++ ;
    }
    if (i>=(int)(strlen(s)-1)) return pretty ;
    /* copy string, changing double quotes to single ones */
    while (i<(int)strlen(s)) {
        if (s[i]=='\'') i++ ;
        pretty[j]=s[i];
        i++ ;
        j++ ;
    }
    /* NULL-terminate the pretty string */
    pretty[j+1]=(char)0;
    /* trim right-side blanks */
    j = (int)strlen(pretty)-1;
    while (pretty[j]==' ') j-- ;
    pretty[j+1]=(char)0;
    return pretty;
}

/* Get the FITS value in a FITS card */
static char * fits_getvalue(char * line)
{
    static char value[FITSLINESZ+1] ;
    int         from, to ;
    int         inq ;
    int         i ;

    if (line==NULL) return NULL ;
    memset(value, 0, FITSLINESZ+1);
    /* Get past the keyword */
    i=0 ;
    while (line[i]!='=' && i<FITSLINESZ) i++ ;
    if (i>FITSLINESZ) return NULL ;
    i++ ;
    while (line[i]==' ' && i<FITSLINESZ) i++ ;
    if (i>FITSLINESZ) return NULL ;
    from=i;
    /* Now in the value section */
    /* Look for the first slash '/' outside of a string */
    inq = 0 ;
    while (i<FITSLINESZ) {
        if (line[i]=='\'') inq=!inq ;
        if (line[i]=='/') if (!inq) break ;
        i++;
    }
    i-- ;
    /* Backtrack on blanks */
    while (line[i]==' ' && i>=0) i-- ;
    if (i<0) return NULL ;
    to=i ;
    if (to<from) return NULL ;
    /* Copy relevant characters into output buffer */
    strncpy(value, line+from, to-from+1);
    /* Null-terminate the string */
    value[to-from+1] = (char)0;
    /*
     * Make it pretty: remove head and tail quote, change double
     * quotes to simple ones.
     */
    strcpy(value, fits_pretty_string(value));
    return value ;
}

/* Replace the MD5 card in the input header */
static int fits_replace_card(char * filename, int off_md5, char * datamd5)
{
    char    *    buf ;
    int            fd ;
    struct stat    sta ;
    char        card[FITSLINESZ];
    int            i ;
    int            err ;

    /* Get file size */
    if (stat(filename, &sta)==-1) {
        fprintf(stderr, "%s: cannot stat file [%s]: no update done\n",
                pname,
                filename);
        return 1 ;
    }
    /* Open file */
    fd = open(filename, O_RDWR);
    if (fd==-1) {
        fprintf(stderr,
                "%s: cannot open file [%s] for modification: no update done\n",
                pname,
                filename);
        return 1 ;
    }
    /* Memory-map the file */
    buf = (char*)mmap(0,
                      sta.st_size,
                      PROT_READ | PROT_WRITE,
                      MAP_SHARED,
                      fd,
                      0);
    if (buf==(char*)-1 || buf==NULL) {
        perror("mmap");
        close(fd);
        return 1 ;
    }
    /* sprintf should be safe, the MD5 signature size is 32 chars */
    sprintf(card, "%s= '%s' / data MD5 signature", FITSMD5KEY, datamd5);
    i=FITSLINESZ-1 ;
    while (card[i]!='e') {
        card[i]=' ';
        i-- ;
    }
    /* Copy card into file */
    memcpy(buf+off_md5, card, FITSLINESZ);
    /* flush output, unmap buffer, close file and quit */
    err=0 ;
    sync();
    if (close(fd)==-1) {
        fprintf(stderr, "%s: error closing modified file [%s]",
                pname,
                filename);
        err++ ;
    }
    if (munmap(buf, sta.st_size)==-1) {
        perror("munmap");
        err++ ;
    }
    return err;
}

/* Display or modify the DATAMD5 value. Returns the number of errors. */
static int fits_md5_check(char * filename, int update_header)
{
    FILE        *    in ;
    char            buf[FITSBLOCKSZ];
    char    *        buf_c ;
    int                i ;
    int                in_header ;
    char        *    hdrmd5 ;
    struct MD5Context    ctx ;
    unsigned char    digest[16] ;
    static char        datamd5[MD5HASHSZ+1];
    int                off_md5 ;
    int                cur_off ;
    int                md5keysz ;
    int                err ;
    int                check_fits ;
    struct stat        sta ;

    if (filename==NULL) return 1 ;

    /* Try to stat file */
    if (stat(filename, &sta)!=0) {
        fprintf(stderr, "%s: cannot stat file %s\n", pname, filename);
        return 1 ;
    }
    /* See if this is a regular file */
    if (!S_ISREG(sta.st_mode)) {
        fprintf(stderr, "%s: not a regular file: %s\n", pname, filename);
        return 1 ;
    }
    /* Open input file */
    if ((in=fopen(filename, "r"))==NULL) {
        fprintf(stderr, "%s: cannot open file [%s]\n", pname, filename);
        return 1 ;
    }
    /* Initialize all variables */
    MD5Init(&ctx);
    in_header=1 ;
    hdrmd5=NULL ;
    off_md5=0;
    cur_off=0;
    md5keysz = (int)strlen(FITSMD5KEY) ;
    check_fits=0 ;
    /* Loop over input file */
    while (fread(buf, 1, FITSBLOCKSZ, in)==FITSBLOCKSZ) {
        /* First time in the loop: check the file is FITS */
        if (check_fits==0) {
            /* Examine first characters in block */
            if (buf[0]!='S' ||
                buf[1]!='I' ||
                buf[2]!='M' ||
                buf[3]!='P' ||
                buf[4]!='L' ||
                buf[5]!='E' ||
                buf[6]!=' ' ||
                buf[7]!=' ' ||
                buf[8]!='=') {
                fprintf(stderr, "%s: file [%s] is not FITS\n",
                        pname,
                        filename);
                fclose(in);
                return 1 ;
            } else {
                check_fits=1 ;
            }
        }
        /* If current block is a header block */
        if (in_header) {
            buf_c = buf ;
            for (i=0 ; i<FITSCARDS ; i++) {
                /* Try to locate MD5 keyword if not located already */
                if (hdrmd5==NULL) {
                    if (!strncmp(buf_c, FITSMD5KEY, md5keysz)) {
                        hdrmd5 = fits_getvalue(buf_c) ;
                        off_md5 = cur_off ;
                    }
                }
                /* Try to locate an END key */
                if (buf_c[0]=='E' &&
                    buf_c[1]=='N' &&
                    buf_c[2]=='D' &&
                    buf_c[3]==' ') {
                    in_header=0 ;
                    break ;
                }
                buf_c += FITSLINESZ ;
                cur_off += FITSLINESZ ;
            }
        } else {
            /* If current block is a data block */
            /* Try to locate an extension header */
            if (buf[0]=='X' &&
                buf[1]=='T' &&
                buf[2]=='E' &&
                buf[3]=='N' &&
                buf[4]=='S' &&
                buf[5]=='I' &&
                buf[6]=='O' &&
                buf[7]=='N' &&
                buf[8]=='=') {
                in_header=1 ;
                buf_c = buf ;
                for (i=0 ; i<FITSCARDS ; i++) {
                    /* Try to find an END marker in this block */
                    if (buf_c[0]=='E' &&
                        buf_c[1]=='N' &&
                        buf_c[2]=='D' &&
                        buf_c[3]==' ') {
                        /* Found END marker in same block as XTENSION */
                        in_header=0 ;
                        break ;
                    }
                    buf_c += FITSLINESZ ;
                }
            } else {
                /* Data block: accumulate for MD5 */
                MD5Update(&ctx, (unsigned char *)buf, FITSBLOCKSZ);
            }
        }
    }
    fclose(in);
    if (check_fits==0) {
        /* Never went through the read loop: file is not FITS */
        fprintf(stderr, "%s: file [%s] is not FITS\n",
                pname,
                filename);
        return 1 ;
    }
    /* Got to the end of file: summarize */
    MD5Final(digest, &ctx);

    /* Write digest into a string */
    sprintf(datamd5,
    "%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x",
    digest[ 0], digest[ 1], digest[ 2], digest[ 3], digest[ 4],
    digest[ 5], digest[ 6], digest[ 7], digest[ 8], digest[ 9],
    digest[10], digest[11], digest[12], digest[13], digest[14],
    digest[15]);
    if (!silent_process) {
        printf("%s  %s", datamd5, filename);
        if (hdrmd5) {
            if (!strcmp(hdrmd5, datamd5)) {
                printf(" (header Ok)");
            } else {
                printf(" (header is wrong)");
            }
        }
        printf("\n");
    }
    /* Update header if requested */
    err=0 ;
    if (update_header) {
        if (hdrmd5==NULL) {
            fprintf(stderr, "%s: cannot update header: missing %s\n",
                    pname,
                    FITSMD5KEY);
            return 1 ;
        }
#if HAVE_ZLIB
        if (is_gzipped(filename)) {
            fprintf(stderr, "%s: cannot update header in gzipped file\n");
            return 1 ;
        }
#endif
        err = fits_replace_card(filename, off_md5, datamd5);
    }
    return err ;
}


/* Compute MD5 sum on the whole file and print out results on stdout */
static int compute_md5(char * filename)
{
    struct MD5Context    ctx ;
    unsigned char        digest[16] ;
    struct stat            sta ;
    int                    fd ;
    unsigned char    *    buf ;

    /* Try to stat file */
    if (stat(filename, &sta)!=0) {
        fprintf(stderr, "%s: cannot stat file %s\n", pname, filename);
        return 1 ;
    }
    /* See if this is a regular file */
    if (!S_ISREG(sta.st_mode)) {
        fprintf(stderr, "%s: not a regular file: %s\n", pname, filename);
        return 1 ;
    }
    /* Open file */
    if ((fd = open(filename, O_RDONLY))==-1) {
        fprintf(stderr, "%s: cannot open file %s\n", pname, filename);
        return 1 ;
    }
    /* Memory-map the file */
    buf = (unsigned char*)mmap(0, sta.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (buf==(unsigned char*)-1 || buf==NULL) {
        perror("mmap");
        close(fd);
        return 1 ;
    }
    /* Initialize MD5 context */
    MD5Init(&ctx);
    /* Compute MD5 on all bits in the file */
    MD5Update(&ctx, buf, sta.st_size);
    /* Finalize and print results */
    close(fd);
    munmap((char*)buf, sta.st_size);
    MD5Final(digest, &ctx);
    printf(
    "%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x  %s\n",
    digest[ 0], digest[ 1], digest[ 2], digest[ 3], digest[ 4],
    digest[ 5], digest[ 6], digest[ 7], digest[ 8], digest[ 9],
    digest[10], digest[11], digest[12], digest[13], digest[14],
    digest[15],
    filename);
    return 0 ;
}

static void usage(void)
{
    printf(
        "%s -- %s\n"
        "version: $Revision: 1.7 $\n",
        pname,
        prog_desc);
    printf(
        "\n"
        "use : %s [-u] [-s] [-a] <FITS files...>\n"
        "options are:\n"
        "\t-u   update MD5 keyword in the file: %s\n"
        "\t-s   silent mode\n"
        "\n"
        "\t-a   compute MD5 sum of the complete file (incl.header)\n"
        "\n"
        "This utility computes the MD5 checksum of all data sections\n"
        "in a given FITS file, and compares it against the value\n"
        "declared in DATAMD5 if present. It can also update the value\n"
        "of this keyword (if present) with its own computed MD5 sum.\n"
        "\n", pname, FITSMD5KEY) ;
    printf(
        "You can also use it with the -a option to compute the MD5 sum\n"
        "on the complete file (all bits). In this case, the file needs\n"
        "not be FITS. This option is only provided to check this program\n"
        "against other MD5 computation tools.\n"
        "NB: Other options cannot be used together with -a.\n"
        "\n");

#if HAVE_ZLIB
    printf(
        "\n"
        "This program was compiled against zlib %s\n"
        "which allows to process gzipped FITS files\n"
        "as if they were normal FITS files.\n"
        "Notice that you cannot use the -u option on\n"
        "gzipped files, though.\n"
        "\n"
        "\n",
        ZLIB_VERSION
          );
#endif
    exit(0) ;
}

/*-----------------------------------------------------------------------------
                                    Main 
 -----------------------------------------------------------------------------*/
int main(int argc, char *argv[])
{
    int            i ;
    int            update_header ;
    int            total_md5 ;
    int            err ;

    /* Initialize */
    pname=argv[0];
    update_header = 0 ;
    total_md5 = 0 ;

    if (argc<2) usage();
    
    /* Parse arguments for options */
    for (i=1 ; i<argc ; i++) {
        if (!strcmp(argv[i], "-u")) {
            update_header=1 ;
        } else if (!strcmp(argv[i], "-s")) {
            silent_process=1;
        } else if (!strcmp(argv[i], "-a")) {
            total_md5=1 ;
        }
    }
    /* Loop on input file names */
    err=0 ;
    for (i=1 ; i<argc ; i++) {
        /* If not a command-line option */
        if (strcmp(argv[i], "-u") &&
            strcmp(argv[i], "-s") &&
            strcmp(argv[i], "-a")) {
            /* Launch MD5 process on this file */
            if (total_md5) {
                err+=compute_md5(argv[i]);
            } else {
                err += fits_md5_check(argv[i], update_header);
            }
        }
    }
    if (err>0) {
        fprintf(stderr, "%s: %d error(s) during process\n", pname, err);
    }
    return err ;
}
