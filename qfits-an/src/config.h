/* Version number of package */
#define VERSION "6.2.0"
/* Define to the version of this package. */
#define PACKAGE_VERSION "6.2.0"


/* Define to 1 if your processor stores words with the most significant byte
   first (like Motorola and SPARC, unlike Intel and VAX). */
/* #undef WORDS_BIGENDIAN */

// MacOSX doesn't have endian.h
#if __APPLE__
# include <sys/types.h>
#elif __FreeBSD__
# include <sys/endian.h>
#else
# include <endian.h>
#endif

#if \
  (defined(__BYTE_ORDER) && (__BYTE_ORDER == __BIG_ENDIAN)) || \
  (defined( _BYTE_ORDER) && ( _BYTE_ORDER ==  _BIG_ENDIAN)) || \
  (defined(  BYTE_ORDER) && (  BYTE_ORDER ==   BIG_ENDIAN))
//#define IS_BIG_ENDIAN 1
#define WORDS_BIGENDIAN 1
#else
#undef WORDS_BIGENDIAN
#endif






/*
#define CPU_X86 686
#define HAVE_ATEXIT 1
// #undef HAVE_DOPRNT
#define HAVE_DLFCN_H 1
#define HAVE_FCNTL_H 1
#define HAVE_GETPAGESIZE 1
#define HAVE_GETTIMEOFDAY 1
#define HAVE_INTTYPES_H 1
#define HAVE_LIBM 1
#define HAVE_MALLOC 1
#define HAVE_MEMCHR 1
#define HAVE_MEMMOVE 1
#define HAVE_MEMORY_H 1
#define HAVE_MEMSET 1
#define HAVE_MKDIR 1
#define HAVE_MMAP 1
#define HAVE_MUNMAP 1
#define HAVE_REALLOC 1
#define HAVE_REGCOMP 1
#define HAVE_RMDIR 1
 // #undef HAVE_STAT_EMPTY_STRING_BUG
#define HAVE_STDINT_H 1
#define HAVE_STDLIB_H 1
#define HAVE_STRCHR 1
#define HAVE_STRDUP 1
#define HAVE_STRINGS_H 1
#define HAVE_STRING_H 1
#define HAVE_STRRCHR 1
#define HAVE_STRSTR 1
#define HAVE_SYS_STAT_H 1
#define HAVE_SYS_TIME_H 1
#define HAVE_SYS_TYPES_H 1
#define HAVE_UNAME 1
#define HAVE_UNISTD_H 1
#define HAVE_VPRINTF 1
// #undef LSTAT_FOLLOWS_SLASHED_SYMLINK
#define PACKAGE "qfits"
#define PACKAGE_BUGREPORT "yjung@eso.org"
#define PACKAGE_NAME "qfits"
#define PACKAGE_STRING "qfits 6.2.0"
#define PACKAGE_TARNAME "qfits"
#define PACKAGE_VERSION "6.2.0"
#define STDC_HEADERS 1
#define TIME_WITH_SYS_TIME 1
// #undef TM_IN_SYS_TIME
#define VERSION "6.2.0"
// #undef const
// #undef malloc
// #undef realloc
// #undef size_t
 */
