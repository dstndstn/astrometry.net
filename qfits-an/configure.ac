#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.59)
AC_INIT(qfits, 6.2.0, yjung@eso.org)

AC_CONFIG_SRCDIR([Makefile.am])
AC_CONFIG_HEADER([config.h])
AC_CONFIG_AUX_DIR([admin])
AM_INIT_AUTOMAKE

# Checks for programs.
AC_PROG_CC

# AC_PROG_RANLIB for static libraries without Libtool
# AC_PROG_RANLIB

# AC_PROG_LIBTOOL for static and shared libraries with Libtool
AC_PROG_LIBTOOL
AC_ENABLE_STATIC([yes])
AC_ENABLE_SHARED([yes])
AC_SUBST(LIBTOOL_DEPS)

# Doxygen and LAtex    
ESO_CHECK_DOCTOOLS

# Libtool versioning. First release 0.0.0
LT_CURRENT=0
LT_REVISION=0
LT_AGE=0
AC_SUBST([LT_CURRENT])
AC_SUBST([LT_REVISION])
AC_SUBST([LT_AGE]) 

# Checks for libraries.
# FIXME: Replace `main' with a function in `-lm':
AC_CHECK_LIB([m], [main])
# FIXME: Replace `main' with a function in `-lqfits':
# In comments next line created by autscan, we do not want to 
# check qfits lib, we want to create it.
#AC_CHECK_LIB([qfits], [main])

# Defined in the m4macros/qfits_memory.m4
QFITS_CHECK_MEMORY

# Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([fcntl.h stdlib.h string.h sys/time.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_CONST
AC_TYPE_SIZE_T
AC_HEADER_TIME
AC_STRUCT_TM

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_MMAP
AC_FUNC_REALLOC
AC_FUNC_STAT
AC_FUNC_VPRINTF
AC_CHECK_FUNCS([atexit getpagesize gettimeofday memchr memmove memset mkdir munmap regcomp rmdir strchr strdup strrchr strstr uname])

# CG. Check for the BIG_ENDIAN (needed by qfits sources)
AC_C_BIGENDIAN

case $host_cpu in
i*86)
    CPU_X86="`echo $host_cpu | cut -c 2-`"
    AC_DEFINE_UNQUOTED([CPU_X86], $CPU_X86, [x86 CPU type])
    ;;
esac

AC_CONFIG_FILES([Makefile
                doc/Doxyfile
                src/Makefile
                main/Makefile
                saft/Makefile
                man/Makefile
                test/Makefile])
AC_OUTPUT
