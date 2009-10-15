AUTOMAKE_OPTIONS = 1.8 foreign
DISTCLEANFILES = *~

INCLUDES = -I$(top_srcdir)/src

AM_CFLAGS = $(QFITS_CFLAGS)

lib_LTLIBRARIES = libqfits.la

libqfits_la_SOURCES = \
	qfits_byteswap.c \
    qfits_cache.c \
    qfits_card.c \
    qfits_error.c \
    qfits_filename.c \
    qfits_float.c \
    qfits_header.c \
    qfits_image.c \
    qfits_convert.c \
    qfits_memory.c \
    qfits_md5.c \
    qfits_rw.c \
    qfits_table.c \
	qfits_thread_pthreads.c \
    qfits_time.c \
    qfits_tools.c \
    md5.c 

include_HEADERS = qfits_cache.h \
    qfits_filename.h \
    qfits_header.h \
    qfits_image.h \
    qfits_convert.h \
    qfits_memory.h \
    qfits_md5.h \
    qfits_rw.h \
    qfits_std.h \
    qfits_table.h \
    qfits_thread.h \
    qfits_thread_pthreads.h \
    qfits_time.h \
    qfits_tools.h \
    qfits_error.h

noinst_HEADERS = qfits_byteswap.h \
    qfits_card.h \
    qfits_float.h \
    qfits_thread.h \
    qfits_thread_pthreads.h \
    md5.h 

libqfits_la_LDFLAGS = -version-info $(LT_CURRENT):$(LT_REVISION):$(LT_AGE)
libqfits_la_LIBADD =  -lm
