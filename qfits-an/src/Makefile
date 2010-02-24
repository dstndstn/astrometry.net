
all: libqfits.a

RANLIB := ranlib

OBJS := \
	anqfits.o \
	md5.o \
	qfits_byteswap.o \
	qfits_cache.o \
	qfits_card.o \
	qfits_convert.o \
	qfits_error.o \
	qfits_filename.o \
	qfits_float.o \
	qfits_header.o \
	qfits_image.o \
	qfits_md5.o \
	qfits_memory.o \
	qfits_rw.o \
	qfits_table.o \
	qfits_thread_pthreads.o \
	qfits_time.o \
	qfits_tools.o

libqfits.a: $(OBJS)
	-rm -f $@
	$(AR) rc $@ $(OBJS)
	$(RANLIB) $@
