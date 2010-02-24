
all:

LIB := libqfits.a

all: $(LIB)

# Programs
RANLIB := ranlib
CP := cp

HEADERS := \
	qfits.h \
	qfits_cache.h \
	qfits_filename.h \
	qfits_keywords.h \
	qfits_header.h \
	qfits_image.h \
	qfits_md5.h \
	qfits_rw.h \
	qfits_std.h \
	qfits_table.h \
	qfits_time.h \
	qfits_tools.h \
	qfits_memory.h \
	qfits_error.h \
	qfits_convert.h \
	anqfits.h

install: $(LIB)
	$(CP) $(LIB) ../lib
	$(CP) $(HEADERS) ../include

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

$(LIB): $(OBJS)
	-rm -f $@
	$(AR) rc $@ $(OBJS)
	$(RANLIB) $@

clean:
	-rm $(LIB) $(OBJS) ../lib/$(LIB) ../include/*

#.SUFFIXES:

SUFFIXES:
%: %.o
%: %.c
%: %.cc
%: %.C
%: %.cpp
%: %.p
%: %.f
%: %.F
%: %.m
%: %.r
%: %.s
%: %.S
%: %.mod
%: %.sh
%: RCS/%,v
%: RCS/%
%: s.%
%: %,v
%: SCCS/s.%

%.c: %.y
%.c: %.l
%.c: %.w
%.c: %.w %.ch

%.tex: %.w %.ch
%.tex: %.w
