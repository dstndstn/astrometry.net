
all:

LIB := libqfits.a

all: $(LIB)

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

$(LIB): $(OBJS)
	-rm -f $@
	$(AR) rc $@ $(OBJS)
	$(RANLIB) $@

clean:
	-rm $(LIB) $(OBJS)

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
