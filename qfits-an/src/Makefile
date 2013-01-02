
all:

BASEDIR := ../..
COMMON := $(BASEDIR)/util
ifneq ($(MAKECMDGOALS),clean)
  include $(COMMON)/makefile.common
endif

LDFLAGS := $(LDFLAGS_DEF)
LDFLAGS += -fPIC
CFLAGS += $(CFLAGS_DEF)


LIB := libqfits.a

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

INSTALL_DIR ?= ..

install: $(LIB)
	$(CP) $(LIB) '$(INSTALL_DIR)/lib'
	$(CP) $(HEADERS) '$(INSTALL_DIR)/include'
.PHONY: install

# Programs we use
RANLIB := ranlib
CP := cp

STAGED_LIB_DIR := $(CURDIR)/../lib
STAGED_INCLUDE_DIR := $(CURDIR)/../include

STAGED_LIB := $(STAGED_LIB_DIR)/$(LIB)
STAGED_HEADERS := $(addprefix $(STAGED_INCLUDE_DIR)/, $(HEADERS))

stage: $(STAGED_LIB) $(STAGED_HEADERS)
.PHONY: stage

all: stage

$(STAGED_LIB): $(LIB)
	$(CP) $< $@

$(STAGED_HEADERS):: $(STAGED_INCLUDE_DIR)/%: %
	$(CP) $< $@

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
	-rm -f $(LIB) $(OBJS) $(STAGED_LIB) $(STAGED_HEADERS)

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
