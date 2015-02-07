# This file is part of the Astrometry.net suite.
# Copyright 2013 Dustin Lang
#
# The Astrometry.net suite is free software; you can redistribute
# it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, version 2.
#
# The Astrometry.net suite is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with the Astrometry.net suite ; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA

BASEDIR := ../..
MAKEFILES_DIR := $(BASEDIR)/makefiles
DATA := .

all:
.PHONY: all

include $(MAKEFILES_DIR)/makefile.common

include $(MAKEFILES_DIR)/makefile.blind
include $(MAKEFILES_DIR)/makefile.png
include $(MAKEFILES_DIR)/makefile.cairo
include $(MAKEFILES_DIR)/makefile.jpeg
include $(MAKEFILES_DIR)/makefile.netpbm

ifneq ($(MAKECMDGOALS),clean)
    include $(MAKEFILES_DIR)/makefile.os-features
endif

CFLAGS := $(CFLAGS_DEF)
CFLAGS += $(BLIND_CFLAGS)

INC := $(BLIND_INC)
INC += $(CAIRO_INC)
INC += $(PNG_INC)
INC += $(JPEG_INC)
INC += $(NETPBM_INC)

CFLAGS += $(INC)

LDFLAGS := $(LDFLAGS_DEF)

LDLIBS := $(LDLIBS_DEF)
LDLIBS += $(BLIND_LIB)
LDLIBS += $(CAIRO_LIB)
LDLIBS += $(PNG_LIB)
LDLIBS += $(JPEG_LIB)
LDLIBS += $(NETPBM_LIB)

LIBPLOTSTUFF := libplotstuff.a

SLIB := $(LIBPLOTSTUFF)
SLIB += $(BLIND_SLIB)

OBJS := plotstuff.o plotfill.o plotxy.o plotimage.o \
	plotannotations.o plotgrid.o plotoutline.o plotindex.o plotradec.o \
	plothealpix.o plotmatch.o cairoutils.o

PROGS := plotstuff

$(LIBPLOTSTUFF): $(OBJS)
	-rm -f $@
	$(AR) rc $@ $(OBJS)
	$(RANLIB) $@

all: $(PROGS) $(LIBPLOTSTUFF)

$(PROGS): %: %-main.o %.o $(SLIB)
ALL_OBJ += $(addsuffix .o,$(PROGS))

