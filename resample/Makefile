# This file is part of the Astrometry.net suite.
# Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.
# Copyright 2008, 2009, 2010, 2011, 2012 Dustin Lang.
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

BASEDIR := ..
MAKEFILES_DIR := ../makefiles

all:
.PHONY: all

include $(MAKEFILES_DIR)/makefile.common
include $(MAKEFILES_DIR)/makefile.anbase

INCLUDE_DIR := .

RESAMPLE_OBJ := convolve-image.o histogram.o histogram2d.o resample.o

RESAMPLE_H := convolve-image.o histogram.h histogram2d.h resample.h

RESAMPLE_DEPS :=

RESAMPLE_LIB_FILE := libresample.a

all: $(RESAMPLE_LIB_FILE)

CFLAGS += $(CFLAGS_DEF)
CFLAGS += -I$(ANBASE_DIR)
CFLAGS += -I.

%.o: %.c
	$(CC) -o $@ $(CFLAGS) -c $<

LDFLAGS += $(LDFLAGS_DEF)

LDLIBS := $(LDLIBS_DEF)
#LDLIBS += $(RESAMPLE_LIB)

ALL_OBJ := $(RESAMPLE_OBJ)
DEP_OBJ := $(RESAMPLE_OBJ)
DEP_PREREQS :=

$(RESAMPLE_LIB_FILE): $(RESAMPLE_OBJ) $(RESAMPLE_DEPS) 
	-rm -f $@
	$(AR) rc $@ $(RESAMPLE_OBJ)
	$(RANLIB) $@

ifneq ($(MAKECMDGOALS),clean)
  ifneq ($(MAKECMDGOALS),reconfig)
    ifneq ($(MAKECMDGOALS),makefile.os-features)
      include $(MAKEFILES_DIR)/makefile.deps
    endif
  endif
endif

clean:
	rm -f $(RESAMPLE_LIB_FILE) \
		$(ALL_OBJ) $(DEPS) deps \
		$(MAKEFILES_DIR)/makefile.os-features *.o *~ *.dep *$(PYTHON_SO_EXT) \
		os-features.log os-features-makefile.log report.txt
