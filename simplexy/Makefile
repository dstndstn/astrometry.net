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
MAKEFILES_DIR := ../old-makefiles

all:
.PHONY: all

include $(MAKEFILES_DIR)/makefile.common
include $(MAKEFILES_DIR)/makefile.anbase
include $(MAKEFILES_DIR)/makefile.qfits
include $(MAKEFILES_DIR)/makefile.resample

INCLUDE_DIR := .

SIMPLEXY_OBJ := ctmf.o dallpeaks.o dcen3x3.o dfind.o dmedsmooth.o \
	dobjects.o dpeaks.o dselip.o dsigma.o dsmooth.o image2xy.o radix.o \
	simplexy.o

SIMPLEXY_H := ctmf.h dimage.h image2xy-files.h image2xy.h radix.h \
	simplexy-common.h simplexy.h

SIMPLEXY_DEPS :=

SIMPLEXY_LIB_FILE := libsimplexy.a

all: $(SIMPLEXY_LIB_FILE)

INC := $(ANBASE_INC)
INC += $(RESAMPLE_INC)
INC += $(QFITS_INC)

CFLAGS += $(CFLAGS_DEF)
CFLAGS += -I$(ANBASE_DIR)
CFLAGS += $(INC)
CFLAGS += -I.

%.o: %.c
	$(CC) -o $@ $(CFLAGS) -c $<

LDFLAGS += $(LDFLAGS_DEF)

LDLIBS := $(LDLIBS_DEF)
#LDLIBS += $(SIMPLEXY_LIB)

ALL_OBJ := $(SIMPLEXY_OBJ)
DEP_OBJ := $(SIMPLEXY_OBJ)
DEP_PREREQS :=

$(SIMPLEXY_LIB_FILE): $(SIMPLEXY_OBJ) $(SIMPLEXY_DEPS) 
	-rm -f $@
	$(AR) rc $@ $(SIMPLEXY_OBJ)
	$(RANLIB) $@

ifneq ($(MAKECMDGOALS),clean)
  ifneq ($(MAKECMDGOALS),reconfig)
    ifneq ($(MAKECMDGOALS),makefile.os-features)
      include $(MAKEFILES_DIR)/makefile.deps
    endif
  endif
endif

clean:
	rm -f $(SIMPLEXY_LIB_FILE) \
		$(ALL_OBJ) $(DEPS) deps \
		$(MAKEFILES_DIR)/makefile.os-features *.o *~ *.dep *$(PYTHON_SO_EXT) \
		os-features.log os-features-makefile.log report.txt
