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
MAKEFILES_DIR := $(BASEDIR)/old-makefiles
DATA := .

all:
.PHONY: all

include $(MAKEFILES_DIR)/makefile.common

include $(MAKEFILES_DIR)/makefile.qfits
include $(MAKEFILES_DIR)/makefile.resample

ANUTILS_OBJ :=  sip-utils.o fit-wcs.o sip.o fitstable.o healpix-utils.o \
	healpix.o sip_qfits.o starutil.o anwcs.o wcs-resample.o gslutils.o

ANUTILS_H := anwcs.h boilerplate.h fit-wcs.h fitstable.h gslutils.h \
	healpix-utils.h healpix.h sip-utils.h sip.h sip_qfits.h starutil.h \
	wcs-resample.h

ANUTILS_DEPS :=

ANUTILS_LIB_FILE := libanutils.a

all: $(ANUTILS_LIB_FILE)

CFLAGS += $(CFLAGS_DEF)
CFLAGS += $(QFITS_INC)
CFLAGS += $(RESAMPLE_INC)
CFLAGS += -I.

%.o: %.c
	$(CC) -o $@ $(CFLAGS) -c $<

LDFLAGS += $(LDFLAGS_DEF)

LDLIBS := $(LDLIBS_DEF)
#LDLIBS += $(ANUTILS_LIB)

ALL_OBJ := $(ANUTILS_OBJ)
DEP_OBJ := $(ANUTILS_OBJ)
DEP_PREREQS :=

$(ANUTILS_LIB_FILE): $(ANUTILS_OBJ) $(ANUTILS_DEPS) 
	-rm -f $@
	$(AR) rc $@ $(ANUTILS_OBJ)
	$(RANLIB) $@

ifneq ($(MAKECMDGOALS),clean)
  ifneq ($(MAKECMDGOALS),reconfig)
    ifneq ($(MAKECMDGOALS),makefile.os-features)
      include $(MAKEFILES_DIR)/makefile.deps
    endif
  endif
endif

clean:
	rm -f $(ANUTILS_LIB_FILE) \
		$(ALL_OBJ) $(DEPS) deps \
		*.o *~ *.dep *$(PYTHON_SO_EXT)

