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

include $(MAKEFILES_DIR)/makefile.anutils
include $(MAKEFILES_DIR)/makefile.libkd


OBJS := codekd.o index.o multiindex.o qidxfile.o quadfile.o rdlist.o \
	scamp-catalog.o scamp.o starkd.o starxy.o tabsort.o wcs-rd2xy.o \
	wcs-xy2rd.o xylist.o matchfile.o matchobj.o

HEADERS := codekd.h index.h multiindex.h qidxfile.h quadfile.h rdlist.h \
	scamp-catalog.h scamp.h starkd.h starxy.h tabsort.h wcs-rd2xy.h \
	wcs-xy2rd.h xylist.h matchfile.h matchobj.h

#HEADERS_PATH := $(addprefix $(INCLUDE_DIR)/,$(HEADERS))

DEP_OBJ := $(OBJS)
DEP_PREREQS :=

CFLAGS += $(CFLAGS_DEF)
CFLAGS += $(ANUTILS_INC)
CFLAGS += $(LIBKD_INC)
CFLAGS += -I.

LDFLAGS += $(LDFLAGS_DEF)

LDLIBS := $(LDLIBS_DEF)
LDLIBS += $(ANUTILS_LIB)
LDLIBS += $(LIBKD_LIB)

LIBANFILES := libanfiles.a

SLIB := $(LIBANFILES)
SLIB += $(ANUTILS_SLIB)
SLIB += $(LIBKD_SLIB)

$(LIBANFILES): $(OBJS)
	-rm -f $@
	$(AR) rc $@ $(OBJS)
	$(RANLIB) $@

all: $(LIBANFILES)

install: $(PYTHON_INSTALL) $(LIBANFILES) $(HEADERS_PATH)
	@echo Installing in base directory '$(INSTALL_DIR)'
	mkdir -p '$(PY_INSTALL_DIR)'
	@for x in $(PYTHON_INSTALL); do \
		echo cp $$x '$(PY_INSTALL_DIR)/'$$x; \
		cp $$x '$(PY_INSTALL_DIR)/'$$x; \
	done
	@for x in $(HEADERS); do \
		echo cp '$(INCLUDE_DIR)/'$$x '$(INCLUDE_INSTALL_DIR)/'$$x; \
		cp '$(INCLUDE_DIR)/'$$x '$(INCLUDE_INSTALL_DIR)/'$$x; \
	done
	@for x in $(LIBANFILES); do \
		echo cp $$x '$(LIB_INSTALL_DIR)/'$$x; \
		cp $$x '$(LIB_INSTALL_DIR)/'$$x; \
	done
	@echo ok

.PHONY: install

# ALL_TEST_FILES = test_tycho2 test_usnob test_nomad test_2mass test_hd \
# 	test_boundaries
# ALL_TEST_EXTRA_OBJS = 
# ALL_TEST_LIBS = $(SLIB)
# ALL_TEST_EXTRA_LDFLAGS = 
# include $(MAKEFILES_DIR)/makefile.tests
# 
# $(ALL_TEST_FILES): $(SLIB)
# 
# tests: $(ALL_TEST_FILES)
# .PHONY: tests

clean:
	rm -f $(LIBANFILES) $(OBJS) $(ALL_OBJ) $(DEPS) *.dep deps \

ifneq ($(MAKECMDGOALS),clean)
          include $(MAKEFILES_DIR)/makefile.deps
endif

