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

BASEDIR := ..
COMMON := $(BASEDIR)/util
DATA := .

all:
.PHONY: all

include $(COMMON)/makefile.common

$(DATA)/%.txt: ;
$(DATA)/%.dat: ;

ngc2000.o: ngc2000.c ngc2000names.c ngc2000entries.c

ngc2000names.c: parse-ngc2000names.awk $(DATA)/ngc2000names.dat
	$(AWK) -f parse-ngc2000names.awk < $(DATA)/ngc2000names.dat  > $@

ngc2000entries.py: parse-ngc2000-py.awk $(DATA)/ngc2000.dat
	$(AWK) -f parse-ngc2000-py.awk < $(DATA)/ngc2000.dat  > $@

ngc2000accurate.py: parse-ngcic-accurate-py.awk $(DATA)/ngc2000_pos.txt $(DATA)/ic2000_pos.txt
	cat $(DATA)/ngc2000_pos.txt $(DATA)/ic2000_pos.txt | $(AWK) -f parse-ngcic-accurate-py.awk > $@

ngc2000.py: ngc2000-pre.py ngc2000entries.py ngc2000-mid.py ngc2000accurate.py ngc2000-post.py
	cat $^ > $@

ngc2000entries.c: parse-ngc2000.awk $(DATA)/ngc2000.dat
	$(AWK) -f parse-ngc2000.awk < $(DATA)/ngc2000.dat  > $@

ngcic-accurate-entries.c: parse-ngcic-accurate.awk $(DATA)/ngc2000_pos.txt $(DATA)/ic2000_pos.txt
	cat $(DATA)/ngc2000_pos.txt $(DATA)/ic2000_pos.txt | $(AWK) -f parse-ngcic-accurate.awk > $@

stellarium-constellations.c:
	-$(MAKE) grab-stellarium-constellations
	-(./grab-stellarium-constellations > $@.tmp  &&  mv $@.tmp $@) || touch $@

grab-stellarium-constellations: grab-stellarium-constellations.o \
		starutil.o mathutil.o bl.o an-endian.o
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS)


OBJS := ngc2000.o ngcic-accurate.o brightstars.o constellations.o

HEADERS := brightstars.h constellations.h ngc2000.h ngcic-accurate.h 

PYTHON_INSTALL := __init__.py ngc2000.py 

PY_INSTALL_DIR := $(INSTALL_DIR)/python/astrometry/catalogs
INCLUDE_INSTALL_DIR := $(INSTALL_DIR)/include
LIB_INSTALL_DIR := $(INSTALL_DIR)/lib
LINK_DIR := ../python/astrometry/catalogs

# MISC_OBJ := grab-stellarium-constellations.o 

DEP_OBJ := $(OBJS)
DEP_PREREQS :=
DEP_PREREQS += ngc2000names.c ngc2000entries.c ngcic-accurate-entries.c

CFLAGS += $(CFLAGS_DEF)

LDFLAGS += $(LDFLAGS_DEF)

LDLIBS := $(LDLIBS_DEF)
LDLIBS += -lm

CFLAGS += -I.
CFLAGS += -I$(COMMON)

LIBCAT := libcatalogs.a

$(LIBCAT): $(OBJS)
	-rm -f $@
	$(AR) rc $@ $(OBJS)
	$(RANLIB) $@

all: ngc2000.py $(LIBCAT)

install: $(PYTHON_INSTALL) $(LIBCAT) $(HEADERS)
	@echo Installing in base directory '$(INSTALL_DIR)'
	mkdir -p '$(PY_INSTALL_DIR)'
	@for x in $(PYTHON_INSTALL); do \
		echo cp $$x '$(PY_INSTALL_DIR)/'$$x; \
		cp $$x '$(PY_INSTALL_DIR)/'$$x; \
	done
	@for x in $(HEADERS); do \
		echo cp $$x '$(INCLUDE_INSTALL_DIR)/'$$x; \
		cp $$x '$(INCLUDE_INSTALL_DIR)/'$$x; \
	done
	@for x in $(LIBCAT); do \
		echo cp $$x '$(LIB_INSTALL_DIR)/'$$x; \
		cp $$x '$(LIB_INSTALL_DIR)/'$$x; \
	done
	@echo ok

.PHONY: install

clean:
	rm -f $(LIBCAT) $(OBJS) $(DEPS) *.dep deps \
		grab-stellarium-constellations

test:
.PHONY: test

ifneq ($(MAKECMDGOALS),clean)
          include $(COMMON)/makefile.deps
endif

