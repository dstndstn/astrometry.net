
# This file is part of the Astrometry.net suite.
# Copyright 2006-2008 Dustin Lang, Keir Mierle and Sam Roweis.
# Copyright 2010, 2011, 2012, 2013 Dustin Lang.
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

# To set the install directory:
#   make install INSTALL_DIR=/path/to/dir
# or see util/makefile.common

# Turn off optimisation?  If the following line is commented-out, the default
# is to turn optimization on.  See util/makefile.common for details.
#export OPTIMIZE = no

all:

BASEDIR := .
COMMON := $(BASEDIR)/util

# The internal Astrometry.net dependency stack, top to bottom, is:
#
#  solver/libastrometry.a  -- astrometry.net core
#    catalogs/libcatalogs.a
#    util/libanfiles.a  -- astrometry.net index files, etc
#      libkd/libkd.a -- kd-trees
#        util/libanutils.a  -- utilities
#          gsl-an/libgsl-an.a OR system gsl -- GNU scientific library
#          [wcslib] -- optional
#          qfits-an/libqfits.a -- FITS files
#            util/libanbase.a  -- basic stuff

# Copy this stuff from makefile.common because it is needed to avoid checking
# implicit rules for makefile.common itself!!
$(COMMON):
$(COMMON)/makefile.common:
# no default rules
.SUFFIXES :=
# Cancel stupid implicit rules.
%: %,v
%: RCS/%,v
%: RCS/%
%: s.%
%: SCCS/s.%

include $(COMMON)/makefile.common

all: subdirs version
.PHONY: all

version:
	echo $(AN_GIT_REVISION) | $(AWK) -F - '{printf "__version__ = \""; if (NF>2) {printf $$1".dev"$$2} else {printf $$1}; print "\""}' > __init__.py
.PHONY: version

check: pkgconfig
.PHONY: check

# Just check that we have pkg-config, since it's needed to get
# wcslib, cfitsio, cairo, etc config information.
pkgconfig:
	pkg-config --version || (echo -e "\nWe require the pkg-config package.\nGet it from http://www.freedesktop.org/wiki/Software/pkg-config" && false)
	pkg-config --modversion cfitsio || (echo -e "\nWe require cfitsio but it was not found.\nGet it from http://heasarc.gsfc.nasa.gov/fitsio/\nOr on Ubuntu/Debian, apt-get install cfitsio-dev\nOr on Mac OS / Homebrew, brew install cfitsio\n" && false)
.PHONY: pkgconfig

# Detect GSL -- this minimum version was chosen to match the version in gsl-an.
# Earlier versions would probably work fine.
SYSTEM_GSL ?= $(shell (pkg-config --atleast-version=1.14 gsl && echo "yes") || echo "no")
# Make this variable visible to recursive "make" calls
export SYSTEM_GSL

ifneq ($(SYSTEM_GSL),yes)
subdirs: gsl-an
solver: gsl-an
endif

SUBDIRS := util catalogs libkd solver sdss qfits-an plot
SUBDIRS_OPT := gsl-an

subdirs: $(SUBDIRS)

util: config qfits-an

libkd: config util

catalogs: config libkd

solver: config util catalogs libkd qfits-an

plot: config util catalogs libkd qfits-an

cfitsio-utils:
	$(MAKE) -C solver cfitsutils

$(SUBDIRS):
	$(MAKE) -C $@
$(SUBDIRS_OPT):
	$(MAKE) -C $@

.PHONY: subdirs $(SUBDIRS) $(SUBDIRS_OPT)

doc:
	$(MAKE) -C doc html
.PHONY: doc
html:
.PHONY: html
	$(MAKE) -C doc html

# Targets that require extra libraries
extra: qfits-an util catalogs plot-extra
.PHONY: extra

plot-extra: cairoutils
	$(MAKE) -C plot cairo
.PHONY: plot-extra

# Targets that create python bindings (requiring swig)
ifneq ($(SYSTEM_GSL),yes)
py: gsl-an
endif
py: catalogs pyutil libkd-spherematch sdss cairoutils pyplotstuff pysolver

libkd-spherematch:
	$(MAKE) -C libkd pyspherematch

cairoutils:
	$(MAKE) -C util cairoutils.o

pyplotstuff: cairoutils
	$(MAKE) -C plot pyplotstuff

pysolver: cairoutils
	$(MAKE) -C solver pysolver

.PHONY: py libkd-spherematch cairoutils pyplotstuff pysolver

pyutil: qfits-an
	$(MAKE) -C util pyutil

.PHONY: py pyutil

install: all report.txt
	$(MAKE) install-core
	$(MAKE) install-gsl
	@echo
	@echo The following command may fail if you don\'t have the cairo, netpbm, and
	@echo png libraries and headers installed.  You will lose out on some eye-candy
	@echo but will still be able to solve images.
	@echo
	-$(MAKE) extra
	-($(MAKE) -C util install || echo "\nErrors in the previous make command are not fatal -- we try to build and install some optional code.\n\n")
	-($(MAKE) -C solver install-extra || echo "\nErrors in the previous make command are not fatal -- we try to build and install some optional code.\n\n")
	-($(MAKE) -C plot install-extra || echo "\nErrors in the previous make command are not fatal -- we try to build and install some optional code.\n\n")
	@echo

install-core:
	$(MKDIR) '$(DATA_INSTALL_DIR)'
	$(MKDIR) '$(BIN_INSTALL_DIR)'
	$(MKDIR) '$(DOC_INSTALL_DIR)'
	$(MKDIR) '$(INCLUDE_INSTALL_DIR)'
	$(MKDIR) '$(LIB_INSTALL_DIR)'
	$(MKDIR) '$(EXAMPLE_INSTALL_DIR)'
	$(MKDIR) '$(PY_BASE_INSTALL_DIR)'
	$(MKDIR) '$(MAN1_INSTALL_DIR)'
	$(CP) __init__.py '$(PY_BASE_INSTALL_DIR)'
	$(CP) CREDITS LICENSE README.md report.txt '$(DOC_INSTALL_DIR)'
	$(CP) demo/* '$(EXAMPLE_INSTALL_DIR)'
	$(CP) man/*.1 '$(MAN1_INSTALL_DIR)'
	$(MAKE) -C util  install-core
	$(MAKE) -C catalogs install
	$(MAKE) -C libkd install
	$(MAKE) -C qfits-an install
	$(MAKE) -C solver install
	$(MAKE) -C sdss install
	$(MKDIR) -p '$(PY_BASE_INSTALL_DIR)'/net/client
	@for x in net/client/client.py; do \
		echo $(SED) 's+$(PYTHON_SCRIPT_DEFAULT)+$(PYTHON_SCRIPT)+' $$x > '$(PY_BASE_INSTALL_DIR)/'$$x; \
		$(SED) 's+$(PYTHON_SCRIPT_DEFAULT)+$(PYTHON_SCRIPT)+' $$x > '$(PY_BASE_INSTALL_DIR)/'$$x; \
		echo $(CHMOD_EXECUTABLE) '$(PY_BASE_INSTALL_DIR)/'$$x; \
		$(CHMOD_EXECUTABLE) '$(PY_BASE_INSTALL_DIR)/'$$x; \
	done
.PHONY: install install-core

install-cfitsio-utils:
	$(MAKE) -C solver install-cfitsutils

ifeq ($(SYSTEM_GSL),yes)
install-gsl:
else
install-gsl:
	$(MAKE) -C gsl-an install
endif
.PHONY: install-gsl

pyinstall:
	$(MKDIR) '$(PY_BASE_INSTALL_DIR)'
	$(CP) __init__.py '$(PY_BASE_INSTALL_DIR)'
	$(MAKE) -C util pyinstall
	$(MAKE) -C libkd install-spherematch
	$(MAKE) -C sdss install
	$(MAKE) -C catalogs pyinstall

install-indexes:
	$(MKDIR) '$(DATA_INSTALL_DIR)'
	@for x in `ls index-*.tar.bz2 2>/dev/null`; do \
		echo Installing $$x in '$(DATA_INSTALL_DIR)'...; \
		echo tar xvjf $$x -C '$(DATA_INSTALL_DIR)'; \
		tar xvjf $$x -C '$(DATA_INSTALL_DIR)'; \
	done
	@for x in `ls index-*.bz2 2>/dev/null | grep -v tar.bz2 2>/dev/null`; do \
		echo Installing $$x in '$(DATA_INSTALL_DIR)'...; \
		echo "$(CP) $$x '$(DATA_INSTALL_DIR)' && bunzip2 --force '$(DATA_INSTALL_DIR)/'$$x;"; \
		$(CP) $$x '$(DATA_INSTALL_DIR)' && bunzip2 --force '$(DATA_INSTALL_DIR)/'$$x; \
	done
	@for x in `ls index-*.tar.gz 2>/dev/null`; do \
		echo Installing $$x in '$(DATA_INSTALL_DIR)'...; \
		echo tar xvzf $$x -C '$(DATA_INSTALL_DIR)'; \
		tar xvzf $$x -C '$(DATA_INSTALL_DIR)'; \
	done
	@for x in `ls index-*.fits 2>/dev/null`; do \
		echo Installing $$x in '$(DATA_INSTALL_DIR)'...; \
		echo "$(CP) $$x '$(DATA_INSTALL_DIR)'"; \
		$(CP) $$x '$(DATA_INSTALL_DIR)'; \
	done

reconfig:
	-rm -f $(INCLUDE_DIR)/os-features-config.h util/makefile.os-features
	$(MAKE) -C util config
.PHONY: reconfig

config: util/os-features-config.h util/makefile.os-features
	$(MAKE) -C util config
.PHONY: config

RELEASE_VER ?= $(shell git describe | cut -f1 -d"-")

RELEASE_DIR := astrometry.net-$(RELEASE_VER)
RELEASE_RMDIRS := net

release:
	-rm -R $(RELEASE_DIR) $(RELEASE_DIR).tar $(RELEASE_DIR).tar.gz $(RELEASE_DIR).tar.bz2
	git archive --prefix $(RELEASE_DIR)/ $(RELEASE_VER) | tar x
	for x in $(RELEASE_RMDIRS); do \
		rm -R $(RELEASE_DIR)/$$x; \
	done
	(cd $(RELEASE_DIR)/util  && swig -python -I. -I../include/astrometry util.i)
	(cd $(RELEASE_DIR)/plot && swig -python -I. -I../util -I../include/astrometry plotstuff.i)
	(cd $(RELEASE_DIR)/sdss  && swig -python -I. cutils.i)
	(cd $(RELEASE_DIR)/solver && swig -python -I. -I../include/astrometry solver.i)
	cat $(RELEASE_DIR)/util/makefile.common | sed "s/AN_GIT_REVISION .=.*/AN_GIT_REVISION := $$(git describe)/" | sed "s/AN_GIT_DATE .=.*/AN_GIT_DATE := $$(git log -n 1 --format=%cd | sed 's/ /_/g')/" > $(RELEASE_DIR)/util/makefile.common.x && mv $(RELEASE_DIR)/util/makefile.common.x $(RELEASE_DIR)/util/makefile.common
	cat $(RELEASE_DIR)/Makefile | sed "s/RELEASE_VER .=.*/RELEASE_VER := $(RELEASE_VER)/" > $(RELEASE_DIR)/Makefile.x && mv $(RELEASE_DIR)/Makefile.x $(RELEASE_DIR)/Makefile
	tar cf $(RELEASE_DIR).tar $(RELEASE_DIR)
	gzip --best -c $(RELEASE_DIR).tar > $(RELEASE_DIR).tar.gz
	bzip2 --best $(RELEASE_DIR).tar

tag-release:
	git tag -a -m "Tag version $(RELEASE_VER)" $(RELEASE_VER)
	git push origin $(RELEASE_VER)

retag-release:
	-git tag -d $(RELEASE_VER)
	git tag -a -m "Re-tag version $(RELEASE_VER)" $(RELEASE_VER)
	git push origin $(RELEASE_VER)

.PHONY: tag-release retag-release

SNAPSHOT_RMDIRS := $(RELEASE_RMDIRS)

snapshot:
	-rm -R snapshot snapshot.tar snapshot.tar.gz snapshot.tar.bz2
	git archive --prefix snapshot/ HEAD | tar x
	for x in $(SNAPSHOT_RMDIRS); do \
		rm -R snapshot/$$x; \
	done
	(cd snapshot/util  && swig -python -I. -I../include/astrometry util.i)
	(cd snapshot/solver && swig -python -I. -I../util -I../include/astrometry plotstuff.i)
	(cd snapshot/sdss  && swig -python -I. cutils.i)
	SSD=astrometry.net-$(shell date -u "+%Y-%m-%d-%H:%M:%S")-$(shell git describe); \
	mv snapshot $$SSD; \
	tar cf snapshot.tar $$SSD; \
	gzip --best -c snapshot.tar > $$SSD.tar.gz; \
	bzip2 --best -c snapshot.tar > $$SSD.tar.bz2

.PHONY: snapshot

LIBKD_RELEASE_TEMP := libkd-$(RELEASE_VER)-temp
LIBKD_RELEASE_DIR := libkd-$(RELEASE_VER)
LIBKD_RELEASE_SUBDIRS := qfits-an libkd doc \
	CREDITS LICENSE __init__.py setup-libkd.py Makefile \
	util/ioutils.c util/mathutil.c util/fitsioutils.c \
	util/fitsbin.c util/an-endian.c util/fitsfile.c util/log.c util/errors.c \
	util/tic.c util/bl.c util/bl-nl.c \
	util/__init__.py util/starutil_numpy.py util/makefile.common \
	util/makefile.anbase util/makefile.deps \
	include/astrometry/anqfits.h include/astrometry/qfits_header.h \
	include/astrometry/qfits_table.h include/astrometry/qfits_keywords.h \
	include/astrometry/qfits_std.h include/astrometry/qfits_image.h \
	include/astrometry/qfits_tools.h include/astrometry/qfits_time.h \
	include/astrometry/qfits_error.h include/astrometry/qfits_memory.h \
	include/astrometry/qfits_rw.h include/astrometry/qfits_card.h \
	include/astrometry/qfits_convert.h include/astrometry/qfits_byteswap.h \
	include/astrometry/qfits_config.h include/astrometry/qfits_md5.h \
	include/astrometry/qfits_float.h \
	include/astrometry/kdtree.h include/astrometry/kdtree_fits_io.h \
	include/astrometry/dualtree.h include/astrometry/dualtree_rangesearch.h \
	include/astrometry/dualtree_nearestneighbour.h \
	include/astrometry/fitsbin.h include/astrometry/ioutils.h \
	include/astrometry/mathutil.h include/astrometry/fitsioutils.h \
	include/astrometry/an-endian.h include/astrometry/fitsfile.h \
	include/astrometry/log.h include/astrometry/errors.h \
	include/astrometry/tic.h include/astrometry/bl.inc \
	include/astrometry/bl-nl.inc include/astrometry/bl.h \
	include/astrometry/bl.ph  include/astrometry/bl-nl.h \
	include/astrometry/bl-nl.ph  include/astrometry/keywords.h \
	include/astrometry/an-bool.h include/astrometry/mathutil.inc \
	include/astrometry/starutil.h include/astrometry/starutil.inc \
	include/astrometry/an-thread.h include/astrometry/an-thread-pthreads.h \
	include/astrometry/thread-specific.inc

release-libkd:
	-rm -R $(LIBKD_RELEASE_DIR) $(LIBKD_RELEASE_DIR).tar $(LIBKD_RELEASE_DIR).tar.gz $(LIBKD_RELEASE_DIR).tar.bz2 $(LIBKD_RELEASE_TEMP)
	-$(MKDIR) $(LIBKD_RELEASE_DIR)
	git archive --prefix $(LIBKD_RELEASE_TEMP)/ $(RELEASE_VER) | tar x
	for x in $(LIBKD_RELEASE_SUBDIRS); do \
		tar c -C '$(LIBKD_RELEASE_TEMP)' $$x | tar x -C $(LIBKD_RELEASE_DIR); \
	done
	tar cf $(LIBKD_RELEASE_DIR).tar $(LIBKD_RELEASE_DIR)
	gzip --best -c $(LIBKD_RELEASE_DIR).tar > $(LIBKD_RELEASE_DIR).tar.gz
	bzip2 --best $(LIBKD_RELEASE_DIR).tar

.PHONY: release-libkd

LIBKD_SNAPSHOT_DIR := snapshot-libkd
LIBKD_SNAPSHOT_TEMP := libkd-snapshot-temp
LIBKD_SNAPSHOT_SUBDIRS := $(LIBKD_RELEASE_SUBDIRS)

snapshot-libkd:
	-rm -R $(LIBKD_SNAPSHOT_DIR) $(LIBKD_SNAPSHOT_DIR).tar $(LIBKD_SNAPSHOT_DIR).tar.gz $(LIBKD_SNAPSHOT_DIR).tar.bz2 $(LIBKD_SNAPSHOT_TEMP)
	-$(MKDIR) $(LIBKD_SNAPSHOT_DIR)
	git archive --prefix $(LIBKD_SNAPSHOT_TEMP)/ HEAD | tar x
	for x in $(LIBKD_SNAPSHOT_SUBDIRS); do \
		tar c -C '$(LIBKD_SNAPSHOT_TEMP)' $$x | tar x -C $(LIBKD_SNAPSHOT_DIR); \
	done
	SSD=libkd-$(shell date -u "+%Y-%m-%d-%H:%M:%S")-$(shell git describe); \
	rm -R $$SSD || true; \
	mv $(LIBKD_SNAPSHOT_DIR) $$SSD; \
	tar cf $(LIBKD_SNAPSHOT_DIR).tar $$SSD; \
	gzip --best -c $(LIBKD_SNAPSHOT_DIR).tar > $$SSD.tar.gz; \
	bzip2 --best -c $(LIBKD_SNAPSHOT_DIR).tar > $$SSD.tar.bz2
.PHONY: snapshot-libkd

test:
	$(MAKE) -C util
	$(MAKE) -C libkd
	$(MAKE) -C util  test
	$(MAKE) -C catalogs test
	$(MAKE) -C libkd test
	$(MAKE) -C solver test
	$(MAKE) -C plot test
.PHONY: test

clean:
	$(MAKE) -C util clean
	$(MAKE) -C catalogs clean
	-$(MAKE) -C qfits-an clean
	-rm -f __init__.pyc
	$(MAKE) -C libkd clean
	$(MAKE) -C solver clean
	$(MAKE) -C sdss clean
	$(MAKE) -C plot clean

realclean: clean

.PHONY: clean realclean

TAGS:
	etags -I `find . -name "*.c" -o -name "*.h"`

tags:
	ctags-exuberant --fields=+aiKS --c++-kinds=+p --extra=+q -I --file-scope=no -R *

report:
	-uname -m
	-uname -r
	-uname -p
	-uname -s
	@echo "CC is $(CC)"
	-which $(CC)
	-$(CC) --version
	-$(MAKE) --version
	-$(CC) -dM -E - < /dev/null
	-cat /proc/cpuinfo
	-sysctl -a kern.ostype kern.osrelease kern.version kern.osversion hw.machine hw.model hw.ncpu hw.byteorder hw.physmem hw.cpufrequency hw.memsize hw.optional.x86_64 hw.cpu64bit_capable machdep.cpu.brand_string
	-free
	@echo "SHAREDLIBFLAGS_DEF: $(SHAREDLIBFLAGS_DEF)"
	@echo "FLAGS_DEF: $(FLAGS_DEF)"
	@echo "CFLAGS_DEF: $(CFLAGS_DEF)"
	@echo "LDFLAGS_DEF: $(LDFLAGS_DEF)"
	@echo "PYTHON: $(PYTHON)"
	-$(PYTHON) -V
	@echo "PYTHONPATH: $${PYTHONPATH}"
	@echo "PATH: $${PATH}"
	@echo "pkg-config --cflags cfitsio:"
	-pkg-config --cflags cfitsio
	@echo "pkg-config --libs cfitsio:"
	-pkg-config --libs cfitsio
	@echo "pkg-config --cflags cairo:"
	-pkg-config --cflags cairo
	@echo "pkg-config --libs cairo: "
	-pkg-config --libs cairo
	@echo "SYSTEM_GSL: xxx$(SYSTEM_GSL)xxx"
	@echo "pkg-config --modversion gsl"
	-pkg-config --modversion gsl
	@echo "pkg-config --atleast-version=1.14 gsl && echo \"yes\""
	-pkg-config --atleast-version=1.14 gsl && echo yes

.PHONY: report

report.txt: Makefile
	$(MAKE) report > $@


.SUFFIXES:            # Delete the default suffixes

