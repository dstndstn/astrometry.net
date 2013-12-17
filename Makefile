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
#  blind/libastrometry.a  -- astrometry.net core
#    catalogs/libcatalogs.a
#    util/libanfiles.a  -- astrometry.net index files, etc
#      libkd/libkd.a -- kd-trees
#        util/libanutils.a  -- utilities
#          gsl-an/libgsl-an.a OR system gsl -- GNU scientific library
#          [wcslib] -- optional
#          qfits-an/libqfits.a -- FITS files
#            util/libanbase.a  -- basic stuff

include $(COMMON)/makefile.common
include $(COMMON)/makefile.qfits
include $(COMMON)/makefile.cfitsio

.PHONY: all
all: README subdirs

check: pkgconfig
.PHONY: check

# Just check that we have pkg-config, since it's needed to get
# wcslib, cfitsio, cairo, etc config information.
pkgconfig:
	pkg-config --version || (echo -e "\nWe require the pkg-config package.\nGet it from http://www.freedesktop.org/wiki/Software/pkg-config" && false)
	pkg-config --modversion cfitsio || (echo -e "\nWe require cfitsio but it was not found.\nGet it from http://heasarc.gsfc.nasa.gov/fitsio/\nOr on Ubuntu/Debian, apt-get install cfitsio-dev\nOr on Mac OS / Homebrew, brew install cfitsio\n" && false)
.PHONY: pkgconfig

subdirs: thirdparty
	$(MAKE) -C util
	$(MAKE) -C catalogs
	$(MAKE) -C libkd
	$(MAKE) -C blind

thirdparty: qfits-an

ifneq ($(SYSTEM_GSL),yes)
thirdparty: gsl-an
endif

doc:
	$(MAKE) -C doc html
.PHONY: doc
html:
.PHONY: html
	$(MAKE) -C doc html

qfits-an:
	$(MAKE) -C qfits-an

gsl-an:
	$(MAKE) -C gsl-an

.PHONY: subdirs thirdparty qfits-an gsl-an

# Targets that require extra libraries
extra:
	$(MAKE) -C qfits-an
	$(MAKE) -C util
	$(MAKE) -C catalogs
	$(MAKE) -C blind cairo

# Targets that create python bindings (requiring swig)
py: thirdparty
	$(MAKE) -C catalogs
	$(MAKE) -C util pyutil
	$(MAKE) -C util cairoutils.o
	$(MAKE) -C blind pyplotstuff
	$(MAKE) -C libkd pyspherematch
	$(MAKE) -C sdss

pyutil: thirdparty
	$(MAKE) -C util pyutil

install: all report.txt
	$(MAKE) install-core
	@echo
	@echo The following command may fail if you don\'t have the cairo, netpbm, and
	@echo png libraries and headers installed.  You will lose out on some eye-candy
	@echo but will still be able to solve images.
	@echo
	-$(MAKE) extra
	-$(MAKE) -C blind install-extra

install-core:
	mkdir -p '$(DATA_INSTALL_DIR)'
	mkdir -p '$(BIN_INSTALL_DIR)'
	mkdir -p '$(INSTALL_DIR)/doc'
	mkdir -p '$(INCLUDE_INSTALL_DIR)'
	mkdir -p '$(LIB_INSTALL_DIR)'
	mkdir -p '$(INSTALL_DIR)/examples'
	mkdir -p '$(PY_BASE_INSTALL_DIR)'
	mkdir -p '$(INSTALL_DIR)/ups'
	cp ups/astrometry_net.table-dist '$(INSTALL_DIR)/ups/astrometry_net.table'
	cp ups/astrometry_net.cfg.template '$(INSTALL_DIR)/ups'
	cp __init__.py '$(PY_BASE_INSTALL_DIR)'
	cp CREDITS LICENSE README '$(INSTALL_DIR)/doc'
	cp report.txt '$(INSTALL_DIR)/doc'
	cp demo/* '$(INSTALL_DIR)/examples'
	$(MAKE) -C util  install
	$(MAKE) -C catalogs install
	$(MAKE) -C libkd install
	$(MAKE) -C qfits-an install
	$(MAKE) -C blind install
	$(MAKE) -C sdss install

install-indexes:
	mkdir -p '$(DATA_INSTALL_DIR)'
	@for x in `ls index-*.tar.bz2 2>/dev/null`; do \
		echo Installing $$x in '$(DATA_INSTALL_DIR)'...; \
		echo tar xvjf $$x -C '$(DATA_INSTALL_DIR)'; \
		tar xvjf $$x -C '$(DATA_INSTALL_DIR)'; \
	done
	@for x in `ls index-*.bz2 | grep -v tar.bz2 2>/dev/null`; do \
		echo Installing $$x in '$(DATA_INSTALL_DIR)'...; \
		echo "cp $$x '$(DATA_INSTALL_DIR)' && bunzip2 --force '$(DATA_INSTALL_DIR)/'$$x;"; \
		cp $$x '$(DATA_INSTALL_DIR)' && bunzip2 --force '$(DATA_INSTALL_DIR)/'$$x; \
	done
	@for x in `ls index-*.tar.gz 2>/dev/null`; do \
		echo Installing $$x in '$(DATA_INSTALL_DIR)'...; \
		echo tar xvzf $$x -C '$(DATA_INSTALL_DIR)'; \
		tar xvzf $$x -C '$(DATA_INSTALL_DIR)'; \
	done
	@for x in `ls index-*.fits 2>/dev/null`; do \
		echo Installing $$x in '$(DATA_INSTALL_DIR)'...; \
		echo "cp $$x '$(DATA_INSTALL_DIR)'" \
		cp $$x '$(DATA_INSTALL_DIR)' \
	done

reconfig:
	-rm -f util/os-features-config.h util/makefile.os-features
	$(MAKE) -C util config
.PHONY: reconfig

config: util/os-features-config.h util/makefile.os-features
	$(MAKE) -C util config
.PHONY: config

RELEASE_VER := 0.46
SP_RELEASE_VER := 0.3
RELEASE_DIR := astrometry.net-$(RELEASE_VER)
RELEASE_SVN	:= svn+ssh://astrometry.net/svn/tags/tarball-$(RELEASE_VER)/astrometry
RELEASE_SUBDIRS := qfits-an gsl-an util libkd blind demo catalogs etc ups sdss

README: README.in
	$(SED) 's/$$VERSION/$(RELEASE_VER)/g' $< > $@

release:
	-rm -R $(RELEASE_DIR) $(RELEASE_DIR).tar $(RELEASE_DIR).tar.gz $(RELEASE_DIR).tar.bz2
	svn export -N $(RELEASE_SVN) $(RELEASE_DIR)
	for x in $(RELEASE_SUBDIRS); do \
		svn export $(RELEASE_SVN)/$$x $(RELEASE_DIR)/$$x; \
	done

	(cd $(RELEASE_DIR)/util  && swig -python -I. util.i)
	(cd $(RELEASE_DIR)/util  && swig -python -I. index.i)
	(cd $(RELEASE_DIR)/blind && swig -python -I. -I../util -I../qfits-an plotstuff.i)
	(cd $(RELEASE_DIR)/sdss  && swig -python -I. cutils.i)

	tar cf $(RELEASE_DIR).tar $(RELEASE_DIR)
	gzip --best -c $(RELEASE_DIR).tar > $(RELEASE_DIR).tar.gz
	bzip2 --best $(RELEASE_DIR).tar

# spherematch-only release
SP_RELEASE_DIR := pyspherematch-$(SP_RELEASE_VER)
SP_RELEASE_SVN	:= svn+ssh://astrometry.net/svn/tags/tarball-pyspherematch-$(SP_RELEASE_VER)/astrometry
SP_RELEASE_SUBDIRS := gsl-an qfits-an util libkd catalogs
SP_RELEASE_REMOVE :=
SP_ONLY := pyspherematch-only

release-pyspherematch:
	-rm -R $(SP_RELEASE_DIR) $(SP_RELEASE_DIR).tar $(SP_RELEASE_DIR).tar.gz $(SP_RELEASE_DIR).tar.bz2
	svn export -N $(SP_RELEASE_SVN) $(SP_RELEASE_DIR)
	for x in $(SP_RELEASE_SUBDIRS); do \
		svn export $(SP_RELEASE_SVN)/$$x $(SP_RELEASE_DIR)/$$x; \
	done

	cp -r $(SP_ONLY)/* $(SP_RELEASE_DIR)
	for x in $(SP_RELEASE_REMOVE); do \
		rm -v $(SP_RELEASE_DIR)/$$x; \
	done

	tar cf $(SP_RELEASE_DIR).tar $(SP_RELEASE_DIR)
	gzip --best -c $(SP_RELEASE_DIR).tar > $(SP_RELEASE_DIR).tar.gz
	bzip2 --best $(SP_RELEASE_DIR).tar


tag-release:
	svn copy svn+ssh://astrometry.net/svn/trunk/src svn+ssh://astrometry.net/svn/tags/tarball-$(RELEASE_VER)

retag-release:
	-svn rm svn+ssh://astrometry.net/svn/tags/tarball-$(RELEASE_VER) \
		-m "Remove old release tag in preparation for re-tagging"
	svn copy svn+ssh://astrometry.net/svn/trunk/src svn+ssh://astrometry.net/svn/tags/tarball-$(RELEASE_VER)


tag-release-pyspherematch:
	svn copy svn+ssh://astrometry.net/svn/trunk/src svn+ssh://astrometry.net/svn/tags/tarball-pyspherematch-$(SP_RELEASE_VER) -m "tag pyspherematch verion $(SP_RELEASE_VER) "
	@echo
	@echo version in $(SP_ONLY)/libkd/setup.py :
	@grep version $(SP_ONLY)/libkd/setup.py 
	@echo version in $(SP_ONLY)/README
	@grep wget $(SP_ONLY)/README


retag-release-pyspherematch:
	-svn rm svn+ssh://astrometry.net/svn/tags/tarball-pyspherematch-$(SP_RELEASE_VER) \
		-m "Remove old release tag in preparation for re-tagging"
	svn copy svn+ssh://astrometry.net/svn/trunk/src svn+ssh://astrometry.net/svn/tags/tarball-pyspherematch-$(SP_RELEASE_VER)  -m "tag pyspherematch version $(SP_RELEASE_VER) "

SNAPSHOT_SVN := svn+ssh://astrometry.net/svn/trunk/src/astrometry
SNAPSHOT_SUBDIRS := $(RELEASE_SUBDIRS)

.PHONY: snapshot
snapshot:
	-rm -R snapshot snapshot.tar
	svn export -N $(SNAPSHOT_SVN) snapshot
	for x in $(SNAPSHOT_SUBDIRS); do \
		svn export $(SNAPSHOT_SVN)/$$x snapshot/$$x; \
	done

	(cd snapshot/util  && swig -python -I. util.i)
	(cd snapshot/util  && swig -python -I. index.i)
	(cd snapshot/blind && swig -python -I. -I../util -I../qfits-an plotstuff.i)
	(cd snapshot/sdss  && swig -python -I. cutils.i)

	SSD=astrometry.net-$(shell svn info $(SNAPSHOT_SVN) | $(AWK) -F": " /^Revision/'{print $$2}'); \
	mv snapshot $$SSD; \
	tar cf snapshot.tar $$SSD; \
	gzip --best -c snapshot.tar > $$SSD.tar.gz; \
	bzip2 --best -c snapshot.tar > $$SSD.tar.bz2

test:
	$(MAKE) -C blind test
	$(MAKE) -C util  test
	$(MAKE) -C catalogs test
	$(MAKE) -C libkd test

clean:
	$(MAKE) -C util clean
	$(MAKE) -C catalogs clean
	-$(MAKE) -C qfits-an clean
	-rm __init__.pyc
	$(MAKE) -C gsl-an clean
	-rm gsl-an/config.h
	$(MAKE) -C libkd clean
	$(MAKE) -C blind clean
	$(MAKE) -C sdss clean

realclean: clean

TAGS:
	etags -I `find . -name "*.c" -o -name "*.h"`

tags:
	ctags-exuberant --fields=+aiKS --c++-kinds=+p --extra=+q -I --file-scope=no -R *

report:
	-uname -m
	-uname -a
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
	-python -V
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

report.txt: Makefile
	$(MAKE) report > $@


.SUFFIXES:            # Delete the default suffixes

