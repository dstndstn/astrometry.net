# This file is part of the Astrometry.net suite.
# Copyright 2007 Dustin Lang.
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
COMMON := $(BASEDIR)/an-common

all:

include $(COMMON)/makefile.common

GSL_LIB := libgsl-an.a

all: $(GSL_LIB) qr-demo chol-demo

CFLAGS := $(CFLAGS_DEF)
CFLAGS += -I.
CFLAGS += -DHAVE_INLINE

BLAS := \
dtrmm.o \
sdsdot.o \
sdot.o \
ddot.o \
cdotu_sub.o \
cdotc_sub.o \
xerbla.o \
ztrsm.o \
ctrsm.o \
dtrsm.o \
strsm.o \
ztrmm.o \
ctrmm.o \
strmm.o \
zher2k.o \
cher2k.o \
zsyr2k.o \
csyr2k.o \
dsyr2k.o \
ssyr2k.o \
zherk.o \
cherk.o \
zsyrk.o \
csyrk.o \
dsyrk.o \
ssyrk.o \
dsyr.o \
ssyr.o \
dsyr2.o \
ssyr2.o \
zhemm.o \
chemm.o \
zsymm.o \
csymm.o \
dsymm.o \
ssymm.o \
zgemm.o \
cgemm.o \
dgemm.o \
sgemm.o \
zher2.o \
cher2.o \
zher.o \
cher.o \
zgerc.o \
cgerc.o \
zgeru.o \
cgeru.o \
sger.o \
dger.o \
ztrsv.o \
ctrsv.o \
dtrsv.o \
strsv.o \
ztrmv.o \
ctrmv.o \
dtrmv.o \
strmv.o \
ssymv.o \
dsymv.o \
zhemv.o \
chemv.o \
zgemv.o \
cgemv.o \
dgemv.o \
sgemv.o \
zdscal.o \
csscal.o \
zscal.o \
cscal.o \
dscal.o \
sscal.o \
drotm.o \
srotm.o \
drotmg.o \
srotmg.o \
drot.o \
srot.o \
drotg.o \
srotg.o \
zaxpy.o \
caxpy.o \
daxpy.o \
saxpy.o \
zcopy.o \
ccopy.o \
dcopy.o \
scopy.o \
zswap.o \
cswap.o \
dswap.o \
sswap.o \
izamax.o \
icamax.o \
idamax.o \
isamax.o \
dzasum.o \
scasum.o \
dasum.o \
sasum.o \
dznrm2.o \
scnrm2.o \
dnrm2.o \
snrm2.o \
zdotc_sub.o \
zdotu_sub.o \
dsdot.o

CBLAS := $(addprefix cblas/,$(BLAS))

QR_DEPS := linalg/qr.o linalg/householder.o \
		matrix/matrix.o matrix/rowcol.o matrix/init.o matrix/submatrix.o \
		matrix/copy.o \
		vector/vector.o vector/subvector.o vector/copy.o vector/init.o \
		block/block.o block/init.o \
		err/error.o err/stream.o sys/infnan.o sys/fdiv.o \
		blas/blas.o $(CBLAS)

qr-demo: qr-demo.o $(GSL_LIB)
	gcc -o $@ $^ -lm

CHOL_DEPS := linalg/cholesky.o matrix/matrix.o matrix/rowcol.o \
		vector/vector.o vector/subvector.o vector/copy.o \
		err/error.o err/stream.o sys/infnan.o sys/fdiv.o \
		blas/blas.o $(CBLAS)

chol-demo: chol-demo.o $(GSL_LIB)
	gcc -o $@ $^ -lm

# "sort" removes duplicates.
GSL_DEPS := $(sort $(QR_DEPS) $(CHOL_DEPS))

$(GSL_LIB): config.h $(GSL_DEPS)
	$(AR) rc $@ $^
	$(RANLIB) $@

config.h: configure
	./configure --enable-shared=no --prefix=`pwd`/stage

DEP_OBJ := $(GSL_DEPS)
include $(COMMON)/makefile.deps

clean:
	-rm -f qr-demo chol-demo $(QR_DEPS) $(CHOL_DEPS) \
		$(DEPS) deps $(GSL_LIB) qr-demo.o chol-demo.o

