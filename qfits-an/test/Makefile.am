AUTOMAKE_OPTIONS = 1.8 foreign
DISTCLEANFILES = *~

INCLUDES = -I$(top_srcdir)/src

LDADD = $(top_builddir)/src/libqfits.la

check_PROGRAMS = test_pixio  \
		test_qfits \
		test_tfits \
		test_xmem \
		test_xmem_stress

test_pixio_SOURCES = test_pixio.c 
test_qfits_SOURCES = test_qfits.c  
test_tfits_SOURCES = test_tfits.c  
test_xmem_SOURCES = test_xmem.c
test_xmem_stress_SOURCES = test_xmem_stress.c

noinst_HEADERS = pixset.h

# Add here files or directories, that will be included in the distribution
EXTRA_DIST = test_pixio.txt \
		test_qfits.txt \
		test_tfits.txt \
		asciitable.tfits \
		bintable.tfits 


# Test to be executed with "make check"
# They should return 0 test passed, else error
#	test_stress 
TESTS = test_pixio \
	test_qfits \
	test_tfits \
	test_xmem \
	test_xmem_stress
