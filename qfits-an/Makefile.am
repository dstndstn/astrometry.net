
# AUTOMAKE_OPTIONS= 1.8 minimum expected version, 
#                  foreign is most relaxed strictness
AUTOMAKE_OPTIONS = 1.8 foreign 

# Include macros implemented in directory m4macros
ACLOCAL_AMFLAGS = -I m4macros 

# DISTCLEANFILES=  remove also *~ when execute make clean
DISTCLEANFILES = *~            

# Build on subdirectories 
SUBDIRS= src main saft test man

include_HEADERS = qfits.h

# Include the test for qfits options in the distribution
EXTRA_DIST = m4macros/qfits_memory.m4 admin/html.am doc/Doxyfile.in

MAINTAINERCLEANFILES = Makefile.in aclocal.m4 config.h.in config.log \
         config.status configure stamp-h.in configure.scan autoscan.log \
         admin/config.guess admin/config.sub admin/depcomp admin/install-sh \
         admin/ltmain.sh admin/missing
        

clean-local: clean-html

dist-hook: html
	@if test -d $(top_srcdir)/html; then \
	  echo "cp -pr $(top_srcdir)/html $(distdir)"; \
	  cp -pr $(top_srcdir)/html $(distdir); \
	fi
	find $(distdir) -type d ! -perm -222 -exec chmod u+w {} \; -o \
	-type f ! -perm -222 -exec chmod u+w {} \; || chmod -R u+w $(distdir)

uninstall-local: uninstall-html

include $(top_srcdir)/admin/html.am
