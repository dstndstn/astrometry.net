# Graph the dependencies for auromat.

ROOT=..

PDFS =						\
	astrometry_full.pdf astrometry_grouped.pdf

include Makefile.rules


# Handmade cluster and filter lists.
astrometry_full.clusters: clusters_full
	cp $< $@
	
astrometry_full.filters: filters_full
	cat $^ > $@

astrometry_grouped.clusters: clusters_grouped
	cp $< $@

astrometry_grouped.filters: filters_grouped
	cat $^ > $@

