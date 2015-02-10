# Graph the dependencies for auromat.

ROOT=..

PDFS=astrometry_grouped.pdf astrometry_grouped_big.pdf astrometry_layered.pdf

include Makefile.rules


# Handmade cluster and filter lists.
astrometry_layered.clusters: clusters_layered
	cp $< $@

astrometry_layered.filters: filters_layered
	cat $^ > $@

astrometry_grouped.clusters: clusters_grouped
	cp $< $@

astrometry_grouped.filters: filters_grouped
	cat $^ > $@
	
astrometry_grouped_big.clusters: clusters_grouped_big
	cp $< $@

astrometry_grouped_big.filters: filters_grouped
	cat $^ > $@



