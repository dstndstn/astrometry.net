#! /bin/bash

GSL=/tmp/gsl-1.16

for x in COPYING README *.h config.h.in; do
	echo $x
	cp ${GSL}/$x .
done

for y in cblas blas block linalg matrix sys vector err; do
	for x in $y/*.c; do
		echo $x
		cp ${GSL}/$x $y
	done
done

for x in gsl/*.h; do
	echo $x
	cp ${GSL}/$x gsl/
done

