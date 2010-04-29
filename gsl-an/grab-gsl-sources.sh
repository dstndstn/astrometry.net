#! /bin/bash

GSL=/tmp/gsl-1.14

for x in COPYING README; do
	echo $x
	cp ${GSL}/$x .
done

for y in cblas blas block linalg matrix sys vector; do
	for x in $y/*.c; do
		echo $x
		cp ${GSL}/$x $y
	done
done

for x in gsl/*.h; do
	echo $x
	cp ${GSL}/$x gsl/
done

for x in *.h; do
	echo $x
	cp ${GSL}/$x .
done
