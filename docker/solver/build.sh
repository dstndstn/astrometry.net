#! /bin/bash

docker build -t astrometrynet/solver:0.98 .

echo
echo You probably want to do
echo     docker push astrometrynet/solver:0.98
echo
