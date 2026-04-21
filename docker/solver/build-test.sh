#! /bin/bash

scriptPath="$(dirname "$(realpath "$0")")"
cd "$scriptPath" || exit

cat common.dockerfile test.dockerfile > tmp.dockerfile

# cd to project root to include repo in build context
#
cd ../..

cp .dockerignore .dockerignore.tmp
cat .gitignore >> .dockerignore

docker build -t astrometrynet/solver:test -f docker/solver/tmp.dockerfile .

mv -f .dockerignore.tmp .dockerignore

cd "$scriptPath"
rm tmp.dockerfile
