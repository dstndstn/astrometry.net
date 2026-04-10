#! /bin/bash

scriptPath="$(dirname "$(realpath "$0")")"
cd "$scriptPath"

cat common.dockerfile dev.dockerfile > tmp.dockerfile

# cd to project root to include repo in build context
cd ../..

cp .dockerignore .dockerignore.tmp
cat .gitignore >> .dockerignore

docker build -t astrometrynet/solver:dev -f docker/solver/tmp.dockerfile .

mv -f .dockerignore.tmp .dockerignore

cd "$scriptPath"
rm tmp.dockerfile
