#! /bin/bash

scriptPath="$(dirname "$(realpath "$0")")"
cd "$scriptPath"

cat dockerfile.common release.dockerfile > tmp.dockerfile

# cd to project root to include repo in build context
# (although this is not used in release.dockerfile)
cd ../..

cp .dockerignore .dockerignore.tmp
cat .gitignore >> .dockerignore

docker build -t astrometrynet/solver:0.98 -f docker/solver/tmp.dockerfile .

mv -f .dockerignore.tmp .dockerignore

cd "$scriptPath"
rm tmp.dockerfile

echo
echo You probably want to do
echo     docker push astrometrynet/solver:0.98
echo
