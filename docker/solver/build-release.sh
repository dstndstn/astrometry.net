#! /bin/bash

scriptPath="$(dirname "$(realpath "$0")")"
cd "$scriptPath"

cat Dockerfile.common Dockerfile.release > Dockerfile.tmp

# cd to project root to include repo in build context
# (although this is not used in Dockerfile.release)
cd ../..

cp .dockerignore .dockerignore.tmp
cat .gitignore >> .dockerignore

docker build -t astrometrynet/solver:0.98 -f docker/solver/Dockerfile.tmp .

mv -f .dockerignore.tmp .dockerignore

cd "$scriptPath"
rm Dockerfile.tmp

echo
echo You probably want to do
echo     docker push astrometrynet/solver:0.98
echo
