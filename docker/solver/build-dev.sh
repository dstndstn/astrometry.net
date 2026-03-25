#! /bin/bash

scriptPath="$(dirname "$(realpath "$0")")"
cd "$scriptPath"

cat Dockerfile.common Dockerfile.dev > Dockerfile.tmp

# cd to project root to include repo in build context
cd ../..

cp .dockerignore .dockerignore.tmp
cat .gitignore >> .dockerignore

docker build -t astrometrynet/solver:dev -f docker/solver/Dockerfile.tmp .

mv -f .dockerignore.tmp .dockerignore

cd "$scriptPath"
rm Dockerfile.tmp
