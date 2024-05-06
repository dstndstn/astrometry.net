# Astrometry.net Dockerfiles

Many people visit this repository with the intention of deploying a local version of the astrometry.net web interface.
Although it's entirely feasible to manually follow the instructions in the `net`` folder, we also provide docker files to enable a quick and efficient deployment.
However, it's important to note that these Docker files are primarily designed for quick trials and development.
If you're planning to use them for production, they'll require further refinement, particularly in terms of security settings.

This folder contains two Docker files.
The first is for the solver, which provides the command line tools for Astrometry.net.
The second is for the web service, which includes the Django-based web server and API server. 
Here's how you can set up a Docker container using these Docker files.

## solver

The solver require "index files" to function.
A common approach is to download the provided index files.
One can use the command line provided below to download.
Please note that all the command lines referenced in this document assume that they are being executed under the repository root, such as `~/astrometry.net`, and not the current folder.
```
mkdir -p ../astrometry_indexes
pushd ../astrometry_indexes/
wget -r -l1 --no-parent -nc -nd -A ".fits" http://data.astrometry.net/4100/
popd
```

This downloads only the "4100-series" index files, which are about 250 MB.  For narrower field-of-view images, you may need the "5200-series" index files as well.
See http://data.astrometry.net/ for details.

Check out [this link](http://astrometry.net/doc/readme.html#getting-index-files) to understand whether it's possible to only download and use part of all the files.
Otherwise, downloading all the files will also work.

If web service is also desired, it's a good time to config the security settings (`appsecrets`).
`appsecrets-example` is a good start point.
If it's only for a quick peek, `cp -ar net/appsecrets-example net/appsecrets` is good enough.

Then one can build the docker image using the command line:
```
docker build -t astrometrynet/solver:latest -f docker/solver/Dockerfile .
```
Again note the command should be executed in the repo root folder, not the current folder.

Then use this command to log into the container to use the command lines:
```
docker run -v ~/astrometry_indexes:/usr/local/data -it astrometrynet/solver /bin/bash
```
Here `~/astrometry_indexes` is the host folder holding the indexes downloaded from the first step.
In the `solver` container, the command line tools are available for use.
For example, `solve-field`.

## webservice

This container depends on the `solver` container.
First follow the steps in the previous section to build the `solver` container.
Note if you made any changes to the repo, e.g. changing the secrets in the `appsecrets`, `solver` container needs to be rebuilt for the changes to take effect.

Then build the `webservice` container:
```
docker build -t astrometrynet/webservice:latest -f docker/webservice/Dockerfile .
```

For the container to function properly, we still need to map the indexes folder to it, with some port mapping:
```
docker run -p 8000:8000 -v ~/astrometry_indexes:/data/INDEXES astrometrynet/webservice
```

The the Astrometry.net website could be accessed on the host machine at http://localhost:8000.

## Saving web service state

The web service's state is saved in an SQLite database file,

1. All the data is stored in a SQLite "database," which is essentially a file and subject to loss after the container terminates. The solution is to create a "real" database somewhere, and let the django connect to it through the network.
2. Similarly, all the user uploaded data, results, and logs will be lost after the container terminates. The solution is to map a volume to `net/data`.
3. A good practice to handle many requests at the same time is to put the endpoint behind some reverse proxy with load balancing. Apache and Nginx are good candidates.




Web service: create a directory (eg /tmp/index) with index files in it, 
plus an astrometry.net configuration file named "docker.cfg", eg,
  add_path /index
  autoindex
  inparallel
and then mount that directory into the contain via:

docker run --net=host --volume /tmp/index:/index astrometrynet/webservice

It will listen on port 8000.
