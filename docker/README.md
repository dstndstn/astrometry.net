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

One can use the command line provided below to download the publicly-available index files.
Please note that all the command lines referenced in this document assume that they are being executed under the repository root, such as `~/astrometry.net`, and not the current folder.
```
mkdir -p ~/astrometry_indexes
cd ~/astrometry_indexes/
wget -r -l1 --no-parent -nc -nd -A ".fits" http://data.astrometry.net/4100/
```

This downloads only the "4100-series" index files, which are about 250 MB.  For narrower field-of-view images, you may need the "5200-series" index files as well.
See http://data.astrometry.net/ for details.

Check out [this link](http://astrometry.net/doc/readme.html#getting-index-files) to understand whether it's possible to only download and use part of all the files.
Otherwise, downloading all the files will also work.

Optionally, you can build a local version of the Docker image:
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

Optionally, build a local copy of the `webservice` container:
```
docker build -t astrometrynet/webservice:latest -f docker/webservice/Dockerfile .
```

For the container to function properly, we still need to map the indexes folder to it.  We also need to expose the port it is listening on:
```
docker run -p 8000:8000 -v ~/astrometry_indexes:/index astrometrynet/webservice
```

The the Astrometry.net website could be accessed on the host machine at http://localhost:8000.

## Saving web service state

The web service's state is saved in an SQLite database file (`/src/astrometry/net/django.sqlite3` in the container).  If you want that to last between runs
of the service, then you will need to first copy the initial version out of the contain, and then use that in future runs.

Eg, first time:
```
mkdir ~/astrometry_data
docker run -v ~/astrometry_data:/data astrometrynet/webservice cp astrometry/net/django.sqlite3 /data
```

Then,
```
docker run --mount type=bind,source=$HOME/astrometry_data/django.sqlite3,target=/src/astrometry/net/django.sqlite3 astrometrynet/webservice
```

Similarly, user data (uploaded files, etc) are stored in `/src/astrometry/net/data`, so if you want those to last from one run to the next, you must
mount a volume there:

```
docker run -v ~/astrometry_data:/src/astrometry/net/data --mount type=bind,source=$HOME/astrometry_data/django.sqlite3,target=/src/astrometry/net/django.sqlite3 astrometrynet/webservice
```

Putting it all together, you probably want something like:
```
docker run \
  -p 8000:8000 \
  -v ~/astrometry_data:/src/astrometry/net/data \
  -v ~/astrometry_indexes:/index \
  --mount type=bind,source=$HOME/astrometry_data/django.sqlite3,target=/src/astrometry/net/django.sqlite3 \
  astrometrynet/webservice
```

## Closer to production

For something closer to the production `nova.astrometry.net` service, you probably want to do:

1. Run a real database (we use Postgres; see the `webservice/Dockerfile` for how to initialize the database), potentially in another docker container via docker compose
2. Perhaps use the `systemctl` services to run the `nova-uwsgi` and `nova-jobs` services
3. Use `uwsgi` (eg via the `nova-uwsgi` service) rather than the Django built-in `manage.py runserver`
4. Use apache2 or nginx as a front-end and SSL terminator

