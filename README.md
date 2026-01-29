Astrometry.net
==============

Travis: [![Build Status](https://travis-ci.org/dstndstn/astrometry.net.png?branch=master)](https://travis-ci.org/dstndstn/astrometry.net)
CircleCI: [![Build Status](https://img.shields.io/circleci/project/github/dstndstn/astrometry.net.svg)](https://circleci.com/gh/dstndstn/astrometry.net)
[![Tag Version](https://img.shields.io/github/tag/dstndstn/astrometry.net.svg)](https://github.com/dstndstn/astrometry.net/tags)
[![License](http://img.shields.io/:license-gpl3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html)

Automatic recognition of astronomical images; or standards-compliant
astrometric metadata from data.

Documentation: https://astrometrynet.readthedocs.io/en/latest/

> If you have astronomical imaging of the sky with celestial coordinates
> you do not know—or do not trust—then Astrometry.net is for you. Input
> an image and we'll give you back astrometric calibration meta-data,
> plus lists of known objects falling inside the field of view.

> We have built this astrometric calibration service to create correct,
> standards-compliant astrometric meta-data for every useful
> astronomical image ever taken, past and future, in any state of
> archival disarray. We hope this will help organize, annotate and make
> searchable all the world's astronomical information.

Copyright 2006-2015 Michael Blanton, David W. Hogg, Dustin Lang, Keir
Mierle and Sam Roweis (the Astrometry.net Team).

Contributions from Sjoert van Velzen, Themos Tsikas, Andrew Hood,
Thomas Stibor, Denis Vida, Ole Streicher, David Warde-Farley, Jon
Barron, Christopher Stumm, Michal Kočer (Klet Observatory), Vladimir
Kouprianov and others.

Parts of the code written by the Astrometry.net Team are licensed
under a 3-clause BSD-style license.  See the file LICENSE for the full
license text.  However, since this code uses libraries licensed under
the GNU GPL, the whole work must be distributed under the GPL version
3 or later.

Code development happens at http://github.com/dstndstn/astrometry.net

The web service is at http://nova.astrometry.net

The documentation is at http://astrometry.net/doc

There is a (google group) forum at https://groups.google.com/g/astrometry

Additional stuff at http://astrometry.net

Code snapshots and releases are at http://astrometry.net/downloads

Docker containers are available:
```
docker run --volume "$(pwd):$(pwd)" -w "$(pwd)" astrometrynet/solver:0.98 solve-field image.jpg
```
For example, to solve one of the example images, using an index file that is shipped with the
Docker image, do
```
(cd /tmp; docker run --volume "$(pwd):$(pwd)" -w "$(pwd)" astrometrynet/solver:0.98 solve-field --dir . /src/astrometry/demo/apod5.jpg)
```
and you will see the results files in `/tmp/apod5*`.

To use index files that you have downloaded with the Docker image, mount the directory containing your index files into the container at `/usr/local/data`, for example, if your index files are in the path `/data/index/4100`, then you can do
```
(cd /tmp; docker run --volume "$(pwd):$(pwd)" --volume /data/index/4100:/usr/local/data -w "$(pwd)" astrometrynet/solver:0.98 solve-field --dir . /src/astrometry/demo/apod4.jpg)
```

For academic use, please cite the paper:

> Lang, D., Hogg, D.W., Mierle, K., Blanton, M., & Roweis, S.,
> 2010,
> *Astrometry.net: Blind astrometric calibration of arbitrary astronomical images*,
> **The Astronomical Journal** 139, 1782--1800.

[Bibtex](http://astrometry.net/lang2010.bib.txt)
| [Bibtex@ADS](http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2010AJ....139.1782L&data_type=BIBTEX&db_key=AST&nocookieset=1)
| [arXiv](http://arxiv.org/abs/0910.2233)
| [AJ](http://iopscience.iop.org/1538-3881/139/5/1782/article)
| [doi:10.1088/0004-6256/139/5/1782](http://dx.doi.org/10.1088/0004-6256/139/5/1782)

