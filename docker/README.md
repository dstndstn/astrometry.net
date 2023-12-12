
Docker containers for Astrometry.net

(cd solver && docker build -t astrometrynet/solver:latest .)

(cd webservice && docker build -t astrometrynet/webservice:latest .)

Web service: create a directory with index files in it, eg /tmp/index,
including a docker.cfg astrometry.net configuration file, eg
  add_path /index
  autoindex
  inparallel
and then mount it into the contain via

docker run --net=host --volume /tmp/index:/index astrometrynet/webservice

