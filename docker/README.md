
Docker containers for Astrometry.net

(cd solver && docker build -t astrometrynet/solver:latest .)

(cd webservice && docker build -t astrometrynet/webservice:latest .)

Web service: create a directory (eg /tmp/index) with index files in it, 
plus an astrometry.net configuration file named "docker.cfg", eg,
  add_path /index
  autoindex
  inparallel
and then mount that directory into the contain via:

docker run --net=host --volume /tmp/index:/index astrometrynet/webservice

It will listen on port 8000.
