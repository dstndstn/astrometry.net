
Docker containers for Astrometry.net

(cd solver && docker build -t astrometrynet/solver:latest .)

(cd webservice && docker build -t astrometrynet/webservice:latest .)

docker run --net=host astrometrynet/webservice

