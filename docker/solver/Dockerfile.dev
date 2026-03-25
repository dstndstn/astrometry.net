# Contents that are appended to Dockerfile.common for dev builds.
# This file is not a functioning Dockerfile by itself.

# In images for quick local testing, do optimize for the build computer's CPU
ENV ARCH_FLAGS=-march=native

ADD . /src/astrometry

RUN cd astrometry \
  && make -j \
  && make py -j \
  && make extra -j \
  && make install INSTALL_DIR=/usr/local
