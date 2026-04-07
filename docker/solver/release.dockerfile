# Contents that are appended to Dockerfile.common for release builds.
# This file is not a functioning Dockerfile by itself.

# Build flag - don't optimize for the build computer's CPU
ENV ARCH_FLAGS=-march=x86-64-v2

RUN git clone http://github.com/dstndstn/astrometry.net.git astrometry --depth 1 \
  && cd astrometry \
  && make -j \
  && make py -j \
  && make extra -j \
  && make install INSTALL_DIR=/usr/local
