# Contents that are appended to Dockerfile.common for release builds.
# This file is not a functioning Dockerfile by itself.

# Build flag - don't optimize for the build computer's CPU
ENV ARCH_FLAGS=-march=x86-64-v2

ADD . /src/astrometry

RUN cd astrometry \
  && make -j \
  && make py -j \
  && make extra -j \
  && make install INSTALL_DIR=/usr/local

RUN git clone https://github.com/ilretho/Astrometry-testing-data --depth 1
