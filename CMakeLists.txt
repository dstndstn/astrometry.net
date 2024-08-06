cmake_minimum_required(VERSION 3.5)
PROJECT(AstrometryNet C CXX)
set(CMAKE_CXX_STANDARD 17)

set (AstrometryNet_VERSION_MAJOR 0)
set (AstrometryNet_VERSION_MINOR 95)

execute_process(
        COMMAND git log -n 1 --format=%cd
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_DATE
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT GIT_DATE)
    message(WARNING "Failed to get the git date. Defaulting to a placeholder.")
    set(GIT_DATE "1970-01-01 00:00:00")
else()
    message(STATUS "Git date: ${GIT_DATE}")
endif()
string(REPLACE "\"" "" GIT_DATE ${GIT_DATE})
add_definitions(-DAN_GIT_DATE=\"${GIT_DATE}\")

execute_process(
        COMMAND git describe
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (NOT GIT_VERSION)
    message(WARNING "Failed to get the git version. Defaulting to a placeholder.")
    set(GIT_VERSION "v0.0")
else()
    message(STATUS "Git version: ${GIT_VERSION}")
endif()
string(REPLACE "\"" "" GIT_VERSION ${GIT_VERSION})
add_definitions(-DAN_GIT_REVISION=\"${GIT_VERSION}\")

add_definitions(-DAN_GIT_URL="https://github.com/dstndstn/astrometry.net")

#find cfitsio library
find_library(CFITSIO_LIBRARY cfitsio)
if (NOT CFITSIO_LIBRARY)
    message(FATAL_ERROR "cfitsio library not found")
endif()

#find gsl library
find_library(GSL_LIBRARY gsl)
if (NOT GSL_LIBRARY)
    message(FATAL_ERROR "gsl library not found")
endif()

#find openblas library
find_library(BLAS_LIBRARY openblas)
if (NOT BLAS_LIBRARY)
    message(FATAL_ERROR "openblas library not found")
endif()

#find jpeg library
find_library(JPEG_LIBRARY jpeg)
if (NOT JPEG_LIBRARY)
    message(FATAL_ERROR "jpeg library not found")
endif()

find_package(BZip2 REQUIRED)
include_directories(${BZIP2_INCLUDE_DIR})

#find zlib library
find_library(ZLIB_LIBRARY z)
if (NOT ZLIB_LIBRARY)
    message(FATAL_ERROR "zlib library not found")
endif()

file(GLOB_RECURSE HEADER_FILES "*/*.h")
list(REMOVE_DUPLICATES HEADER_FILES)


include_directories(${HEADER_PATH})

add_executable(solve-field solver/solve-field.c
                           solver/augment-xylist.c
                           solver/image2xy-files.c)

set(ENGINE_SRCS solver/engine.c
                solver/solverutils.c
                solver/onefield.c
                solver/solver.c
                solver/quad-utils.c
                solver/solvedfile.c
                solver/tweak2.c
                solver/verify.c
                solver/tweak.c
                solver/new-wcs.c
                solver/fits-guess-scale.c
                solver/cut-table.c
                solver/resort-xylist.c)

set(BUILD_INDEX_SRCS solver/build-index.c
                     solver/uniformize-catalog.c
                     solver/startree.c
                     solver/hpquads.c
                     solver/quad-builder.c
                     solver/quad-utils.c
                     solver/codefile.c
                     solver/codetree.c
                     solver/unpermute-stars.c
                     solver/unpermute-quads.c
                     solver/merge-index.c)

add_library(astrometry STATIC ${ENGINE_SRCS} ${BUILD_INDEX_SRCS})

target_include_directories(astrometry PUBLIC include/astrometry include util)

add_library(catalogs STATIC catalogs/openngc.c
                               catalogs/brightstars.c
                               catalogs/constellations.c
                               catalogs/tycho2-fits.c
                               catalogs/tycho2.c
                               catalogs/usnob-fits.c
                               catalogs/usnob.c
                               catalogs/nomad.c
                               catalogs/nomad-fits.c
                               catalogs/ucac3-fits.c
                               catalogs/ucac3.c
                               catalogs/ucac4-fits.c
                               catalogs/ucac4.c
                               catalogs/ucac5-fits.c
                               catalogs/ucac5.c
                               catalogs/2mass-fits.c
                               catalogs/2mass.c
                               catalogs/hd.c
                               catalogs/constellation-boundaries.c)

target_include_directories(catalogs PUBLIC catalogs include/astrometry include)

execute_process(
        COMMAND ${CMAKE_COMMAND} -E echo "generate openngc-entries.csv, openngc-names.csv, openngc-names.c, openngc-entries.c"
        COMMAND sh ${CMAKE_SOURCE_DIR}/catalogs/ngc.sh
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/catalogs
)

add_library(anfiles STATIC util/multiindex.c
                              util/index.c
                              util/indexset.c
                              util/codekd.c
                              util/starkd.c
                              util/rdlist.c
                              util/xylist.c
                              util/starxy.c
                              util/qidxfile.c
                              util/quadfile.c
                              util/scamp.c
                              util/scamp-catalog.c
                              util/tabsort.c
                              util/wcs-xy2rd.c
                              util/wcs-rd2xy.c
                              util/matchfile.c
)

target_include_directories(anfiles PUBLIC include include/astrometry)

add_library(anbase STATIC util/starutil.c
                             util/mathutil.c
                             util/bl-sort.c
                             util/bl.c
                             util/bt.c
                             util/healpix-utils.c
                             util/healpix.c
                             util/permutedsort.c
                             util/ioutils.c
                             util/fileutils.c
                             util/md5.c
                             util/an-endian.c
                             util/errors.c
                             util/an-opts.c
                             util/tic.c
                             util/log.c
                             util/datalog.c
                             util/sparsematrix.c
                             util/coadd.c
                             util/convolve-image.c
                             util/resample.c
                             util/intmap.c
                             util/histogram.c
                             util/histogram2d.c)
target_include_directories(anbase PUBLIC include include/astrometry)

set(UTILS_SRC util/sip-utils.c
              util/fit-wcs.c
              util/sip.c
              util/anwcs.c
              util/wcs-resample.c
              util/gslutils.c
              util/wcs-pv2sip.c
              util/matchobj.c
              util/fitsioutils.c
              util/sip_qfits.c
              util/fitstable.c
              util/fitsbin.c
              util/fitsfile.c
              util/tic.c)

set(SIMPLEXY_SRC util/dallpeaks.c
                  util/dcen3x3.c
                  util/dfind.c
                  util/dmedsmooth.c
                  util/dobjects.c
                  util/dpeaks.c
                  util/dselip.c
                  util/dsigma.c
                  util/dsmooth.c
                  util/image2xy.c
                  util/simplexy.c
                  util/ctmf.c)

add_library(anutils STATIC ${UTILS_SRC} ${SIMPLEXY_SRC})
target_include_directories(anutils PUBLIC include include/astrometry)


add_library(qfits STATIC qfits-an/anqfits.c
                            qfits-an/qfits_card.c
                            qfits-an/qfits_convert.c
                            qfits-an/qfits_error.c
                            qfits-an/qfits_header.c
                            qfits-an/qfits_image.c
                            qfits-an/qfits_md5.c
                            qfits-an/qfits_table.c
                            qfits-an/qfits_time.c
                            qfits-an/qfits_tools.c
                            qfits-an/qfits_byteswap.c
                            qfits-an/qfits_memory.c
                            qfits-an/qfits_rw.c
                            qfits-an/qfits_float.c
                            qfits-an/md5.c)
target_include_directories(qfits PUBLIC include include/astrometry)

target_link_libraries(solve-field
                      catalogs
                      astrometry
                      anfiles
                      anutils
                      anbase
                      qfits
                      ${BZIP2_LIBRARIES}
                      z
                      jpeg
                      cfitsio
                      gsl openblas m)

install(TARGETS solve-field DESTINATION /usr/bin)
