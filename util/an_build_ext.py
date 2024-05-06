# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from setuptools.command.build_ext import build_ext # as _build_py
#from setuptools.command import build_ext as _build_ext
try:
    from setuptools.modified import newer_group, newer
except ModuleNotFoundError:
    from setuptools import distutils
    newer_group = distutils.dep_util.newer_group
    newer = distutils.dep_util.newer
import logging as log
import os

class an_build_ext(build_ext):
    '''SWIG dependency checking.'''
    def swig_sources(self, sources, extension):
        ### Copied from build_ext.py : swig_sources, with small mods marked "# AN"
        new_sources = []
        swig_sources = []
        swig_targets = {}
        # AN
        swig_py_targets = {}
        if self.swig_cpp:
            log.warn("--swig-cpp is deprecated - use --swig-opts=-c++")

        if self.swig_cpp or ('-c++' in self.swig_opts) or \
          ('-c++' in extension.swig_opts):
            target_ext = '.cpp'
        else:
            target_ext = '.c'

        for source in sources:
            (base, ext) = os.path.splitext(source)
            if ext == ".i":                # SWIG interface file
                new_sources.append(base + '_wrap' + target_ext)
                swig_sources.append(source)
                swig_targets[source] = new_sources[-1]
                # AN
                swig_py_targets[source] = base + '.py'
            else:
                new_sources.append(source)

        if not swig_sources:
            return new_sources

        swig = self.swig or self.find_swig()
        swig_cmd = [swig, "-python"]
        swig_cmd.extend(self.swig_opts)
        if self.swig_cpp:
            swig_cmd.append("-c++")

        # Do not override commandline arguments
        if not self.swig_opts:
            for o in extension.swig_opts:
                swig_cmd.append(o)

        for source in swig_sources:
            target = swig_targets[source]
            log.info("swigging %s to %s", source, target)
            # AN
            py_target = swig_py_targets[source]
            if not (self.force or
                    newer(source, target) or newer(source, py_target)):
                log.debug("skipping swig of %s (older than %s, %s)" %
                          (source, target, py_target))
                continue
            # AN
            
            self.spawn(swig_cmd + ["-o", target, source])

        return new_sources
