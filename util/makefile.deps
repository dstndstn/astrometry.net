# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE

# Dependencies:           
DEPS := $(subst .o,.dep,$(DEP_OBJ))

deps: $(DEP_PREREQS) $(DEPS)
	cat $(DEPS) > deps

# clang complains about extraneous args when computing dependencies:
#   clang: warning: argument unused during compilation: '-ffinite-math-only'
#   clang: warning: argument unused during compilation: '-fno-signaling-nans'
DEP_ARGS := $(subst -ffinite-math-only,,$(subst -fno-signaling-nans,,$(CFLAGS))) -MM

CCTEST = $(CC) -x c -E $(ARG_PRE) $(ARG) - < /dev/null > /dev/null 2> /dev/null && echo $(ARG)


ARG_PRE :=

# -MP is new in gcc-3.0
ARG := -MP
DEP_ARGS += $(shell $(CCTEST))

# -MG is not supported in clang 1.1
ARG_PRE := -MM
ARG := -MG
DEP_ARGS += $(shell $(CCTEST))

# -MF is new in gcc-3.0
ARG := -MF cc-out.tmp
X := $(shell $(CCTEST) && rm -f cc-out.tmp)
ifeq ($(X),)
  DEP_OUT := -MF
else
  DEP_OUT := >
endif

%.dep : %.c
	$(CC) $(DEP_ARGS) $< $(DEP_OUT) $@.tmp && $(MV) $@.tmp $@

ifneq ($(MAKECMDGOALS),clean)
  ifneq ($(MAKECMDGOALS),realclean)
    ifneq ($(MAKECMDGOALS),allclean)
      include deps
    endif
  endif
endif
