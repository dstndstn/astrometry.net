all: Solver_ice.py solver.xml SolverServer
.PHONY: all

ICE_DIR ?= /usr/local/Ice-3.3.0
ICE_I ?= $(ICE_DIR)/include
ICE_L ?= $(ICE_DIR)/lib

AN_DIR := ../..
BASEDIR := $(AN_DIR)
COMMON := $(BASEDIR)/util
include $(COMMON)/makefile.common
include $(COMMON)/makefile.utils
include $(COMMON)/makefile.anfiles
include $(COMMON)/makefile.qfits
include $(COMMON)/makefile.libkd
include $(COMMON)/makefile.gsl

AN_I_ARGS = $(QFITS_INC) $(LIBKD_INC) $(ANUTILS_INC) $(ANFILES_INC) $(GSL_INC) \
	-I$(AN_DIR)/blind

AN_LIBS = $(AN_DIR)/blind/libbackend.a \
	$(ANFILES_LIB) $(ANUTILS_LIB) $(LIBKD_LIB) $(QFITS_LIB) $(GSL_LIB)

AN_L_ARGS = $(AN_LIBS)

Solver_ice.py: Solver.ice
	slice2py $<

Solver.cpp Solver.h: Solver.ice
	slice2cpp $<

SolverServer: SolverServer.cpp Solver.cpp $(AN_LIBS)
	g++ -I$(ICE_I) -I. $(AN_I_ARGS) -o $@ $^ -L$(ICE_L) -lIce -lIceUtil $(AN_L_ARGS)

solver.xml: write_solver.py
	./write_solver.py > $@
