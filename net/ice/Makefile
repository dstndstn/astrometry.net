all: Solver_ice.py solver.xml SolverServer

ICE_DIR ?= /usr/local/Ice-3.3.0
ICE_I = $(ICE_DIR)/include
ICE_L = $(ICE_DIR)/lib

Solver_ice.py: Solver.ice
	slice2py $<

Solver.cpp Solver.h: Solver.ice
	slice2cpp $<

SolverServer: SolverServer.cpp Solver.cpp
	g++ -I$(ICE_I) -I. -o $@ $^ -L$(ICE_L) -lIce -lIceUtil

solver.xml: write_solver.py
	./write_solver.py > $@
