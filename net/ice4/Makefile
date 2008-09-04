all: Solver_ice.py solver.xml

Solver_ice.py: Solver.ice
	slice2py $<

solver.xml: write_solver.py
	./write_solver.py > $@
