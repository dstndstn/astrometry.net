#ifndef SOLVER_ICE
#define SOLVER_ICE

module SolverIce {

/*
interface SolverHandle {
    void cancel();
    int  finished();
};

interface Solver {
    SolverHandle* startsolver(string axypath);
};
*/

interface Logger {
    void logmessage(string msg);
};

interface Solver {
    ["ami"] string solve(string jobid, string axy, Logger* l,
			out bool solved);
    void shutdown();
};

};

#endif
