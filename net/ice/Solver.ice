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
    ["ami"] string solve(string axypath, Logger* l);
    void shutdown();
};

};

#endif
