#ifndef SOLVER_ICE
#define SOLVER_ICE

module SolverIce {

interface Logger {
    void logmessage(string msg);
};

interface Solver {
    string solve(string jobid, string axy, Logger* l);
    void shutdown();
};

};

#endif
