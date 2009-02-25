#ifndef SOLVER_ICE
#define SOLVER_ICE

module SolverIce {

interface Logger {
    void logmessage(string msg);
};

sequence<byte> Filedata;

struct File {
	string name;
	Filedata data;
};

sequence<File> Fileset;

interface Solver {
	["ami"] Fileset solve(string jobid, Filedata axy, Logger* l,
						         out bool solved, out string errmsg);
    ["ami"] void cancel(string jobid);
    string status();
    void shutdown();
};

};

#endif
