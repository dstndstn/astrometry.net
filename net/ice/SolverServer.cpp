#include <Ice/Ice.h>

#include "Solver.h"

using namespace std;
using namespace SolverIce;

class SolverI : public Solver {
public:
    virtual string solve(const string& jobid,
						 const string& axy,
						 const LoggerPrx& logger,
						 bool& solved,
						 const ::Ice::Current& current);

    virtual void cancel(const string& jobid,
						const ::Ice::Current& current);

    virtual string status(const ::Ice::Current& current);

    virtual void shutdown(const ::Ice::Current& current);
};

string SolverI::status(const ::Ice::Current& current) {
	cout << "status() called." << endl;
	return "A-OK";
}

string SolverI::solve(const string& jobid,
					  const string& axy,
					  const LoggerPrx& logger,
					  bool& solved,
					  const ::Ice::Current& current) {
	cout << "solve() called." << endl;
	return "OK";
}

void SolverI::cancel(const string& jobid,
					 const ::Ice::Current& current) {
	cout << "cancel() called." << endl;
}

void SolverI::shutdown(const ::Ice::Current& current) {
	cout << "shutdown() called." << endl;
}


class Server : virtual public Ice::Application {
public:
    virtual int run(int x, char* y[]) {
		cout << "run() called.  x=" << x << endl;
		//cout << "y=" << y << endl;
		for (int i=0; i<x; i++)
			cout << "  y[" << i << "] = " << y[i] << endl;
		Ice::CommunicatorPtr ic = this->communicator();
        Ice::ObjectAdapterPtr adapter
            = ic->createObjectAdapterWithEndpoints
			("SimplePrinterAdapter", "default -p 10000");
        Ice::ObjectPtr object = new SolverI;
        adapter->add(object,
                     ic->stringToIdentity("SimplePrinter"));
        adapter->activate();
        ic->waitForShutdown();
        return 0;
    }
};

int main(int argc, char* argv[]) {
    cout << "SolverServer starting.  Command-line args:" << endl;
	for (int i=0; i<argc; i++) 
		cout << "  " << argv[i] << endl;

    Server s;
    return s.main(argc, argv);
}

/*
	int status = 0;
    Ice::CommunicatorPtr ic;
    try {
        ic = Ice::initialize(argc, argv);
        Ice::ObjectAdapterPtr adapter
            = ic->createObjectAdapterWithEndpoints(
                "SimplePrinterAdapter", "default -p 10000");
        Ice::ObjectPtr object = new SolverI;
        adapter->add(object,
                     ic->stringToIdentity("SimplePrinter"));
        adapter->activate();
        ic->waitForShutdown();
    } catch (const Ice::Exception& e) {
        cerr << e << endl;
        status = 1;
    } catch (const char* msg) {
        cerr << msg << endl;
        status = 1;
    }
    if (ic) {
        try {
            ic->destroy();
        } catch (const Ice::Exception& e) {
            cerr << e << endl;
            status = 1;
        }
    }
    return status;
}

 */
