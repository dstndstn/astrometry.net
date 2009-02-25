#include <Ice/Ice.h>

#include "Solver.h"

using namespace std;
using namespace SolverIce;

class SolverI : public Solver {
public:
	SolverI(const string& progname, int scale);

    virtual string solve(const string& jobid,
						 const string& axy,
						 const LoggerPrx& logger,
						 bool& solved,
						 const ::Ice::Current& current);

    virtual void cancel(const string& jobid,
						const ::Ice::Current& current);

    virtual string status(const ::Ice::Current& current);

    virtual void shutdown(const ::Ice::Current& current);

private:
	string configfn;
};

SolverI::SolverI(const string& progname, int scale) {
	cout << "Solver constructor: name " << progname << ", scale " << scale << endl;

	char configfnbuf[256];
	sprintf(configfnbuf, "/data1/dstn/dsolver/backend-config/backend-scale%i.cfg", scale);
	
	configfn = string(configfnbuf);
	cout << "Using config file " << configfn << endl;
}

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
    virtual int run(int argc, char* args[]) {
		cout << "run() called." << endl;
		cout << "Args:" << endl;
		for (int i=0; i<argc; i++)
			cout << "  " << args[i] << endl;

		if (argc != 2) {
			cout << "Need one arg: index scale." << endl;
			return -1;
		}

		int scale = atoi(args[1]);

		Ice::CommunicatorPtr ic = this->communicator();
		Ice::PropertiesPtr props = ic->getProperties();
		Ice::ObjectAdapterPtr adapter = ic->createObjectAdapter("OneSolver");
		string idstr = props->getProperty("Identity");
		Ice::Identity myid = ic->stringToIdentity(idstr);
		string progname = props->getProperty("Ice.ProgramName");
		SolverPtr solver = new SolverI(progname, scale);
		adapter->add(solver, myid);
		adapter->activate();
        ic->waitForShutdown();
		return 0;
		/*
		 Ice::ObjectAdapterPtr adapter
		 = ic->createObjectAdapterWithEndpoints
		 ("SimplePrinterAdapter", "default -p 10000");
		 Ice::ObjectPtr object = new SolverI;
		 adapter->add(object,
		 ic->stringToIdentity("SimplePrinter"));
		 adapter->activate();
		 return 0;
		 */
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
