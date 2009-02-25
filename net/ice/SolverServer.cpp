#include <Ice/Ice.h>
#include <IceUtil/Thread.h>

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>

#include "Solver.h"

extern "C" {
#include "backend.h"
#include "log.h"
#include "fitsioutils.h"
}

using namespace std;
using namespace SolverIce;

class SolverI : public Solver {
public:
	SolverI(const string& progname, int scale);

	int init();

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
	backend_t* backend;
};

SolverI::SolverI(const string& progname, int scale) {
	cout << "Solver constructor: name " << progname << ", scale " << scale << endl;

	char configfnbuf[256];
	sprintf(configfnbuf, "/data1/dstn/dsolver/backend-config/backend-scale%i.cfg", scale);
	
	configfn = string(configfnbuf);
	cout << "Using config file " << configfn << endl;

	backend = backend_new();
	log_init((log_level)3);
	log_set_thread_specific();
	fits_use_error_system();
}

int SolverI::init() {
	if (backend_parse_config_file(backend, configfn.c_str())) {
		cout << "ERROR parsing config file" << endl;
		return -1;
	}
}

string SolverI::status(const ::Ice::Current& current) {
	cout << "status() called." << endl;
	return "A-OK";
}

class LogMessagePiper : public IceUtil::Thread {
public:
	LogMessagePiper(int therpipe, const LoggerPrx& thelogger) {
		quitNow = false;
		rpipe = therpipe;
		logger = thelogger;
		logger->logmessage("Hello from LogMessagePiper.\n");
	}

	virtual void run() {
		for (;;) {
			char buf[1024];
			int nread;
			if (quitNow) {
				cout << "Quitting at user request" << endl;
				break;
			}
			nread = read(rpipe, buf, sizeof(buf));
			if (nread == 0) {
				cout << "Hit end-of-file on log message pipe." << endl;
				break;
			}
			if (nread == -1) {
				cout << "Error reading from log message pipe: " << strerror(errno) << endl;
				break;
			}
			cout << "Read " << nread << " bytes from log message pipe." << endl;
			cout << "Sending to remote logger..." << endl;
			string logstr = string(buf, nread);
			logger->logmessage(logstr);
			cout << "Sent to remote logger." << endl;
			sleep(1);
		}
    }

	void quit() {
		quitNow = true;
	}

private:
	int rpipe;
	LoggerPrx logger;
	bool quitNow;
};

string SolverI::solve(const string& jobid,
					  const string& axydata,
					  const LoggerPrx& logger,
					  bool& solved,
					  const ::Ice::Current& current) {
	cout << "solve() called." << endl;
	cout << "  jobid = " << jobid << endl;

	char* tempdir = "/tmp";

	char templ[256];
	sprintf(templ, "%s/backend-%s-XXXXXX", tempdir, jobid.c_str());
	char* mydir = mkdtemp(templ);

	string cancelfn = string(mydir) + "/cancel";
	string axyfn = string(mydir) + "/job.axy";
	FILE* f = fopen(axyfn.c_str(), "w");
	if (fwrite(axydata.c_str(), 1, axydata.length(), f) != axydata.length() ||
		fclose(f)) {
		cout << "Failed to write axy data to file " << axyfn << endl;
		return "failed to write axy data";
	}
	cout << "Wrote axyfn = " << axyfn << endl;


	int pipes[2];
	if (pipe(pipes)) {
		cout << "Error creating pipe for log messages." << endl;
		return "Error creating pipe for log messages.";
	}
	/*
	 long flags = fcntl(pipes[0], F_GETFL);
	 if (fcntl(pipes[0], F_SETFL, flags | O_NONBLOCK)) {
	 cout << "Failed to set read pipe non-blocking.\n" << endl;
	 }
	 */
	log_to_fd(pipes[1]);

	LogMessagePiper* t = new LogMessagePiper(pipes[0], logger);
	IceUtil::ThreadControl tc = t->start();

	job_t* job = backend_read_job_file(backend, axyfn.c_str());
	if (!job) {
		cout << "Failed to read job file " << axyfn << endl;
		return "failed to read job file";
	}

	job_set_base_dir(job, mydir);
	job_set_cancel_file(job, cancelfn.c_str());

	cout << "backend_run_job()" << endl;
	backend_run_job(backend, job);
	cout << "backend_run_job() done!" << endl;

	job_free(job);

	cout << "cleaning up logging thread..." << endl;
	log_to(stdout);
	
	if (t->isAlive()) {
		t->quit();
	}
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
		cout << "Creating solver..." << endl;
		SolverI* si = new SolverI(progname, scale);
		cout << "Reading config file..." << endl;
		if (si->init()) {
			cout << "Error reading config file." << endl;
			return -1;
		}
		SolverPtr solver = si;
		adapter->add(solver, myid);
		cout << "Activating adapter..." << endl;
		adapter->activate();
		cout << "Running!" << endl;
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
