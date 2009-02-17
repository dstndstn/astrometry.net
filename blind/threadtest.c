
#include <pthread.h>

#include "backend.h"
#include "log.h"
#include "fitsioutils.h"

backend_t* be;

//pthread_mutex_t read_job_mutex;

void* threadfunc(void* arg) {
    char* jobfn = arg;

	logverb("Hello from thread-land!\n");
    
    //pthread_mutex_lock(&read_job_mutex);
    job_t* job = backend_read_job_file(be, jobfn);
    //pthread_mutex_unlock(&read_job_mutex);

    backend_run_job(be, job);
    job_free(job);
    return NULL;
}

int main(int argc, char** args) {
    pthread_t thread1;
    pthread_t thread2;
    pthread_attr_t attr;
    char* job1 = "job1.axy";
    char* job2 = "job2.axy";

    fits_use_error_system();
    
    log_init(LOG_VERB);
    log_set_thread_specific();

	logverb("Hello world!\n");

    be = backend_new();
    backend_parse_config_file(be, "backend.cfg");

    pthread_mutex_init(&read_job_mutex, NULL);

    pthread_attr_init(&attr);
    pthread_create(&thread1, &attr, threadfunc, job1);
    pthread_create(&thread2, &attr, threadfunc, job2);

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    pthread_mutex_destroy(&read_job_mutex);

    backend_free(be);

    return 0;
}


