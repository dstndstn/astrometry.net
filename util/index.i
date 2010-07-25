
%module index_c

%{
#include "index.h"
#include "codekd.h"
#include "starkd.h"
#include "qidxfile.h"
#include "log.h"

#define true 1
#define false 0

/**
For returning single codes and quads as python lists, do something like this:

%typemap(out) float [ANY] {
  int i;
  $result = PyList_New($1_dim0);
  for (i = 0; i < $1_dim0; i++) {
    PyObject *o = PyFloat_FromDouble((double) $1[i]);
    PyList_SetItem($result,i,o);
  }
}
**/

double* code_alloc(int DC) {
	 return malloc(DC * sizeof(double));
}
void code_free(double* code) {
	 free(code);
}
double code_get(double* code, int i) {
	return code[i];
}

long codekd_addr(index_t* ind) {
	 return (long)ind->codekd;
}
long starkd_addr(index_t* ind) {
	 return (long)ind->starkd;
}

long quadfile_addr(index_t* ind) {
	 return (long)ind->quads;
}

long qidxfile_addr(qidxfile* qf) {
	 return (long)qf;
}

/*
void codetree_get_N(codetree* s, unsigned int codeid_start, int N, double* code) {
	 int i;
	 int DQ = codetree_D(s);
	 for (i=0; i<N; i++) {
	 codetree_get(s, codeid_start + i, code + i*DQ);
	 }
}
*/

%}

%include "index.h"
%include "codekd.h"
%include "starkd.h"
%include "qidxfile.h"

double* code_alloc(int DC);
void code_free(double* code);
double code_get(double* code, int i);

long codekd_addr(index_t* ind);
long starkd_addr(index_t* ind);
long quadfile_addr(index_t* ind);
long qidxfile_addr(qidxfile* qf);

void log_init(int level);

/*
void codetree_get_N(codetree* s, unsigned int codeid_start, int N, double* code);
*/


