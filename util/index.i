
%module index_c

%{
#include "index.h"
#include "codekd.h"
#include "starkd.h"

#define true 1
#define false 0

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

double* code_alloc(int DC);
void code_free(double* code);
double code_get(double* code, int i);

long codekd_addr(index_t* ind);

/*
void codetree_get_N(codetree* s, unsigned int codeid_start, int N, double* code);
*/


