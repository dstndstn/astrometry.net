#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#include "cutest.h"

#include "os-features.h"

char* true_baton_val = "MYBATON";

CuTest* g_tc = NULL;

static int QSORT_COMPARISON_FUNCTION(compints, void* token, const void* v1, const void* v2) {
  const int* i1 = v1;
  const int* i2 = v2;

  CuAssertPtrEquals(g_tc, true_baton_val, token);

  if (*i1 < *i2)
    return -1;
  if (*i1 > *i2)
    return 1;
  return 0;
}

void test_qsort_r(CuTest* tc) {
  int array[] = { 4, 17, 88, 34, 12, 12, 17 };
  int N = sizeof(array)/sizeof(int);

  void* token = true_baton_val;
  g_tc = tc;

  QSORT_R(array, N, sizeof(int), token, compints);

  CuAssertIntEquals(tc,  4, array[0]);
  CuAssertIntEquals(tc, 12, array[1]);
  CuAssertIntEquals(tc, 12, array[2]);
  CuAssertIntEquals(tc, 17, array[3]);
  CuAssertIntEquals(tc, 17, array[4]);
  CuAssertIntEquals(tc, 34, array[5]);
  CuAssertIntEquals(tc, 88, array[6]);

}
