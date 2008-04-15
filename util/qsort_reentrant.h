#ifndef QSORT_REENTRANT_H__
#define QSORT_REENTRANT_H__

typedef int cmp_t(void *, const void *, const void *);

void qsort_r(void *base, size_t Nelements, size_t elementSize,
             void *userdata, cmp_t* comparison_function);

#endif  // QSORT_REENTRANT_H__
