/*
 # This file is part of libkd.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <time.h>

typedef double dtype;

static int kdtree_quickselect_partition(dtype *arr, unsigned int *parr,
                                        int L, int R, int D, int d);

int main(int argc, char** args) {
    int i, d;
    int D = 3;
    int N = 10000;
    int T = 1000, t;
    dtype* arr;
    unsigned int* perm;

    srand(time(NULL));

    perm = malloc(N * sizeof(int));
    arr = malloc(N * D * sizeof(dtype));
    for (t=0; t<T; t++) {
        for (i=0; i<N; i++) {
            perm[i] = i;
            for (d=0; d<D; d++) {
                //arr[i*D + d] = rand() / (double)RAND_MAX;
                arr[i*D + d] = (int)(100.0 * rand() / (double)RAND_MAX);
            }
        }
        for (d=0; d<D; d++) {
            kdtree_quickselect_partition(arr, perm, 0, N-1, D, d);
        }
    }
    return 0;
}


#define GET(x) (arr[(x)*D+d])
#if defined(KD_DIM)
#define ELEM_SWAP(il, ir) {                             \
        if ((il) != (ir)) {                             \
            tmpperm  = parr[il];                        \
            assert(tmpperm != -1);                      \
            parr[il] = parr[ir];                        \
            parr[ir] = tmpperm;                         \
            { int d; for (d=0; d<D; d++) {              \
                    tmpdata[0] = arr[(il)*D+d];         \
                    arr[(il)*D+d] = arr[(ir)*D+d];      \
                    arr[(il)*D+d] = tmpdata[0]; }}}}
#else
#define ELEM_SWAP(il, ir) {                                     \
        if ((il) != (ir)) {                                     \
            tmpperm  = parr[il];                                \
            assert(tmpperm != -1);                              \
            parr[il] = parr[ir];                                \
            parr[ir] = tmpperm;                                 \
            memcpy(tmpdata,    arr+(il)*D, D*sizeof(dtype));    \
            memcpy(arr+(il)*D, arr+(ir)*D, D*sizeof(dtype));    \
            memcpy(arr+(ir)*D, tmpdata,    D*sizeof(dtype)); }}
#endif
#define ELEM_ROT(iA, iB, iC) {                                  \
        tmpperm  = parr[iC];                                    \
        parr[iC] = parr[iB];                                    \
        parr[iB] = parr[iA];                                    \
        parr[iA] = tmpperm;                                     \
        assert(tmpperm != -1);                                  \
        memcpy(tmpdata,    arr+(iC)*D, D*sizeof(dtype));        \
        memcpy(arr+(iC)*D, arr+(iB)*D, D*sizeof(dtype));        \
        memcpy(arr+(iB)*D, arr+(iA)*D, D*sizeof(dtype));        \
        memcpy(arr+(iA)*D, tmpdata,    D*sizeof(dtype)); }

static int kdtree_quickselect_partition(dtype *arr, unsigned int *parr, int L, int R, int D, int d) {
    dtype* tmpdata = alloca(D * sizeof(dtype));
    dtype midval;
    int tmpperm = -1, i;
    int low, middle, high;
    int median;

#if defined(KD_DIM)
    // tell the compiler this is a constant...
    D = KD_DIM;
#endif

    /* sanity is good */
    assert(R >= L);

    /* Find the median and partition the data */
    low = L;
    high = R;
    median = (low + high + 1) / 2;
    while(1) {
        dtype vals[3];
        dtype tmp;
        dtype pivot;
        int i,j;
        int iless, iequal, igreater;
        int endless, endequal, endgreater;
        int nless, nequal;

        if (high == low)
            break;

        /* Choose the pivot: find the median of the values in low, middle, and high
         positions. */
        middle = (low + high) / 2;
        vals[0] = GET(low);
        vals[1] = GET(middle);
        vals[2] = GET(high);
        /* Bubblesort the three elements. */
        for (i=0; i<2; i++)
            for (j=0; j<(2-i); j++)
                if (vals[j] > vals[j+1]) {
                    tmp = vals[j];
                    vals[j] = vals[j+1];
                    vals[j+1] = tmp;
                }
        /* unrolled:
         if (vals[0] > vals[1]) {
         tmp = vals[0];
         vals[0] = vals[1];
         vals[1] = tmp;
         }
         if (vals[1] > vals[2]) {
         tmp = vals[1];
         vals[1] = vals[2];
         vals[2] = tmp;
         }
         if (vals[0] > vals[1]) {
         tmp = vals[0];
         vals[0] = vals[1];
         vals[1] = tmp;
         }
         */
        assert(vals[0] <= vals[1]);
        assert(vals[1] <= vals[2]);

        pivot = vals[1];

        /* Count the number of items that are less than, equal to, and greater than the pivot. */
        nless = nequal = 0;
        for (i=low; i<=high; i++) {
            if (GET(i) < pivot)
                nless++;
            else if (GET(i) == pivot)
                nequal++;
        }

        /* These are the indices where the <, =, and > entries will start. */
        iless = low;
        iequal = low + nless;
        igreater = low + nless + nequal;

        /* These are the indices where they will end; ie the elements less than the
         pivot will live in [iless, endless).  (But note that we'll be incrementing
         "iequal" et al in the loop below.) */
        endless = iequal;
        endequal = igreater;
        endgreater = high+1;

        int n1, n2, n3, n4, n5, n6;
        n1 = n2 = n3 = n4 = n5 = n6 = 0;

        while (1) {
            /* Find an element in the "less" section that is out of place. */
            while ( (GET(iless) < pivot) && (iless < endless) )
                iless++;

            /* Find an element in the "equal" section that is out of place. */
            while ( (GET(iequal) == pivot) && (iequal < endequal) )
                iequal++;

            /* Find an element in the "greater" section that is out of place. */
            while ( (GET(igreater) > pivot) && (igreater < endgreater) )
                igreater++;


            /* We're looking at three positions, and each one has three cases:
             we're finished that segment, or the element we're looking at belongs in
             either of the other two segments.  This yields 27 cases, but many of them
             are ruled out because, eg, if the element at "iequal" belongs in the "less"
             segment, then we can't be done the "less" segment.

             There are only 6 cases to handle:
			   
             ---------------------------------------------
             case   iless    iequal   igreater    action
             ---------------------------------------------
             1      D        D        D           done
             2      G        ?        L           swap l,g
             3      E        L        ?           swap l,e
             4      ?        G        E           swap e,g
             5      E        G        L           rotate A
             6      G        L        E           rotate B
             ---------------------------------------------

			   
             legend:
             D: done
             L: (element < pivot)
             E: (element == pivot)
             G: (element > pivot)
             */

            /* case 1: done? */
            if ((iless == endless) && (iequal == endequal) && (igreater == endgreater)) {
                n1++;
                break;
            }

            /* case 2: swap l,g */
            if ((iless < endless) && (igreater < endgreater) &&
                (GET(iless) > pivot) && (GET(igreater) < pivot)) {
                ELEM_SWAP(iless, igreater);
                assert(GET(iless) < pivot);
                assert(GET(igreater) > pivot);
                n2++;
                continue;
            }

            /* cases 3,4,5,6 */
            assert(iequal < endequal);
            if (GET(iequal) > pivot) {
                /* cases 4,5: */
                assert(igreater < endgreater);
                if (GET(igreater) == pivot) {
                    /* case 4: swap e,g */
                    ELEM_SWAP(iequal, igreater);
                    assert(GET(iequal) == pivot);
                    assert(GET(igreater) > pivot);
                    n4++;
                } else {
                    /* case 5: rotate. */
                    assert(GET(iless) == pivot);
                    assert(GET(iequal) > pivot);
                    assert(GET(igreater) < pivot);
                    ELEM_ROT(iless, iequal, igreater);
                    assert(GET(iless) < pivot);
                    assert(GET(iequal) == pivot);
                    assert(GET(igreater) > pivot);
                    n5++;
                }
            } else {
                /* cases 3,6 */
                assert(GET(iequal) < pivot);
                assert(iless < endless);
                if (GET(iless) == pivot) {
                    /* case 3: swap l,e */
                    ELEM_SWAP(iless, iequal);
                    assert(GET(iless) < pivot);
                    assert(GET(iequal) == pivot);
                    n3++;
                } else {
                    dtype vequal, vless, vgreater;
					
                    vequal = GET(iequal);
                    vless = GET(iless);
                    vgreater = GET(igreater);
                    assert(GET(iless) > pivot);
                    assert(GET(iequal) < pivot);
                    assert(GET(igreater) == pivot);
					
                    /* case 6: rotate. */
                    ELEM_ROT(igreater, iequal, iless);
                    vequal = GET(iequal);
                    vless = GET(iless);
                    vgreater = GET(igreater);
                    assert(GET(iless) < pivot);
                    assert(GET(iequal) == pivot);
                    assert(GET(igreater) > pivot);
                    n6++;
                }
            }
        }

        printf("% 8i % 8i % 8i % 8i % 8i % 8i\n", n1, n2, n3, n4, n5, n6);

        /* Reset the indices of where the segments start. */
        iless = low;
        iequal = low + nless;
        igreater = low + nless + nequal;

        /* Assert that "<" values are in the "less" partition, "=" values are in the
         "equal" partition, and ">" values are in the "greater" partition. */
        for (i=iless; i<iequal; i++)
            assert(GET(i) < pivot);
        for (i=iequal; i<igreater; i++)
            assert(GET(i) == pivot);
        for (i=igreater; i<=high; i++)
            assert(GET(i) > pivot);

        /* Is the median in the "<", "=", or ">" partition? */
        if (median < iequal)
            /* median is in the "<" partition.
             low is unchanged.
             */
            high = iequal - 1;
        else if (median < igreater) {
            /* the median is inside the "=" partition; we're done! */
            low = high = median;
            break;
        } else
            /* median is in the ">" partition.
             high is unchanged. */
            low = igreater;
    }

    /*
     if (high == low + 1) {
     if (GET(low) > GET(high))
     ELEM_SWAP(low, high);
     }
     median = low;
     assert(median != 0);
     midval = GET(median);
     */

    /* check that it worked. */
    midval = GET(median);
    for (i=L; i<median; i++)
        assert(GET(i) <= midval);
    for (i=median; i<=R; i++)
        assert(GET(i) >= midval);

    assert (L < median);
    assert (median <= R);

    return median;
}
#undef ELEM_SWAP
#undef ELEM_ROT
#undef GET
