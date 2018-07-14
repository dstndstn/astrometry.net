/*
 # This file is part of libkd.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>

#define GET(x) (arr[(x)])
#define ELEM_SWAP(il, ir) {                     \
        if ((il) != (ir)) {                     \
            int tmp = arr[(il)];                \
            arr[(il)] = arr[(ir)];              \
            arr[(ir)] = tmp; }}

int main() {
    int* arr;
    int N = 30;
    int mx = 100;
    int i;
    int L = 0;
    int R = N-1;
    time_t t;

    /*
     t = 981;
     */
    t = time(NULL);
    t = t % 1000;
    printf("t=%i\n", (int)t);
    srand(t);

    arr = malloc(N * sizeof(int));
    for (i=0; i<N; i++)
        arr[i] = rand() % mx;

    {
        int low, high, median, middle, ll, hh;
        int nless, nequal;
        int equal, greater;
        int midval;
        int midpos;

        /* Find the median and partition the data */
        low = L;
        high = R;
        median = (low + high) / 2;
        while(1) {
            if (high <= (low+1))
                break;

            /* Find median of low, middle and high items; swap into position low */
            middle = (low + high) / 2;

            printf("low=%i, middle=%i, high=%i.\n", low, middle, high);
            printf("median=%i\n", median);

            if (GET(middle) > GET(high))
                ELEM_SWAP(middle, high);
            if (GET(low) > GET(high))
                ELEM_SWAP(low, high);
            if (GET(middle) > GET(low))
                ELEM_SWAP(middle, low);
			
            /* Swap low item (now in position middle) into position (low+1) */
            ELEM_SWAP(middle, low + 1) ;

            midval = GET(low);
            printf("middle value: %i\n", midval);

            printf("beginning:\n");
            for (i=L; i<=R; i++) {
                if (i == low) printf("( ");
                printf("%i ", GET(i));
                if (i == high) printf(") ");
            }
            printf("\n");

            /* Count the number of items in each category. */
            nless = nequal = 0;
            for (i=low; i<=high; i++) {
                if (GET(i) < midval)
                    nless++;
                else if (GET(i) == midval)
                    nequal++;
            }
            /* "equal" is the index where the items equal to "midval" will
             end up. */
            equal = low + nless;
            /* "greater" is the index where the items greater than "midval"
             will end up. */
            greater = equal + nequal;

            /* Nibble from each end towards middle, swapping items when stuck */
            ll = low + 1;
            hh = high;
            for (;;) {
                while ((GET(ll) < midval) && (ll <= hh))
                    ll++;
                while ((GET(hh) >= midval) && (ll <= hh))
                    hh--;
                if (hh < ll)
                    break;
                ELEM_SWAP(ll, hh);
            }
			
            /* Swap middle item (in position low) back into correct position */
            ELEM_SWAP(low, hh);

            /* Where did the middle value end up? */
            midpos = hh;

            printf("after swapping middle value back into place:\n");
            for (i=L; i<=R; i++) {
                if (i == low) printf("( ");
                if (i == midpos) printf("*");
                printf("%i ", GET(i));
                if (i == high) printf(") ");
            }
            printf("\n");

            for (i=low; i<equal; i++)
                assert(GET(i) < midval);
            for (i=equal; i<=high; i++)
                assert(GET(i) >= midval);

            /* Collect all items equal to the middle value.
             At this point items less than "midval" are in the left part
             of the array, and items equal to or greater than "midval" are
             in the right side.
             Nibble the right side, moving "=" and ">" items into their
             respective halves.
             */
            ll = equal;
            hh = high;
            for (;;) {
                while ((GET(ll) == midval) && (ll < greater))
                    ll++;
                while ((GET(hh) > midval) && (hh >= ll))
                    hh--;
                if (hh < ll)
                    break;
                ELEM_SWAP(ll, hh);
            }

            printf("after collecting equal items:\n");
            for (i=L; i<=R; i++) {
                if (i == low) printf("( ");
                if (i == midpos) printf("*");
                printf("%i ", GET(i));
                if (i == high) printf(") ");
            }
            printf("\n");

            for (i=low; i<equal; i++)
                assert(GET(i) < midval);
            for (i=equal; i<greater; i++)
                assert(GET(i) == midval);
            for (i=greater; i<=high; i++)
                assert(GET(i) > midval);

            /* You might want to choose some value other than the
             median to produce splits of closer to equal size. */
            {
                int nl = equal - L;
                int nh = R + 1 - greater;
                printf("nl=%i, ne=%i, nh=%i.\n", nl, nequal, nh);
                printf("greater=%i, high=%i.\n", greater, high);
                /* "greater <= high" ensures that there is at least one
                 element in the high partition. */
                if ((greater <= high) &&
                    (nh > nl) && (nl + nequal >= nh)) {
                    /* The high partition is already bigger, so select
                     the first value in the high partition, which means
                     the middle partition will end up to the left of
                     the new "median". */
                    median = greater;
                    printf("Changed \"median\" to %i.\n", median);
                    low = greater;
                    continue;
                }
            }

            /* Is the median in the "<", "=", or ">" partition? */
            if (median < equal) {
                // low is unchanged.
                high = equal - 1;
                printf("median is in low partition.\n");
            } else if (median < greater) {
                /* the median is inside the "=" partition; we've
                 isolated the median. */
                low = high = equal;
                printf("median is in middle partition.\n");
                break;
            } else {
                // high is unchanged.
                low = greater;
                printf("median is in high partition.\n");
            }
        }

        if (high == low + 1) {  /* Two elements only */
            if (GET(low) > GET(high))
                ELEM_SWAP(low, high);
        }

        median = low;
        midval = GET(median);

        printf("end:\n");
        for (i=L; i<=R; i++) {
            if (i == median)
                printf("*");
            printf("%i ", GET(i));
        }
        printf("\n");

        /* check that it worked. */
        for (i = L; i < median; i++) {
            assert(GET(i) < midval);
        }
        for (i = median; i <= R; i++) {
            /* Assert contention: i just changed this assert to ">" from ">="
             * because for the inttree, i need strict median guarentee
             * (i.e. all the median values are on one side or the other of
             * the return value of this function) If this causes problems
             * let me know --k */
            assert(GET(i) >= midval);
        }
    }

    free(arr);

    return 0;
}



