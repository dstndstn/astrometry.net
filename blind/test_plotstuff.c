/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "cutest.h"
#include "plotstuff.h"
#include "plotfill.h"
#include "plotxy.h"
#include "plotimage.h"
#include "log.h"
#include "cairoutils.h"

void test_plot_wcs1(CuTest* tc) {
    plot_args_t myargs;
    plot_args_t* pargs = &myargs;
    int W, H;
    plotxy_t* xy;
    //plotimage_t* img;


    log_init(LOG_VERB);

    W = H = 5;
    plotstuff_init(pargs);
    plotstuff_set_size(pargs, W, H);
    pargs->outformat = PLOTSTUFF_FORMAT_PNG;
    pargs->outfn = "test-out1.png";
    plotstuff_set_color(pargs, "black");
    plotstuff_run_command(pargs, "fill");
    plotstuff_output(pargs);
    plotstuff_free(pargs);

    // perfectly centered circle.
    plotstuff_init(pargs);
    plotstuff_set_size(pargs, W, H);
    pargs->outformat = PLOTSTUFF_FORMAT_PNG;
    pargs->outfn = "test-out2.png";
    plotstuff_set_color(pargs, "black");
    plotstuff_run_command(pargs, "fill");
    xy = plotstuff_get_config(pargs, "xy");
    plot_xy_vals(xy, 3, 3);
    plotstuff_set_color(pargs, "white");
    plotstuff_set_marker(pargs, "circle");
    plotstuff_set_markersize(pargs, 1);
    plotstuff_run_command(pargs, "xy");
    plotstuff_output(pargs);
    plotstuff_free(pargs);

    // perfectly centered circle.
    plotstuff_init(pargs);
    plotstuff_set_size(pargs, W, H);
    pargs->outformat = PLOTSTUFF_FORMAT_PNG;
    pargs->outfn = "test-out3.png";
    plotstuff_set_color(pargs, "black");
    plotstuff_run_command(pargs, "fill");
    xy = plotstuff_get_config(pargs, "xy");
    plot_xy_vals(xy, 2, 2);
    plot_xy_set_offsets(xy, 0, 0);
    plotstuff_set_color(pargs, "white");
    plotstuff_set_marker(pargs, "circle");
    plotstuff_set_markersize(pargs, 1);
    plotstuff_run_command(pargs, "xy");
    plotstuff_output(pargs);
    plotstuff_free(pargs);

    {
        unsigned char* img;
        int ww, hh;
        int i;

        img = cairoutils_read_png("test-out2.png", &ww, &hh);
        CuAssertPtrNotNull(tc, img);
        CuAssertIntEquals(tc, W, ww);
        CuAssertIntEquals(tc, H, hh);

        printf("image:\n");
        for (i=0; i<W*H; i++) {
            printf("%02x  ", (int)img[i*4]);
            if (i%W == (W-1))
                printf("\n");
        }
        printf("\n");

        /*
         00  00  00  00  00  
         00  80  e8  80  00  
         00  f3  44  f3  00  
         00  7c  e8  7c  00  
         00  00  00  00  00  
         */

    }

}
