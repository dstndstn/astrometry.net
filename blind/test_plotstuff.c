#include <math.h>
#include <stdio.h>
#include <string.h>

#include "cutest.h"
#include "plotstuff.h"
#include "plotfill.h"
#include "plotxy.h"
#include "plotimage.h"
#include "log.h"

void test_plot_wcs1(CuTest* tc) {
	plot_args_t myargs;
	plot_args_t* pargs = &myargs;
	int W, H;
	//unsigned char* img;
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

}
