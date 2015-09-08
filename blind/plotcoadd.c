/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/
#include <string.h>
#include <math.h>
#include <sys/param.h>

#include "plotcoadd.h"
#include "cairoutils.h"
#include "ioutils.h"
#include "log.h"
#include "errors.h"

const plotter_t plotter_coadd = {
	.name = "coadd",
	.init = plot_coadd_init,
	.command = plot_coadd_command,
	.doplot = plot_coadd_plot,
	.free = plot_coadd_free
};

plotcoadd_t* plot_coadd_get(plot_args_t* pargs) {
	return plotstuff_get_config(pargs, "coadd");
}

void* plot_coadd_init(plot_args_t* plotargs) {
	plotcoadd_t* args = calloc(1, sizeof(plotcoadd_t));
	return args;
}

int plot_coadd_plot(const char* command,
					cairo_t* cairo, plot_args_t* pargs, void* baton) {
	plotcoadd_t* args = (plotcoadd_t*)baton;
	return 0;
}

int plot_coadd_command(const char* cmd, const char* cmdargs,
					   plot_args_t* pargs, void* baton) {
	plotcoadd_t* args = (plotcoadd_t*)baton;
	if (streq(cmd, "coadd_file")) {
		//plot_image_set_filename(args, cmdargs);
	} else {
		ERROR("Did not understand command \"%s\"", cmd);
		return -1;
	}
	return 0;
}

void plot_coadd_free(plot_args_t* plotargs, void* baton) {
	plotcoadd_t* args = (plotcoadd_t*)baton;
	free(args);
}

