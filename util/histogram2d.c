/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "os-features.h"
#include "histogram2d.h"
#include "errors.h"

static histogram2d* hist_new(int nx, int ny) {
    histogram2d* h = calloc(1, sizeof(histogram2d));
    if (!h) {
        ERROR("Couldn't allocate a histogram2d.");
        return NULL;
    }
    h->hist = calloc((size_t)nx*(size_t)ny, sizeof(int));
    if (!h->hist) {
        ERROR("Couldn't allocate a histogram2d with %ix%i bins.", nx, ny);
        free(h);
        return NULL;
    }
    h->NX = nx;
    h->NY = ny;
    return h;
}


histogram2d* histogram2d_new_nbins(double minX, double maxX, int NbinsX,
                                   double minY, double maxY, int NbinsY) {
    double binsizeX = (maxX - minX) / (double)(NbinsX);
    double binsizeY = (maxY - minY) / (double)(NbinsY);
    histogram2d* h = hist_new(NbinsX, NbinsY);
    h->minx = minX;
    //h->maxx = maxX;
    h->miny = minY;
    //h->maxy = maxY;
    h->binsizex = binsizeX;
    h->binsizey = binsizeY;
    h->NX = NbinsX;
    h->NY = NbinsY;
    h->edgex = HIST2D_TRUNCATE;
    h->edgey = HIST2D_TRUNCATE;
    return h;
}


void histogram2d_free(histogram2d* h) {
    free(h->hist);
    free(h);
}

int histogram2d_add(histogram2d* h, double x, double y) {
    int binx = (x - h->minx) / h->binsizex;
    int biny = (y - h->miny) / h->binsizey;
    int bin;
    if (h->edgex == HIST2D_TRUNCATE)
        binx = MIN(h->NX-1, MAX(0, binx));
    else if (h->edgex == HIST2D_DISCARD) {
        if (binx < 0 || binx >= h->NX)
            return HIST2D_DISCARDED_X;
    } else
        assert(0);

    if (h->edgey == HIST2D_TRUNCATE)
        biny = MIN(h->NY-1, MAX(0, biny));
    else if (h->edgey == HIST2D_DISCARD) {
        if (biny < 0 || biny >= h->NY)
            return HIST2D_DISCARDED_Y;
    } else
        assert(0);

    bin = biny * h->NX + binx;
    h->hist[bin]++;
    return bin;
}

void histogram2d_set_x_edges(histogram2d* h, int edgepolicy) {
    h->edgex = edgepolicy;
}

void histogram2d_set_y_edges(histogram2d* h, int edgepolicy) {
    h->edgey = edgepolicy;
}

