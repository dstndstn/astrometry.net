#include <sys/param.h>

#include "mercrender.h"
#include "tilerender.h"

struct mercargs {
	render_args_t* args;
	float* fluximg;
	merctree* merc;
    int symbol;
};
typedef struct mercargs mercargs;

static void leaf_node(const kdtree_t* kd, int node, void* vargs);
static void expand_node(const kdtree_t* kd, int node, void* vargs);
static void add_star_merc(double xp, double yp, merc_flux* flux, mercargs* args);

float* mercrender_file(char* fn, render_args_t* args, int symbol) {
	float* fluximg;
	merctree* merc = merctree_open(fn);
	if (!merc) {
		fprintf(stderr, "Failed to open merctree %s\n", fn);
		return NULL;
	}

	fluximg = calloc(args->W*args->H*3, sizeof(float));
	if (!fluximg) {
		fprintf(stderr, "Failed to allocate flux image.\n");
		return NULL;
	}
	mercrender(merc, args, fluximg, symbol);
	merctree_close(merc);
	return fluximg;
}

void mercrender(merctree* merc, render_args_t* args,
				float* fluximg, int symbol) {
	double querylow[2], queryhigh[2];
	double xmargin, ymargin;
	mercargs margs;

	margs.args = args;
	margs.fluximg = fluximg;
	margs.merc = merc;
    margs.symbol = symbol;

	// Magic number 2: number of pixels around a star.
	xmargin = 2.0 / args->xpixelpermerc;
	ymargin = 2.0 / args->ypixelpermerc;

 	querylow [0] = MAX(0.0, args->xmercmin - xmargin);
	queryhigh[0] = MIN(1.0, args->xmercmax + xmargin);
	querylow [1] = MAX(0.0, args->ymercmin - ymargin);
	queryhigh[1] = MIN(1.0, args->ymercmax + ymargin);

	kdtree_nodes_contained(merc->tree, querylow, queryhigh,
						   expand_node, leaf_node, &margs);

	// Handle wrap-around requests.
	if (args->xmercmin < 0) {
		args->xmercmin += 1.0;
		args->xmercmax += 1.0;
		querylow [0] = MAX(0.0, args->xmercmin - xmargin);
		queryhigh[0] = MIN(1.0, args->xmercmax + xmargin);
		kdtree_nodes_contained(merc->tree, querylow, queryhigh,
							   expand_node, leaf_node, &margs);
		args->xmercmin -= 1.0;
		args->xmercmax -= 1.0;
	}
	if (args->xmercmax > 1) {
		args->xmercmin -= 1.0;
		args->xmercmax -= 1.0;
		querylow [0] = MAX(0.0, args->xmercmin - xmargin);
		queryhigh[0] = MIN(1.0, args->xmercmax + xmargin);
		kdtree_nodes_contained(merc->tree, querylow, queryhigh,
							   expand_node, leaf_node, &margs);
		args->xmercmin += 1.0;
		args->xmercmax += 1.0;
	}
}

static void expand_node(const kdtree_t* kd, int node, void* vargs) {
	int xp0, xp1, yp0, yp1;
	int D = 2;
	double bblo[D], bbhi[D];
	mercargs* margs = vargs;

	if (KD_IS_LEAF(kd, node)) {
		leaf_node(kd, node, margs);
		return;
	}

	// check if this whole box fits inside a pixel.
	if (!kdtree_get_bboxes(kd, node, bblo, bbhi)) {
		fprintf(stderr, "Error, node %i does not have bounding boxes.\n", node);
		exit(-1);
	}
	xp0 = xmerc2pixel(bblo[0], margs->args);
	xp1 = xmerc2pixel(bbhi[0], margs->args);
	if (xp1 == xp0) {
		yp0 = ymerc2pixel(bblo[1], margs->args);
		yp1 = ymerc2pixel(bbhi[1], margs->args);
		if (yp1 == yp0) {
			// This node fits inside a single pixel of the output image.
			add_star_merc(bblo[0], bblo[1], &(margs->merc->stats[node].flux), margs);
			return;
		}
	}

	expand_node(kd,  KD_CHILD_LEFT(node), margs);
	expand_node(kd, KD_CHILD_RIGHT(node), margs);
}

static void leaf_node(const kdtree_t* kd, int node, void* vargs) {
	int k;
	int L, R;
	mercargs* margs = vargs;

	L = kdtree_left(kd, node);
	R = kdtree_right(kd, node);

	for (k=L; k<=R; k++) {
		double pt[2];
		merc_flux* flux;

		kdtree_copy_data_double(kd, k, 1, pt);
		flux = margs->merc->flux + k;
		add_star_merc(pt[0], pt[1], flux, margs);
	}
}

int add_star(double xp, double yp, double rflux, double bflux, double nflux,
		float* fluximg, int render_symbol, render_args_t* args)
{
	// this is stupid
	int ndrops[] = {25, 18, 16, 25};
	int dx0[] =  { -2, -1,  0,  1,  2, 
	               -2, -1,  0,  1,  2, 
	               -2, -1,  0,  1,  2, 
	               -2, -1,  0,  1,  2, 
	               -2, -1,  0,  1,  2 };
	int dx1[] =  {  0, -1, -2,  1,  2,  1,  2, -1, -2,
	                1,  0, -1,  2,  3,  2,  3,  0, -1 };
	int dx2[] =  { -1,  0,  1,  2, -1,  0,  1,  2, -2,
	               -2, -2, -2,  3,  3,  3,  3 };

	int dy0[] = {  -2, -2, -2, -2, -2,
	               -1, -1, -1, -1, -1,
	                0,  0,  0,  0,  0,
	                1,  1,  1,  1,  1,
	                2,  2,  2,  2,  2 };
	int dy1[] = {   0, -1, -2,  1,  2,
	               -1, -2,  1,  2,  0,
	               -1, -2,  1,  2, -1,
	               -2,  1,  2 };
	int dy2[] = {  -2, -2, -2, -2,  3,
	                3,  3,  3, -1,  0,
	                1,  2, -1,  0,  1, 2 };

    int* dx3 = dx0;
    int* dy3 = dy0;
	/*
	  % How to create the "scale" array in Matlab:
	  x=[-2:2];
	  y=[-2:2];
	  xx=repmat(x, [5,1]);
	  yy=repmat(y', [1,5]);
	  E=exp(-(xx.^2 + yy.^2)/(2 * 0.5));
	  E./sum(sum(E))
	*/
	float scale5x5gaussian[] = {
		0.00010678874539,  0.00214490928858,   0.00583046794284,  0.00214490928858,  0.00010678874539,
		0.00214490928858,  0.04308165471265,  0.11710807914534,  0.04308165471265,  0.00214490928858,
		0.00583046794284,  0.11710807914534,  0.31833276350651,  0.11710807914534,  0.00583046794284,
		0.00214490928858,  0.04308165471265,  0.11710807914534,  0.04308165471265,  0.00214490928858,
		0.00010678874539,  0.00214490928858,  0.00583046794284,  0.00214490928858,  0.00010678874539 };

	float scale5x5dot[] = {
        0, 1, 1, 1, 0,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        0, 1, 1, 1, 0,
    };

	// clearly constructing these arrays should be done at runtime
	int* dxs[] = {dx0, dx1, dx2, dx3};
	int* dys[] = {dy0, dy1, dy2, dy3};
	float* scales[] = {scale5x5gaussian, NULL, NULL, scale5x5dot};
	float *scale;
	int *dx,*dy;
	int ndrop,mindx,maxdx,mindy,maxdy;
	int x,y,w,h,i;

	if (render_symbol < 0 || render_symbol >= 4) {
		fprintf(stderr, "tilerender: add_star: invalid render symbol %d\n", render_symbol);
		return 0;
	}
	dx    =    dxs[render_symbol];
	dy    =    dys[render_symbol];
	scale = scales[render_symbol];
	ndrop = ndrops[render_symbol];

	mindx = -2;
	maxdx = 2;
	mindy = -2;
	maxdy = 2;

	w = args->W;
	h = args->H;

	x = xmerc2pixel(xp, args);
	if (x+maxdx < 0 || x+mindx >= w) {
		return 0;
	}
	y = ymerc2pixel(yp, args);
	if (y+maxdy < 0 || y+mindy >= h) {
		return 0;
	}

	for (i=0; i<ndrop; i++) {
      int ix, iy;
	   float thisscale;
		ix = x + dx[i];
		if ((ix < 0) || (ix >= w))
			continue;
		iy = y + dy[i];
		if ((iy < 0) || (iy >= h))
			continue;
		thisscale = 1.0;
		if (scale)
			thisscale = scale[i];
		fluximg[3*(iy*w + ix) + 0] += rflux * thisscale;
		fluximg[3*(iy*w + ix) + 1] += bflux * thisscale;
		fluximg[3*(iy*w + ix) + 2] += nflux * thisscale;
	}
	return 1;
}

void add_star_merc(double xp, double yp, merc_flux* flux, mercargs* margs) {
	add_star(xp, yp, flux->rflux, flux->bflux, flux->nflux, margs->fluximg, margs->symbol, margs->args);
}
