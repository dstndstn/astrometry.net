/*
   This file is part of the Astrometry.net suite.
   Copyright 2007-2009 Dustin Lang and Keir Mierle.

   The Astrometry.net suite is free software; you can redistribute
   it and/or modify it under the terms of the GNU General Public License
   as published by the Free Software Foundation, version 2.

   The Astrometry.net suite is distributed in the hope that it will be
   useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with the Astrometry.net suite ; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
*/
#include <stdio.h>
#include <math.h>
#include <stdarg.h>
#include <sys/param.h>

#include "ioutils.h"
#include "sip-utils.h"
#include "tilerender.h"
#include "render_images.h"
#include "sip_qfits.h"
#include "cairoutils.h"
#include "keywords.h"
#include "md5.h"

const char* image_dirs[] = {
	"/home/gmaps/ontheweb-data/",
	"/home/gmaps/test/web-data/",
	"/home/gmaps/gmaps-rdls/",
	"./",
};

const char* cachedomain = "images";

static void
ATTRIB_FORMAT(printf,1,2)
logmsg(char* format, ...) {
    va_list args;
    va_start(args, format);
    fprintf(stderr, "render_images: ");
    vfprintf(stderr, format, args);
    va_end(args);
}

static void heatmap(float inpix, unsigned char* outpix) {
	inpix = MAX(0.0, MIN(255.0, inpix));
	if (inpix <= 96.0) {
		outpix[0] = inpix * 255.0 / 96.0;
		outpix[1] = 0;
		outpix[2] = 0;
	} else if (inpix <= 192.0) {
		outpix[0] = 255;
		outpix[1] = (inpix - 96.0) * 255.0 / 96.0;
		outpix[2] = 0;
	} else {
		outpix[0] = 255;
		outpix[1] = 255;
		outpix[2] = (inpix - 192.0) * 255.0 / 63.0;
	}
}

static void add_ink(float* ink, float* counts, float* thisink, float* thiscounts,
                    int W, int H) {
    int i;
	if (thisink) {
		for (i=0; i<(3*W*H); i++)
			ink[i] += thisink[i];
	}
    for (i=0; i<(W*H); i++)
        counts[i] += thiscounts[i];
}

int render_images(unsigned char* img, render_args_t* args) {
    int I;
    sl* imagefiles;
	sl* imagetypes;
	sl* wcsfiles;
    float* counts;
    float* ink;
    int i, j, w;
    double *ravals, *decvals;
	const char* imgtypes[] = {"jpegfn ", "pngfn "};
	double nilval = -1e100;

	logmsg("starting.\n");

    imagefiles = sl_new(256);
    imagetypes = sl_new(256);
    wcsfiles = sl_new(256);
    get_string_args_of_types(args, imgtypes, 2,
							 imagefiles, imagetypes);
    get_string_args_of_type(args, "wcsfn ", wcsfiles);

	nilval = get_double_arg_of_type(args, "nilval ", nilval);

	// When plotting density, we only need the WCS files.
    if (!args->density && (sl_size(imagefiles) != sl_size(wcsfiles))) {
        logmsg("Got %i jpeg files but %i wcs files.\n",
               sl_size(imagefiles), sl_size(wcsfiles));
        return -1;
    }
        
    w = args->W;

    counts = calloc(args->W * args->H, sizeof(float));
    ink = calloc(3 * args->W * args->H, sizeof(float));
    ravals  = malloc(args->W * sizeof(double));
    decvals = malloc(args->H * sizeof(double));
    for (i=0; i<w; i++)
        ravals[i] = pixel2ra(i, args);
    for (j=0; j<args->H; j++)
        decvals[j] = pixel2dec(j, args);
    for (I=0; I<sl_size(wcsfiles); I++) {
		char* fn;
        char* imgfn = NULL;
        char* imgtype = NULL;
        char* wcsfn = NULL;
        sip_t wcs;
        unsigned char* userimg = NULL;
        int W, H;
        sip_t* res;
        double ra, dec;
        double imagex, imagey;
        double ramin, ramax, decmin, decmax;
		double xmerclo, xmerchi;
        int xlo, xhi, xwraplo, xwraphi, ylo, yhi;
        float pixeldensity, weight;
        char cachekey[33];
        float* cached;
        int len;
		int expectlen;

		imgfn = wcsfn = fn = NULL;
        wcsfn = sl_get(wcsfiles, I);
		if (I < sl_size(imagefiles)) {
			fn = sl_get(imagefiles, I);
			imgtype = sl_get(imagetypes, I);
			imgfn = find_file_in_dirs(image_dirs, sizeof(image_dirs)/sizeof(char*),
									  fn, TRUE);
		}
		if (!args->density && !imgfn) {
            logmsg("Couldn't find image file \"%s\"\n", fn);
            continue;
        }
        // FIXME - wcs files must be absolute paths.

        res = sip_read_header_file(wcsfn, &wcs);
        if (!res) {
            logmsg("failed to parse SIP header from %s\n", wcsfn);
			goto nextimage;
        }
        W = wcs.wcstan.imagew;
        H = wcs.wcstan.imageh;

        // find the bounds in RA,Dec of this image.
        // magic 10 = step size in pixels for walking the image boundary.
        sip_get_radec_bounds(&wcs, 10, &ramin, &ramax, &decmin, &decmax);

		logmsg("RA,Dec range for this image: (%g to %g, %g to %g)\n",
			   ramin, ramax, decmin, decmax);

        // increasing DEC -> decreasing Y pixel coord
		/*
		 ylo = floor(dec2pixelf(MAX(decmax, decmin), args));
		 yhi = ceil (dec2pixelf(MIN(decmax, decmin), args));
		 */
		ylo = floor(dec2pixelf(decmax, args));
		yhi = ceil (dec2pixelf(decmin, args));
        if ((yhi < 0) || (ylo >= args->H)) {
            // No need to read the image!
			logmsg("No overlap between this image and the requested RA,Dec region (Y pixel range %i to %i).\n", ylo, yhi);
			goto nextimage;
		}

		// min ra -> max merc -> max pixel
		/* FIXME - It would be nice if we could handle requests for flipped regions!!
		 xmerclo = radeg2merc(MAX(ramax, ramin));
		 xmerchi = radeg2merc(MIN(ramin, ramax));
		 xlo = floor(xmerc2pixelf(MAX(xmerclo, xmerchi), args));
		 xhi =  ceil(xmerc2pixelf(MIN(xmerchi, xmerclo), args));
		 */
		xmerclo = radeg2merc(ramax);
		xmerchi = radeg2merc(ramin);
		xlo = floor(xmerc2pixelf(xmerclo, args));
		xhi =  ceil(xmerc2pixelf(xmerchi, args));
		logmsg("x range: %i, %i\n", xlo, xhi);
		if ((xhi < 0) || (xlo >= args->W)) {
			if (xmerclo < 0.5) {
				xwraplo = floor(xmerc2pixelf(xmerclo + 1.0, args));
				xwraphi =  ceil(xmerc2pixelf(xmerchi + 1.0, args));
			} else {
				xwraplo = floor(xmerc2pixelf(xmerclo - 1.0, args));
				xwraphi =  ceil(xmerc2pixelf(xmerchi - 1.0, args));
			}
			if ((xwraphi < 0) || (xwraplo >= args->W)) {
				logmsg("No overlap between this image and the requested RA,Dec region (X pixel range %i to %i or %i to %i).\n", xlo, xhi, xwraplo, xwraphi);
				goto nextimage;
			}
			xlo = xwraplo;
			xhi = xwraphi;
		}
		logmsg("Pixel range: (%i to %i, %i to %i)\n", xlo, xhi, ylo, yhi);

        // Check the cache...
        {
            md5_context md5;
            md5_starts(&md5);
			if (imgfn)
				md5_update(&md5, imgfn, strlen(imgfn));
			if (wcsfn)
				md5_update(&md5, wcsfn, strlen(wcsfn));
            md5_update(&md5, &(args->ramin), sizeof(double));
            md5_update(&md5, &(args->ramax), sizeof(double));
            md5_update(&md5, &(args->decmin), sizeof(double));
            md5_update(&md5, &(args->decmax), sizeof(double));
            md5_update(&md5, &(args->W), sizeof(int));
            md5_update(&md5, &(args->H), sizeof(int));
            md5_finish_hex(&md5, cachekey);
        }
		if (args->density) {
			cached = cache_load(args, "density", cachekey, &len);
			expectlen = args->W * args->H * 1 * sizeof(float);
		} else {
			cached = cache_load(args, cachedomain, cachekey, &len);
			expectlen = args->W * args->H * 4 * sizeof(float);
		}
        if (cached && (len != expectlen)) {
            logmsg("Cached object (%s/%s) was wrong size.\n", cachedomain, cachekey);
            free(cached);
            cached = NULL;
        }
        if (cached) {
            float* thisink;
            float* thiscounts;
			if (args->density) {
				thisink = NULL;
				thiscounts = cached;
			} else {
				thisink = cached;
				thiscounts = cached + args->W * args->H * 3;
			}
            logmsg("Cache hit: %s/%s.\n", cachedomain, cachekey);
            add_ink(ink, counts, thisink, thiscounts, args->W, args->H);
            free(cached);
        } else {
            int sz;
            float* chunk;
            float* thisink;
            float* thiscounts;
            sz = args->W * args->H * 4 * sizeof(float);
            // FIXME - realloc
            chunk = calloc(sz, 1);
            thisink = chunk;
            thiscounts = chunk + args->W * args->H * 3;

            // clamp to image bounds
            xlo = MAX(0, xlo);
            ylo = MAX(0, ylo);
            xhi = MIN(args->W-1, xhi);
            yhi = MIN(args->H-1, yhi);

            pixeldensity = 1.0 / square(sip_pixel_scale(&wcs));
            weight = pixeldensity;

			if (args->density) {
				// iterate over mercator space (ie, output pixels)
				for (j=ylo; j<=yhi; j++) {
					dec = decvals[j];
					for (i=xlo; i<=xhi; i++) {
						ra = ravals[i];
						if (!sip_radec2pixelxy_check(&wcs, ra, dec, &imagex, &imagey))
							continue;
                        if (imagex < 0 || imagex >= W ||
                            imagey < 0 || imagey >= H)
                            continue;
						thiscounts[j*w+i] = weight;
					}
				}
				thisink = NULL;
			} else {
				/*
                 // there used to be image pyramid handling here...

				  In order to use an image pyramid, we need to have some idea of how many
				  pixels in the original image will cover each pixel in the output image.
				  In principle this could be quite different over the range of the image,
				  but let's just ignore that for now...

				  We need to convert:
				  -one pixel in the output image
				  -to a change in Mercator coordinates
				  -to a change in RA,Dec (or distance on the unit sphere)
				  -through the CD matrix to pixels in the input image.
				*/

                logmsg("Opening image \"%s\".\n", imgfn);
				if (starts_with(imgtype, "jpeg")) {
					userimg = cairoutils_read_jpeg(imgfn, &W, &H);
				} else if (starts_with(imgtype, "png")) {
					userimg = cairoutils_read_png(imgfn, &W, &H);
				}
                if (!userimg) {
                    logmsg("failed to read image file %s\n", imgfn);
                    goto nextimage;
                }
				// logmsg("Image %s is %i x %i.\n", imgfn, W, H);
				
				//logmsg("Clamped to pixel range: (%i to %i, %i to %i)\n", xlo, xhi, ylo, yhi);
				
				// iterate over mercator space (ie, output pixels)
				for (j=ylo; j<=yhi; j++) {
					dec = decvals[j];
					for (i=xlo; i<=xhi; i++) {
						int pppx,pppy;
						float thisw;
						ra = ravals[i];
						if (!sip_radec2pixelxy_check(&wcs, ra, dec, &imagex, &imagey))
							continue;
						pppx = lround(imagex-1); // The -1 is because FITS uses 1-indexing for pixels. DOH
						pppy = lround(imagey-1);
						if (pppx < 0 || pppx >= W || pppy < 0 || pppy >= H)
							continue;
						// FIXME -- just look at red channel...
						if (userimg[4*(pppy*W + pppx) + 0] == nilval)
							continue;
						// combine "weight" with this image's alpha channel.
						thisw = weight * (float)userimg[4*(pppy*W + pppx) + 3] / 255.0;
						// nearest neighbour.
						thisink[3*(j*w + i) + 0] = userimg[4*(pppy*W + pppx) + 0] * thisw;
						thisink[3*(j*w + i) + 1] = userimg[4*(pppy*W + pppx) + 1] * thisw;
						thisink[3*(j*w + i) + 2] = userimg[4*(pppy*W + pppx) + 2] * thisw;
						thiscounts[j*w + i] = thisw;
					}
				}
				free(userimg);
			}

            add_ink(ink, counts, thisink, thiscounts, args->W, args->H);
			if (thisink && thiscounts) {
				logmsg("Caching: %s/%s (%d bytes).\n", cachedomain, cachekey, sz);
				cache_save(args, cachedomain, cachekey, chunk, sz);
			} else if (args->density && thiscounts) {
				cache_save(args, "density", cachekey, thiscounts, args->W * args->H * sizeof(float));
			}
            free(chunk);
        }

	nextimage:
		free(imgfn);
    }

    sl_free2(imagetypes);
    sl_free2(imagefiles);
    sl_free2(wcsfiles);
    free(ravals);
    free(decvals);

	// We produce RGBA images here, which get converted to cairo ordering in tilerender.c

    if (args->density) {
		double mincounts = 1e100;
		double maxcounts = 0;
        double maxden = -1e100;
		double scale = (pow(4.0, args->zoomlevel) *
						pow(4.0, args->gain));
        for (j=0; j<args->H; j++) {
            for (i=0; i<w; i++) {
                uchar* pix;
                double den;
                pix = pixel(i, j, img, args);

                mincounts = MIN(counts[j*w + i], mincounts);
                maxcounts = MAX(counts[j*w + i], maxcounts);

                den = counts[j*w + i];
				if (args->nlscale != 0.0)
					den /= args->nlscale;
				den *= scale;
				if (args->arc)
					den = asinh(den);
				else if (args->sqrt)
					den = sqrt(den);
				if (args->nlscale != 0.0)
					den *= args->nlscale;

                if (den > maxden)
                    maxden = den;
                if (den > 0.0) {
					heatmap(den, pix);
                    pix[3] = 255;
                }
            }
        }
        logmsg("range of counts: [%g, %g]\n", mincounts, maxcounts);
        logmsg("max density value: %g\n", maxden);
    } else {
        for (j=0; j<args->H; j++) {
            for (i=0; i<w; i++) {
                uchar* pix;
                pix = pixel(i, j, img, args);
                if (counts[j*w + i]) {
                    pix[0] = MAX(0, MIN(255, ink[3 * (j*w + i) + 0] / counts[j*w + i]));
                    pix[1] = MAX(0, MIN(255, ink[3 * (j*w + i) + 1] / counts[j*w + i]));
                    pix[2] = MAX(0, MIN(255, ink[3 * (j*w + i) + 2] / counts[j*w + i]));
                    pix[3] = 255;
                }
            }
        }
    }

    free(counts);
    free(ink);
	return 0;
}
