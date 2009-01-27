/*
   This file is part of the Astrometry.net suite.
   Copyright 2007 Dustin Lang and Keir Mierle.

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

char* image_dir = "/home/gmaps/apod-solves";
/*
  char* image_dirs[] = {
  "/home/gmaps/apod-solves",
  "/data2/test/userimages",
  }
*/

char* user_image_dirs[] = {
	"/home/gmaps/ontheweb-data/",
	"/home/gmaps/test/web-data/",
	"/home/gmaps/gmaps-rdls/",
};

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

const char* cachedomain = "apod";

int render_images(unsigned char* img, render_args_t* args) {
    int I;
    sl* imagefiles;
	sl* wcsfiles = NULL;
    float* counts;
    float* ink;
    int i, j, w;
    double *ravals, *decvals;
	bool fullfilename = TRUE;

	logmsg("starting.\n");

	if (strcmp("images", args->currentlayer) == 0) {
		if (!args->filelist) {
			logmsg("Layer is \"images\" but no filelist was given.\n");
			return -1;
		}
		fullfilename = FALSE;
        imagefiles = file_get_lines(args->filelist, FALSE);
        if (!imagefiles) {
            logmsg("failed to read filelist \"%s\".\n", args->filelist);
            return -1;
        }
        logmsg("read %i filenames from the file \"%s\".\n", sl_size(imagefiles), args->filelist);
	} else if (strcmp("userimage", args->currentlayer) == 0) {
		int j;
		if (!(sl_size(args->imagefns) && sl_size(args->imwcsfns))) {
			logmsg("both imagefn and imwcsfn are required.\n");
			return -1;
		}
		imagefiles = sl_new(4);
		wcsfiles = sl_new(4);
		for (j=0; j<MIN(sl_size(args->imagefns), sl_size(args->imwcsfns)); j++) {
			char* imgfn = sl_get(args->imagefns, j);
			char* wcsfn = sl_get(args->imwcsfns, j);
			for (i=0; i<sizeof(user_image_dirs)/sizeof(char*); i++) {
				char* fn = sl_appendf(imagefiles, "%s/%s", user_image_dirs[i], imgfn);
				if (!file_readable(fn)) {
					sl_pop(imagefiles);
					free(fn);
					continue;
				}
				logmsg("Found user image %s.\n", fn);
				break;
			}
			for (i=0; i<sizeof(user_image_dirs)/sizeof(char*); i++) {
				char* fn = sl_appendf(wcsfiles, "%s/%s", user_image_dirs[i], wcsfn);
				if (!file_readable(fn)) {
					sl_pop(wcsfiles);
					free(fn);
					continue;
				}
				logmsg("Found user WCS %s.\n", fn);
				break;
			}
			if (sl_size(imagefiles) != sl_size(wcsfiles)) {
				logmsg("Failed to find user image or WCS file.\n");
				return -1;
			}
		}
	} else {
		logmsg("Current layer is \"%s\", neither \"images\" nor \"userimage\".\n",
			   args->currentlayer);
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
    for (I=0; I<sl_size(imagefiles); I++) {
		char* basefn;
		char* basepath;
        char* imgfn;
        char* wcsfn;
        bool jpeg, png;
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

		imgfn = wcsfn = basepath = NULL;

        basefn = sl_get(imagefiles, I);
		if (!strlen(basefn)) {
            logmsg("empty filename.\n");
            continue;
		}
		logmsg("Base filename: \"%s\"\n", basefn);

		if (args->filelist) {
			// absolute path?
			if (basefn[0] == '/') {
				basepath = strdup(basefn);
			} else {
				asprintf_safe(&basepath, "%s/%s", image_dir, basefn);
			}
			basefn = basepath;
			logmsg("Base path: \"%s\"\n", basefn);
		}
		if (fullfilename) {
			// HACK - strip off the filename suffix... only to reappend it below...
			char* dot = strrchr(basefn, '.');
			if (!dot) {
				logmsg("no filename suffix: %s\n", basefn);
				continue;
			}
			*dot = '\0';
		}

		if (wcsfiles) {
			wcsfn = sl_get(wcsfiles, I);
		} else {
			asprintf_safe(&wcsfn, "%s.wcs", basefn);
			if (!file_readable(wcsfn)) {
				logmsg("filename %s: WCS file %s not readable.\n", basefn, wcsfn);
				goto nextimage;
			}
		}
		logmsg("WCS: \"%s\"\n", wcsfn);

		{
			char* suffixes[] = { "jpeg", "jpg", "png" };
			bool isjpegs[] = { TRUE,  TRUE,  FALSE };
			bool ispngs[]  = { FALSE, FALSE, TRUE  };
			bool gotit = FALSE;
			for (i=0; i<sizeof(suffixes)/sizeof(char*); i++) {
				asprintf_safe(&imgfn, "%s.%s", basefn, suffixes[i]);
				if (file_readable(imgfn)) {
					jpeg = isjpegs[i];
					png = ispngs[i];
					logmsg("Image: \"%s\"\n", imgfn);
					gotit = TRUE;
					break;
				}
				free(imgfn);
			}
			if (!gotit) {
				logmsg("Found no image file for basename \"%s\".\n", basefn);
				imgfn = NULL;
				goto nextimage;
			}
		}

        res = sip_read_header_file(wcsfn, &wcs);
        free(wcsfn);
		wcsfn = NULL;
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
        ylo = floor(dec2pixelf(decmax, args));
        yhi = ceil (dec2pixelf(decmin, args));
        if ((yhi < 0) || (ylo >= args->H)) {
            // No need to read the image!
			logmsg("No overlap between this image and the requested RA,Dec region (Y pixel range %i to %i).\n", ylo, yhi);
			goto nextimage;
		}

		// min ra -> max merc -> max pixel
		xmerclo = radeg2merc(ramax);
		xmerchi = radeg2merc(ramin);
		xlo = floor(xmerc2pixelf(xmerclo, args));
		xhi =  ceil(xmerc2pixelf(xmerchi, args));
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
            md5_update(&md5, imgfn, strlen(imgfn));
            md5_update(&md5, &(args->ramin), sizeof(double));
            md5_update(&md5, &(args->ramax), sizeof(double));
            md5_update(&md5, &(args->decmin), sizeof(double));
            md5_update(&md5, &(args->decmax), sizeof(double));
            md5_update(&md5, &(args->W), sizeof(int));
            md5_update(&md5, &(args->H), sizeof(int));
            md5_finish_hex(&md5, cachekey);
        }
        cached = cache_load(args, cachedomain, cachekey, &len);
        if (cached && (len != (args->W * args->H * 4 * sizeof(float)))) {
            logmsg("Cached object (%s/%s) was wrong size.\n", cachedomain, cachekey);
            free(cached);
            cached = NULL;
        }
        if (cached) {
            float* thisink = cached;
            float* thiscounts = cached + args->W * args->H * 3;
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
						thiscounts[j*w+i] = weight;
					}
				}
				thisink = NULL;
			} else {
				/*
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
				//double mx, my;
				double xyzA[3];
				double xyzB[3];
				double raB, decB;
				double arcsec;
				double inpix;
				double zoom;
				int ir, id;
				int izoom;
				int zoomscale;
				int zoomW, zoomH;

				// find the RA,Dec at the middle of the range of output pixels...
				ir = (xlo + xhi + 1) / 2;
				id = (ylo + yhi + 1) / 2;
				ra  =  ravals[ir];
				dec = decvals[id];
				radecdeg2xyzarr(ra, dec, xyzA);
				raB  = ravals [ir + ((ir == W-1) ? -1 : +1)];
				decB = decvals[id + ((id == H-1) ? -1 : +1)];
				radecdeg2xyzarr(raB, decB, xyzB);
				// divide distsq by half because I added one pixel in x and y.
				arcsec = distsq2arcsec(distsq(xyzA, xyzB, 3) / 2.0);
				inpix = arcsec / sip_pixel_scale(&wcs);
				zoom = log(inpix) / log(2.0);
				izoom = floor(zoom);

				logmsg("One output pixel covers %g arcsec, %g input pixels (zoom %i)\n",
					   arcsec, inpix, izoom);

				userimg = NULL;
				zoomscale = 1;
				zoomW = W;

				if (izoom > 0) {
					char* pyrfn;
					zoomscale = (1 << izoom);
					// Look for image pyramid.
					while (izoom > 0) {
						asprintf_safe(&pyrfn, "%s-%i.jpg", basefn, izoom);
						if (file_readable(pyrfn)) {
							userimg = cairoutils_read_jpeg(pyrfn, &zoomW, &zoomH);
							if (userimg)
								break;
							else
								logmsg("failed to read image file %s\n", pyrfn);
						} else
							logmsg("no such file %s\n", pyrfn);
						free(pyrfn);
						izoom--;
						zoomscale /= 2;
					}
					if (userimg) {
						logmsg("Found pyramid image: %s, size %i x %i (full size %i x %i), zoom scale %i.\n",
							   pyrfn, zoomW, zoomH, W, H, zoomscale);
						free(pyrfn);
					}
				}

				if (!userimg) {
					logmsg("Opening image \"%s\".\n", imgfn);
					if (jpeg)
						userimg = cairoutils_read_jpeg(imgfn, &W, &H);
					else if (png)
						userimg = cairoutils_read_png(imgfn, &W, &H);
					if (!userimg) {
						logmsg("failed to read image file %s\n", imgfn);
						goto nextimage;
					}
				}
				//            logmsg("Image %s is %i x %i.\n", imgfn, W, H);
				
				//logmsg("Clamped to pixel range: (%i to %i, %i to %i)\n", xlo, xhi, ylo, yhi);
				
				// iterate over mercator space (ie, output pixels)
				for (j=ylo; j<=yhi; j++) {
					dec = decvals[j];
					for (i=xlo; i<=xhi; i++) {
						int pppx,pppy;
						ra = ravals[i];
						if (!sip_radec2pixelxy_check(&wcs, ra, dec, &imagex, &imagey))
							continue;
						pppx = lround(imagex-1); // The -1 is because FITS uses 1-indexing for pixels. DOH
						pppy = lround(imagey-1);
						if (pppx < 0 || pppx >= W || pppy < 0 || pppy >= H)
							continue;
						// nearest neighbour. bilinear is for weenies.
						pppx /= zoomscale;
						pppy /= zoomscale;
						thisink[3*(j*w + i) + 0] = userimg[4*(pppy*zoomW + pppx) + 0] * weight;
						thisink[3*(j*w + i) + 1] = userimg[4*(pppy*zoomW + pppx) + 1] * weight;
						thisink[3*(j*w + i) + 2] = userimg[4*(pppy*zoomW + pppx) + 2] * weight;
						thiscounts[j*w + i] = weight;
					}
				}
				free(userimg);
			}

            add_ink(ink, counts, thisink, thiscounts, args->W, args->H);
			if (thisink && thiscounts) {
				logmsg("Caching: %s/%s (%d bytes).\n", cachedomain, cachekey, sz);
				cache_save(args, cachedomain, cachekey, chunk, sz);
			}
            free(chunk);
        }

	nextimage:
		free(wcsfn);
		free(imgfn);
		free(basepath);

    }

    sl_free2(imagefiles);
	if (wcsfiles)
		sl_free_nonrecursive(wcsfiles);
    free(ravals);
    free(decvals);

    if (args->density) {
		double mincounts = 1e100;
		double maxcounts = 0;
        double maxden = -1e100;
        for (j=0; j<args->H; j++) {
            for (i=0; i<w; i++) {
                uchar* pix;
                double den;
                pix = pixel(i, j, img, args);

                mincounts = MIN(counts[j*w + i], mincounts);
                maxcounts = MAX(counts[j*w + i], maxcounts);

                //den = log(counts[j*w + i]);
                //den = counts[j*w + i];
                den = sqrt(counts[j*w + i]);
                den *= pow(4.0, args->zoomlevel);
                den *= exp(args->gain * log(4.0));
				//den = log(den);
                if (den > maxden)
                    maxden = den;
                if (den > 0.0) {
                    //pix[0] = pix[1] = pix[2] = MAX(0, MIN(255, den));
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
