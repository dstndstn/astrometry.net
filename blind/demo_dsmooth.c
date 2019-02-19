/*
 A test suite for dsmooth, compares dsmooth with dsmooth2

 Jon Barron, 2007
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <regex.h>
#include <cairo.h>
#include <math.h>

#include "cairoutils.h"
#include "dimage.h"

#define PEAKDIST 4

int is_png(const struct dirent *de) {
    regex_t preq;
    regcomp(&preq, "[^ ]*[.](png|PNG)$", REG_EXTENDED);
    return !regexec(&preq, de->d_name, (size_t) 0, NULL, 0);
}

int is_jpeg(const struct dirent *de) {
    regex_t preq;
    regcomp(&preq, "[^ ]*[.](jpg|JPG|jpeg|JPEG)$", REG_EXTENDED);
    return !regexec(&preq, de->d_name, (size_t) 0, NULL, 0);
}

int is_image(const struct dirent *de) {
    return is_jpeg(de) || is_png(de);
}

int is_output(const struct dirent *de) {
    regex_t preq;
    regcomp(&preq, "out_[^ ]*", REG_EXTENDED);
    return !regexec(&preq, de->d_name, (size_t) 0, NULL, 0);
}

int is_input_image(const struct dirent *de) {
    return is_image(de) && !is_output(de);
}

float* to_bw_f(unsigned char *image, int imW, int imH) {
    int w, h, c;
    float *image_bw;
    float v;

    image_bw = malloc(sizeof(float) * imW * imH);
    for (w = 0; w < imW; w++) {
        for (h = 0; h < imH; h++) {
            v = 0.0;
            for (c = 0; c <= 2; c++) {
                v = v + ((float)(image[4*(w*imH+h) + c])) / 3.0;
            }
            image_bw[w*imH + h] = v;
        }
    }

    return image_bw;
}

unsigned char* to_bw_u8(unsigned char *image, int imW, int imH) {
    int i;
    unsigned char *image_bw, *p;
    image_bw = p = malloc(sizeof(unsigned char) * imW * imH);
    for (i = 0; i < imW*imH; i++, image += 4, p++) {
        int total = image[0] + image[1] + image[2];
        *p = total / 3;
    }
    return image_bw;
}

unsigned char* to_cairo_bw(float *image_bw, int imW, int imH) {
    int w, h, c;
    unsigned char* image_cairo;

    image_cairo = malloc(sizeof(unsigned char) * imW * imH * 4);
    for (w = 0; w < imW; w++) {
        for (h = 0; h < imH; h++) {
            for (c = 0; c <= 2; c++) {
                image_cairo[4*(w*imH + h) + c] = (unsigned char)(image_bw[w*imH + h]);
            }
            image_cairo[4*(w*imH + h) + 3] = 255;
        }
    }

    return image_cairo;
}

int main(void) {
    struct dirent **namelist;
    int i, n, N;
    unsigned char *image = NULL;
    float *image_bw_f;
    unsigned char *image_bw_u8;
    float *image_out_old;
    float *image_out_new;
    char fullpath[255];
    char outpath_old[255];
    char outpath_new[255];
    int imW, imH;

    float sigma;
    float err;

    sigma = 1.0;

    N = scandir("demo_simplexy_images", &namelist, is_input_image, alphasort);
    if (N < 0) {
        perror("scandir");
        return 1;
    }

    for (n = 0; n < N; n++) {

        strcpy(fullpath, "demo_simplexy_images/");
        strcat(fullpath, namelist[n]->d_name);

        strcpy(outpath_old, "demo_simplexy_images/out_");
        strcat(outpath_old, namelist[n]->d_name);
        outpath_old[strlen(outpath_old)-4] = '\0';
        strcat(outpath_old, "_old");
        strcat(outpath_old, ".png");

        strcpy(outpath_new, "demo_simplexy_images/out_");
        strcat(outpath_new, namelist[n]->d_name);
        outpath_new[strlen(outpath_new)-4] = '\0';
        strcat(outpath_new, "_new");
        strcat(outpath_new, ".png");

        fprintf(stderr,"demo_dsmooth: loading %s ", fullpath);

        if (is_png(namelist[n])) {
            fprintf(stderr, "as a PNG\n");
            image = cairoutils_read_png(fullpath, &imW, &imH);
        }

        if (is_jpeg(namelist[n])) {
            fprintf(stderr, "as a JPEG\n");
            image = cairoutils_read_jpeg(fullpath, &imW, &imH);
        }

        image_bw_u8 = to_bw_u8(image, imW, imH);
        image_bw_f = malloc(sizeof(float) * imW * imH);
        for (i = 0; i < imW*imH; i++) {
            image_bw_f[i] = (float)image_bw_u8[i];
        }

        image_out_old = malloc(sizeof(float)*imW*imH);
        image_out_new = malloc(sizeof(float)*imW*imH);

        fprintf(stderr,"demo_dsmooth: running %s through dsmooth\n", fullpath);		
        dsmooth(image_bw_f, imW, imH, sigma, image_out_old);

        fprintf(stderr,"demo_dsmooth: running %s through dsmooth2\n", fullpath);
        dsmooth2(image_bw_f, imW, imH, sigma, image_out_new);
		
        err = 0.0;
        for (i = 0; i < imW*imH; i++) {
            err += fabs(image_out_old[i]-image_out_new[i]);
        }
        err = err / (imW*imH);
		
        fprintf(stderr, "demo_dsmooth: error between smooths: %f per pixel\n", err);

        //		fprintf(stderr, "demo_dsmooth: writing old dsmoothed image to %s\n", outpath_old);
        //		cairoutils_write_png(outpath_old, to_cairo_bw(image_out_old, imW, imH), imW, imH);

        //		fprintf(stderr, "demo_dsmooth: writing new dsmoothed image to %s\n", outpath_new);
        //		cairoutils_write_png(outpath_new, to_cairo_bw(image_out_new, imW, imH), imW, imH);

        free(namelist[n]);
        free(image);
        free(image_bw_f);
        free(image_bw_u8);
        free(image_out_old);
        free(image_out_new);

    }
    free(namelist);

    return 0;
}


