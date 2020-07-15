/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */


#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "healpix.h"
#include "mathutil.h"
#include "pnpoly.h"

/* Size of the line buffer */
#define BUFSIZE 1000

/* Need to extend this structure to handle more complicated polygons */
typedef struct {
    double *corners;
} rect_field;

/* Dead simple linked list */
struct ll_node {
    int data;
    struct ll_node *next;
};

/* TODO: factor this out into mathutil or something similar. Eventually
 * this file won't need it anymore anyway if the healpix code is rewritten
 * to abandon the current behaviour of using hierarchical numbering for 
 * powers of 4. */
static Inline anbool ispowerof4(unsigned int x) {
    if (x >= 0x40000)
        return (					x == 0x40000   ||
                                                        x == 0x100000   || x == 0x400000  ||
                                                        x == 0x1000000  || x == 0x4000000 ||
                                                        x == 0x10000000 || x == 0x40000000);
    else
        return (x == 0x1		|| x == 0x4	   ||
                x == 0x10	   || x == 0x40	  ||
                x == 0x100	  || x == 0x400	 ||
                x == 0x1000	 || x == 0x4000	||
                x == 0x10000);
}

/* Compute the min of n doubles */
static double nmin(double *p, int n)
{
    int i;
    double cmin;
    assert (n > 0);
    cmin = p[0];
    for (i = 1; i < n; i++)
	{   
            if (p[i] < cmin)
		cmin = p[i];
	}
    return cmin;
}

/* Compute the min of n doubles and return the index rather than value */
static int nminind(double *p, int n)
{
    int i;
    int cminind;
	
    assert (n > 0);
    cminind = 0;
    for (i = 1; i < n; i++)
	{   
            if (p[i] < p[cminind])
		{   
                    cminind = i;
		}
	}
    return cminind;
}

/* Compute the max of n doubles */
static double nmax(double *p, int n)
{
    int i;
    double cmax;

    assert (n > 0);
    cmax = p[0];
    for (i = 1; i < n; i++)
	{   
            if (p[i] > cmax)
                cmax = p[i];
	}
    return cmax;															  
}

/* Compute the range from max to min of n points stored in a double vector. */
static double range(double *p, int n)
{
    double max = nmax (p, n);
    double min = nmin (p, n);
    return fabs (max - min);
}


/* Wraps around point_in_poly to choose the axis along which the corners 
 * have minimum variance in order to do the chop-off projection.
 *
 * TODO: generalize this to n-point convex polygons. Notes have been made
 * as to where changes need to occur and what.
 */
int is_inside_field(rect_field *f, double *p)
{
    // DIMQUADS
    /* FIXME assumes 4 points, will need malloc'ing */
    double coordtrans[12];
	
    double *coord1;
    double *coord2;

    /* FIXME assumes 4 points */
    double *x = coordtrans;
    double *y = coordtrans + 4;
    double *z = coordtrans + 8;
    double t_coord1, t_coord2;
    double ranges[3];
    int i;
    int xgot = 0, ygot = 0, zgot = 0;
    /* FIXME Assumes 4 points */
    for (i = 0; i < 12; i++)
	{
            int j = i % 3;
            switch (j)
		{
                case 0:
                    coordtrans[xgot++] = f->corners[i];
                    break;
                case 1:
                    /* FIXME 4+ should be n+ */
                    coordtrans[4 + ygot++] = f->corners[i];
                    break;
                case 2:
                    /* FIXME 4+ should be n+ */
                    coordtrans[8 + zgot++] = f->corners[i];
                    break;
		}
	}
    /* FIXME all these calls should use n */
    ranges[0] = range(x, 4);
    ranges[1] = range(y, 4);
    ranges[2] = range(z, 4);
	
    switch (nminind(ranges, 3))
	{
        case 0:
            coord1 = y;
            coord2 = z;
            t_coord1 = p[1];
            t_coord2 = p[2];
            break;
        case 1:
            coord1 = x;
            coord2 = z;
            t_coord1 = p[0];
            t_coord2 = p[2];
            break;
        case 2:
            coord1 = x;
            coord2 = y;
            t_coord1 = p[0];
            t_coord2 = p[1];
            break;
        default:
            return -1;
	}
    /* FIXME should call with n rather than 4 */
    return point_in_poly(coord1, coord2, 4, t_coord1, t_coord2);

}
/* An ugly recursive implementation written while still trying to get 
 * the algorithm right, left here for... some reason */
#if 0
void fill_maps_recursive(char *minmap, char *maxmap, uint hpx, uint Nside,
                         rect_field *curfield, char *visited)
{
    double thishpx_coords[3];
    uint neighbours[8];
    uint nn;
    if (visited[hpx / 8] & (1 << (hpx % 8)))
        return;
    visited[hpx / 8] |= 1 << (hpx % 8);
    healpix_to_xyzarr_lex(0.5, 0.5, hpx, Nside, thishpx_coords);
    //printf("Examining healpix %d, centered at (%f, %f, %f)\n", hpx, 
    //		thishpx_coords[0], thishpx_coords[1], thishpx_coords[2]);
    if (is_inside_field(curfield, thishpx_coords))
	{
            int j;
            maxmap[hpx / 8] |= (1 << (hpx % 8));
            nn = healpix_get_neighbours_nside(hpx, neighbours, Nside);
            for (j = 0; j < nn; j++)
		{
				
                    double ncoords[3];
                    healpix_to_xyzarr_lex(0.5, 0.5, neighbours[j], Nside, ncoords);

                    //printf("- Examining neighbour healpix %d, centered at (%f, %f, %f)\n", neighbours[j],
                    //	ncoords[0], ncoords[1], ncoords[2]);
			
			
                    if (!is_inside_field(curfield, ncoords)) {
                        //printf("-- Not in field, breaking off neighbour search\n");
                        break;
                    }
		}
            if (j == nn)
                minmap[(hpx / 8)] |= (1 << (hpx % 8));
            for (j = 0; j < nn; j++)
		{
                    //printf("Recursing on neighbour of %d, %d\n", hpx, neighbours[j]);
                    fill_maps_recursive(minmap, maxmap, neighbours[j], Nside,
					curfield, visited);
		}
	}
}
void fill_maps(char *minmap, char *maxmap, uint hpx, uint Nside,
               rect_field *curfield)
{
    char *visited = malloc(2 * Nside * Nside * sizeof(char));
    uint i;
    uint visitedcnt = 0;
    for (i = 0; i < 2 * Nside * Nside; i++)
        visited[i] = 0;

    fill_maps_recursive(minmap, maxmap, hpx, Nside, curfield, visited);
}
#endif


/* This basically takes two appropriately sized char[] arrays to be used
 * as bitmaps for the min/max healpix mapping of the sky, as well as a 
 * field we'd like to process and add to the bitmaps, and a healpix number
 * to start adding from, and an Nside factor.
 *
 * This basically starts at the healpix closest to the center of a field 
 * (actually it starts at hpx, see todo below) and percolates outward by
 * checking neighbouring healpixes. A healpix is included in the upper bound
 * map if it's center is inside the field boundaries, and included in the
 * lower bound map only if all of its neighbours' centers are inside the
 * field boundaries as well.
 * 
 * TODO: bring the center-computing code from main() up here
 * TODO: factor that bitmap access ugliness out into some damned macros
 */
void fill_maps(char *minmap, char *maxmap, uint hpx, uint Nside,
               rect_field *curfield)

{
    /* Gotta love "are we done" switch variables */
    anbool done = FALSE;
	
    /* Bitmap we'll use to keep from revisiting the same healpixes */ 
    char *visited = malloc(2 * Nside * Nside * sizeof(char));
	
    /* Store the head of the queue (actually a LIFO) of healpixes to examine */
    struct ll_node *queue = NULL;
    double thishpx_coords[3];

    int i;

    /* Initialize the visited map */	
    for (i = 0; i < 2 * Nside * Nside; i++)
        visited[i] = 0;

    do {
        /* nn = "number of neighbours */
        uint nn;
        anbool found_neighbour_outside = FALSE;
		
        /* always need room for at least 8 neighbours; actual number
         * gets stored in nn */
        uint neighbours[8];
        if (visited[hpx / 8] & (1 << (hpx % 8)))
            {
                /* should never happen */
                assert(1 == 0);
            }
        /* set that we've visited hpx */
        visited[hpx / 8] |= (1 << (hpx % 8));
		
        /* compute the xyz location of the center of hpx */
        healpix_to_xyzarr(hpx, Nside, 0.5, 0.5, thishpx_coords);
		
        /* skip the body of this loop if we can */
        if (!is_inside_field(curfield, thishpx_coords))
            goto getnext;
		
        /* always include in maxmap */
        maxmap[hpx / 8] |= 1 << (hpx % 8);

        nn = healpix_get_neighbours(hpx, neighbours, Nside);

        /* check inclusion for each neighbour and enqueue it if unvisited */
        for (i = 0; i < nn; i++)
            {
                double ncoords[3];
                healpix_to_xyzarr(neighbours[i], Nside, 0.5, 0.5, ncoords);
			
                if (!is_inside_field(curfield, ncoords))
                    found_neighbour_outside = TRUE;
			
                if (!(visited[neighbours[i] / 8] & (1 << (neighbours[i] % 8))))
                    {
                        struct ll_node *newnode = malloc(sizeof(struct ll_node));
                        newnode->next = queue;
                        newnode->data = neighbours[i];
                        queue = newnode;
                    }
            }

        /* If no neighbours lie outside then include in the minmap */
        if (!found_neighbour_outside)
            minmap[hpx / 8] |= 1 << (hpx % 8);
		
        /* dequeue the next one to be checked and set hpx */
    getnext:
        if (queue != NULL)
            {
                struct ll_node *newhead = queue->next;
                hpx = queue->data;
                free(queue);
                queue = newhead;
            }
        else {
            /* If the queue is empty and we've gone through the last iteration
             * then it's okay to terminate. note that right after dequeueing
             * you can have queue == NULL but still not be done, which is 
             * why we can't just use "queue != NULL" as our loop condition */
            done = TRUE;
        }
    } while (!done);
	
    free(visited);
}

static void print_help(FILE *f, char *name)
{
    fprintf(f, "This program computes statistics about a set of (for the moment) rectangular\nfields on the sky.\n\n");
    fprintf(f, "\tUsage: %s -N <Nside> [-I <inputfile>]\n\n", name);
    fprintf(f, "In the absence of a -I switch, reads coordinates from standard input.\n");
    fprintf(f, "Input should be 4 XYZ coordinates per line, with components and coordinates\nseparated by tabs (i.e. 12 tab-delimited doubles)\n");
}

int main(int argc, char **argv)
{
    double max;
    rect_field curfield;
    int filled_min = 0, filled_max = 0;
    char *hpmap_min, *hpmap_max;
    char *buf = malloc(BUFSIZE * sizeof(char));
    int ich, i;
    int Nside = -1;
    uint fields;
    char *infilename = NULL;
    FILE *input;
    if (argc == 0)
	{
            print_help(stderr, argv[0]);
            exit(1);
	}
    while ((ich = getopt(argc, argv, "N:I:")) != EOF)
	{
            switch (ich)
		{
                case 'N':
                    Nside = atoi(optarg);
                    break;
                case 'I':
                    if (optarg == NULL)
                        {
                            print_help(stderr, argv[0]);
                            fprintf(stderr, "Error: -I requires argument");
                            exit(1);
                        }
                    infilename = strdup(optarg);
                    break;
		}
	}
    if (Nside < 1)
	{
            print_help(stderr, argv[0]);
            fprintf(stderr, "\nError: specify a positive Nside value with -N\n");
            exit(1);
	}
    else if (ispowerof4(Nside))
	{
            print_help(stderr, argv[0]);
            fprintf(stderr, "Error: Nside values that are powers of 4 \
				are bad news at\n the moment, choose a different one\n");
            exit(1);
	}
	
    if (infilename) {
        input = fopen(infilename, "r");
        if (input == NULL)
            {
                perror(argv[0]);
                exit(1);
            }
    }
    else {
        input = stdin;
    }
    /* We could get away with allocating ceil(2/3 * Nside * Nside) */
    hpmap_min = malloc(2 * Nside * Nside * sizeof(char));
    hpmap_max = malloc(2 * Nside * Nside * sizeof(char));
	
    if (hpmap_min == NULL || hpmap_max == NULL)
	{
            fprintf(stderr, "malloc failed!\n");
            exit(1);
	}	
    for (i = 0; i < 2 * Nside * Nside; i++) {
        hpmap_min[i] = 0;
        hpmap_max[i] = 0;
    }
	
    curfield.corners = malloc(3 * 4 * sizeof(double));
    fields = 0;
    while (fgets(buf, BUFSIZE, input) != NULL)
	{
            uint centerhp;
            int i, j;
            double center[3];
            center[0] = center[1] = center[2] = 0;
            fields++;
            //printf("Doing field %d\n",fields);
            curfield.corners[0] = atof(strtok(buf, "\t"));
		
            /* 12 = 3 coords x 4 pts, got 1 */
            for (j = 1; j < 12; j++)
		{
                    char *tok = strtok(NULL, "\t");
                    if (tok == NULL)
			{
                            fprintf(stderr, "Premature end of line!\n");
                            exit(1);
			}
                    curfield.corners[j] = atof(tok);
		}
		
            for (i = 0; i < 4; i++)
		{
                    for (j = 0; j < 3; j++)
			{
                            center[j] += curfield.corners[3*i + j];
			}
		}
            for (i = 0; i < 3; i++)
                center[i] /= 4;
            normalize_3(center);
		
            centerhp = xyzarrtohealpix(center, (uint)Nside);
            fill_maps(hpmap_min, hpmap_max, centerhp, (uint)Nside, &curfield);
	}
    for (i = 0; i < 12 * Nside * Nside; i++)
	{
            if (hpmap_min[i / 8] & (1 << (i % 8)))
                filled_min++;
            if (hpmap_max[i / 8] & (1 << (i % 8)))
                filled_max++;
	}
    max = 12 * Nside * Nside;

    printf("Min: %f, Max: %f\n",
           ((double)filled_min) / max, 
           ((double)filled_max) / max);

    fclose(input);
    return 0;
}
