/*
  This file is part of libkd.
  Copyright 2006-2008 Dustin Lang and Keir Mierle.

  libkd is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, version 2.

  libkd is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with libkd; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*/

#ifndef KDTREE_H
#define KDTREE_H

#include <stdio.h>
#include <stdint.h>

#include "an-bool.h"

#define KDTREE_MAX_LEVELS 1000

#define KDT_INFTY_DOUBLE HUGE_VAL
#define KDT_INFTY_FLOAT  HUGE_VALF

enum kd_rangesearch_options {
	KD_OPTIONS_COMPUTE_DISTS    = 0x1,
	KD_OPTIONS_RETURN_POINTS    = 0x2,
	KD_OPTIONS_SORT_DISTS       = 0x4,
	KD_OPTIONS_SMALL_RADIUS     = 0x8,
	/* If both bounding box and splitting plane are available,
	   use splitting plane (default is bounding box)... */
	KD_OPTIONS_USE_SPLIT        = 0x10,
	/* If the tree is u32 and bounding boxes are being used and the
	   rangesearch only fits in 64 bits, use doubles instead of u64 in
	   the distance computation (default is to use u64).
	   (Likewise for u16/u32)
	*/
	KD_OPTIONS_NO_BIG_INT_MATH      = 0x20,
	/*
	  In bounding-box trees that also have a "splitdim" array,
	  do a quick check along the splitting dimension.
	*/
	KD_OPTIONS_SPLIT_PRECHECK   = 0x40,
	/*
	  In integer bounding-box trees, do an L1 distance pre-check.
	*/
	KD_OPTIONS_L1_PRECHECK      = 0x80,
	/*
	  Don't resize the kdtree_qres_t* result structure to take only the
	  space required (assume it's going to be reused and we're letting the
	  memory usage do the "high water mark" thing).
	*/
	KD_OPTIONS_NO_RESIZE_RESULTS = 0x100
};

enum kd_build_options {
	KD_BUILD_BBOX           = 0x1,
	KD_BUILD_SPLIT          = 0x2,
	/* Only applicable to integer trees: use a separate array to hold the
	   splitting dimension, rather than packing it into the bottom bits
	   of the splitting plane location. */
	KD_BUILD_SPLITDIM  = 0x4,
    KD_BUILD_NO_LR     = 0x8,
    /* Twiddle the split locations so that computing LR is O(1).
     Only works for double trees or int trees with KD_BUILD_SPLITDIM. */
    KD_BUILD_LINEAR_LR     = 0x10,
    // DEBUG
    KD_BUILD_FORCE_SORT    = 0x20,
    
};

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t  u8;

// Legacy kdtree node.
struct kdtree_node {
    unsigned int l,r;              /* data(l:r) are coordinates below this node */
	/* Implicit hyperrectangle 
	real HR[sizeof(real)*D] Lower
	real HR[sizeof(real)*D] Upper
	*/
};
typedef struct kdtree_node kdtree_node_t;

enum kd_types {
	KDT_NULL = 0,   // note, this MUST be 0 because it's used as a boolean value.
	KDT_DATA_NULL   = 0,
	KDT_DATA_DOUBLE = 0x1,
	KDT_DATA_FLOAT  = 0x2,
	KDT_DATA_U32    = 0x4,
	KDT_DATA_U16    = 0x8,
	KDT_TREE_NULL   = 0,
	KDT_TREE_DOUBLE = 0x100,
	KDT_TREE_FLOAT  = 0x200,
	KDT_TREE_U32    = 0x400,
	KDT_TREE_U16    = 0x800,
	KDT_EXT_NULL    = 0,
	KDT_EXT_DOUBLE  = 0x10000,
	KDT_EXT_FLOAT   = 0x20000
};
typedef enum kd_types kd_types;

#define KDT_DATA_MASK  0xf
#define KDT_TREE_MASK  0xf00
#define KDT_EXT_MASK   0x30000

/*
  Possible values for the "treetype" member.

  There are three relevant data type: (a) the type of the raw data; (b) the type used in the tree's
  bounding boxes and splitting planes; (c) the external type.

  These are called the "data", "tree", and "ext" types.
*/
enum kd_tree_types {
	KDTT_NULL = 0,

	/* "All doubles, all the time". */
	KDTT_DOUBLE = KDT_EXT_DOUBLE | KDT_DATA_DOUBLE | KDT_TREE_DOUBLE,

	/* "All floats, all the time". */
	KDTT_FLOAT  = KDT_EXT_FLOAT | KDT_DATA_FLOAT  | KDT_TREE_FLOAT,

	/* Data are "doubles", tree is u32.  aka inttree. */
	KDTT_DOUBLE_U32 = KDT_EXT_DOUBLE | KDT_DATA_DOUBLE | KDT_TREE_U32,

	/* Data are "doubles", tree is u16.  aka shorttree. */
	KDTT_DOUBLE_U16 = KDT_EXT_DOUBLE | KDT_DATA_DOUBLE | KDT_TREE_U16,

	/* Data and tree are "u32". */
	KDTT_DUU = KDT_EXT_DOUBLE | KDT_DATA_U32 | KDT_TREE_U32,

	/* Data and tree are "u16". */
	KDTT_DSS = KDT_EXT_DOUBLE | KDT_DATA_U16 | KDT_TREE_U16,

	KDTT_DDU = KDT_EXT_DOUBLE | KDT_DATA_DOUBLE | KDT_TREE_U32,
};

struct kdtree;
typedef struct kdtree kdtree_t;

struct kdtree_qres;
typedef struct kdtree_qres kdtree_qres_t;

struct kdtree_funcs {
	void* (*get_data)(const kdtree_t* kd, int i);
	void  (*copy_data_double)(const kdtree_t* kd, int start, int N, double* dest);
    double (*get_splitval)(const kdtree_t* kd, int nodeid);
    anbool (*get_bboxes)(const kdtree_t* kd, int node, void* bblo, void* bbhi);

    int (*check)(const kdtree_t* kd);
    void (*fix_bounding_boxes)(kdtree_t* kd);

    void  (*nearest_neighbour_internal)(const kdtree_t* kd, const void* query, double* bestd2, int* pbest);
	kdtree_qres_t* (*rangesearch)(const kdtree_t* kd, kdtree_qres_t* res, const void* pt, double maxd2, int options);

    void (*nodes_contained)(const kdtree_t* kd,
                            const void* querylow, const void* queryhi,
                            void (*callback_contained)(const kdtree_t* kd, int node, void* extra),
                            void (*callback_overlap)(const kdtree_t* kd, int node, void* extra),
                            void* cb_extra);

    //anbool (*node_node_mindist2_exceeds)(const kdtree_t* kd1, int node1, const kdtree_t* kd2, int node2, double maxd2);

    // instrumentation functions - set these to get callbacks about
    // the progress of the algorithm.

    // a node was enqueued to be searched during nearest-neighbour.
    void (*nn_enqueue)(const kdtree_t* kd, int nodeid, int place);
    // a node was pruned during nearest-neighbour.
    void (*nn_prune)(const kdtree_t* kd, int nodeid, double d2, double bestd2, int place);
    // a node is being explored during nearest-neighbour.
    void (*nn_explore)(const kdtree_t* kd, int nodeid, double d2, double bestd2);
    // a point is being examined during nearest-neighbour.
    void (*nn_point)(const kdtree_t* kd, int nodeid, int pointindex);
    // a new best point has been found.
    void (*nn_new_best)(const kdtree_t* kd, int nodeid, int pointindex, double d2);
};
typedef struct kdtree_funcs kdtree_funcs;


struct kdtree {
	/*
	  A bitfield describing the type of this tree.
	*/
	u32 treetype;

	// (compatibility mode)
	kdtree_node_t *nodes;   /* Flat tree storing nodes and
                             bounding boxes. */

	int32_t* lr;            /* Points owned by leaf nodes, stored and manipulated
							 in a way that's too complicated to explain in this comment.
							 (nbottom) */
               
	u32* perm;           /* Permutation index / hairstyle from the 80s
                          (ndata) */

	/* Bounding box: list of D-dimensional lower hyperrectangle corner followed by
     D-dimensional upper corner.
     (ttype x ndim x nnodes) */
	union {
		float* f;
		double* d;
		u32* u;
		u16* s;
		void* any;
	} bb;
    // how many bounding-boxes were found?
    int n_bb;

	/* Split position (& dimension for ints) (ttype x ninterior). */
	union {
		float* f;
		double* d;
		u32* u;
		u16* s;
		void* any;
	} split;

	/* Split dimension for floating-point types (x ninterior) */
	u8* splitdim;

	/* bitmasks for the split dimension and location. */
	u8 dimbits;
	u32 dimmask;
	u32 splitmask;

	/* Raw coordinate data as xyzxyzxyz (dtype) */
	union {
		float* f;
		double* d;
		u32* u;
		u16* s;
		void* any;
	} data;

	anbool converted_data;

	double* minval;
	double* maxval;
	double scale;    /* kdtype per real -- isotropic */
	double invscale; /* real per kdtype */

	int ndata;     /* Number of items */
	int ndim;      /* Number of dimensions */
	int nnodes;    /* Number of nodes */
	int nbottom;   /* Number of leaf nodes */
	int ninterior; /* Number of internal nodes */
	int nlevels;

    anbool has_linear_lr;

    // For i/o: the name of this tree in the file.
    char* name;

    void* io;

	struct kdtree_funcs fun;
};

struct kdtree_qres {
	unsigned int nres;
	unsigned int capacity; /* Allocated size. */
	union {
		double* d;
		float* f;
		u32* u;
		u16* s;
		void* any;
	} results;
	double *sdists;          /* Squared distance from query point */
	u32 *inds;    /* Indexes into original data set */
};

// Returns the number of data points in this kdtree.
int kdtree_n(const kdtree_t* kd);

// Returns the number of nodes in this kdtree.
int kdtree_nnodes(const kdtree_t* kd);

anbool kdtree_has_old_bb(const kdtree_t* kd);

double kdtree_get_conservative_query_radius(const kdtree_t* kd, double radius);

/* These functions return the number of bytes each entry in the kdtree is
 expected to have.  These return positive values EVEN IF THE ACTUAL array
 is NULL in the particular tree.  Ie, it answers the question "how big
 would the LR array for this tree be, if it had one?" */
size_t kdtree_sizeof_lr(const kdtree_t* kd);
size_t kdtree_sizeof_perm(const kdtree_t* kd);
size_t kdtree_sizeof_bb(const kdtree_t* kd);
size_t kdtree_sizeof_split(const kdtree_t* kd);
size_t kdtree_sizeof_splitdim(const kdtree_t* kd);
size_t kdtree_sizeof_data(const kdtree_t* kd);
size_t kdtree_sizeof_nodes(const kdtree_t* kd);

static inline int kdtree_exttype(const kdtree_t* kd) {
	return kd->treetype & KDT_EXT_MASK;
}

static inline int kdtree_datatype(const kdtree_t* kd) {
	return kd->treetype & KDT_DATA_MASK;
}

/*
  What type are my bounding boxes / split planes?
*/
static inline int kdtree_treetype(const kdtree_t* kd) {
	return kd->treetype & KDT_TREE_MASK;
}

void kdtree_memory_report(kdtree_t* kd);

kdtree_t* kdtree_new(int N, int D, int Nleaf);

void kdtree_print(kdtree_t* kd);

/*
  Reinitialize the table of function pointers "kd->fun".
*/
void kdtree_update_funcs(kdtree_t* kd);

void kdtree_set_limits(kdtree_t* kd, double* low, double* high);

kdtree_t* kdtree_convert_data(kdtree_t* kd, void *data,
							  int N, int D, int Nleaf, int treetype);

void* kdtree_get_data(const kdtree_t* kd, int i);

void kdtree_copy_data_double(const kdtree_t* kd, int i, int N, double* dest);

const char* kdtree_kdtype_to_string(int kdtype);

const char* kdtree_build_options_to_string(int opts);

int kdtree_kdtype_parse_data_string(const char* str);
int kdtree_kdtype_parse_tree_string(const char* str);
int kdtree_kdtype_parse_ext_string(const char* str);

int kdtree_kdtypes_to_treetype(int exttype, int treetype, int datatype);

int kdtree_permute(const kdtree_t* tree, int ind);

/*
  Compute the inverse permutation of tree->perm and place it in "invperm".
 */
void kdtree_inverse_permutation(const kdtree_t* tree, int* invperm);

/* Free results */
void kdtree_free_query(kdtree_qres_t *kd);

/* Free a tree; does not free kd->data */
void kdtree_free(kdtree_t *kd);

int kdtree_is_node_empty(const kdtree_t* kd, int nodeid);

int kdtree_is_leaf_node_empty(const kdtree_t* kd, int nodeid);

/* The leftmost point owned by this node. */
int kdtree_left(const kdtree_t* kd, int nodeid);

/* The rightmost point owned by this node. */
int kdtree_right(const kdtree_t* kd, int nodeid);

/* Shortcut kdtree_{left,right} for leaf nodes. */
int kdtree_leaf_right(const kdtree_t* kd, int nodeid);
int kdtree_leaf_left(const kdtree_t* kd, int nodeid);


/* How many points are owned by node "nodeid"? */
int kdtree_npoints(const kdtree_t* kd, int nodeid);

/*
  Returns the node index of the first/last leaf within the subtree
  rooted at "nodeid".
*/
int kdtree_first_leaf(const kdtree_t* kd, int nodeid);
int kdtree_last_leaf(const kdtree_t* kd, int nodeid);

/*
 Return the first and last node id (resp) of a given level in the
 tree.  The root is level 0.
 */
int kdtree_level_start(const kdtree_t* kd, int level);
int kdtree_level_end(const kdtree_t* kd, int level);

/*
 Returns the level of the given nodeid; 0=root.
 */
int kdtree_get_level(const kdtree_t* kd, int nodeid);

/*
 How many levels are in a tree with "Nnodes" nodes?

 A tree with one node (Nnodes = 1) has one level.
 */
int kdtree_nnodes_to_nlevels(int Nnodes);

/* Nearest neighbour: returns the index _in the kdtree_ of the nearest point;
 * the point is at  (kd->data + ind * kd->ndim)  and its permuted index is
 * (kd->perm[ind]).
 *
 * If "bestd2" is non-NULL, the distance-squared to the nearest neighbour
 * will be placed there.
 */
int kdtree_nearest_neighbour(const kdtree_t* kd, const void *pt, double* bestd2);

/* Nearest neighbour (if within a maximum range): returns the index
 * _in the kdtree_ of the nearest point, _if_ its distance is less than
 * maxd2.  (Otherwise, -1).
 *
 * If "bestd2" is non-NULL, the distance-squared to the nearest neighbour
 * will be placed there.
 */
int kdtree_nearest_neighbour_within(const kdtree_t* kd, const void *pt,
                                    double maxd2, double* bestd2);

/*
 * Finds the set of non-leaf nodes that are completely contained
 * within the given query rectangle, plus the leaf nodes that
 * overlap with the query.  (In other words, all nodes that overlap
 * the query, without recursing down to leaf nodes unnecessarily.)
 * Calls one of two callbacks for fully-contained and
 * partly-contained nodes.
 */
void kdtree_nodes_contained(const kdtree_t* kd,
							const void* querylow, const void* queryhi,
							void (*callback_contained)(const kdtree_t* kd, int node, void* extra),
							void (*callback_overlap)(const kdtree_t* kd, int node, void* extra),
							void* cb_extra);

#define KD_IS_LEAF(kd, i)       ((i) >= ((kd)->ninterior))
#define KD_IS_LEFT_CHILD(i)    ((i) & 1)
#define KD_PARENT(i)     (((i)-1)/2)
#define KD_CHILD_LEFT(i)  (2*(i)+1)
#define KD_CHILD_RIGHT(i)  (2*(i)+2)

/*
 * Copies the bounding box of the given node into the given arrays,
 * which are of the external type.
 *
 * Returns FALSE if the tree does not have bounding boxes.
 */
anbool kdtree_get_bboxes(const kdtree_t* kd, int node, void* bblo, void* bbhi);

double kdtree_get_splitval(const kdtree_t* kd, int nodeid);

int kdtree_get_splitdim(const kdtree_t* kd, int nodeid);

double kdtree_node_node_mindist2(const kdtree_t* kd1, int node1,
                                 const kdtree_t* kd2, int node2);

double kdtree_node_node_maxdist2(const kdtree_t* kd1, int node1,
                                 const kdtree_t* kd2, int node2);

anbool kdtree_node_node_mindist2_exceeds(const kdtree_t* kd1, int node1,
									   const kdtree_t* kd2, int node2,
									   double dist2);

anbool kdtree_node_node_maxdist2_exceeds(const kdtree_t* kd1, int node1,
									   const kdtree_t* kd2, int node2,
									   double dist2);

double kdtree_node_point_mindist2(const kdtree_t* kd, int node, const void* pt);

double kdtree_node_point_maxdist2(const kdtree_t* kd, int node, const void* pt);

anbool kdtree_node_point_mindist2_exceeds(const kdtree_t* kd, int node,
                                        const void* pt, double dist2);

anbool kdtree_node_point_maxdist2_exceeds(const kdtree_t* kd, int node,
                                        const void* pt, double dist2);


/* Sanity-check a tree. 0=okay. */
int kdtree_check(const kdtree_t* t);

void kdtree_fix_bounding_boxes(kdtree_t* kd);

#if 0
/* Range seach using callback */
void kdtree_rangesearch_callback(kdtree_t *kd, real *pt, real maxdistsquared,
								 void (*rangesearch_callback)(kdtree_t* kd, real* pt, real maxdist2, real* computed_dist2, int indx, void* extra),
								 void* extra);

/* Counts points within range. */
int kdtree_rangecount(kdtree_t* kd, real* pt, real maxdistsquared);

/* Output Graphviz .dot format version of the tree */
void kdtree_output_dot(FILE* fid, kdtree_t* kd);
#endif



// include dimension-generic versions of the dimension-specific code.
#define KD_DIM_GENERIC 1

#endif /* KDTREE_H */


#if defined(KD_DIM) || defined(KD_DIM_GENERIC)

#if defined(KD_DIM)
  #undef KDID
  #undef GLUE2
  #undef GLUE

  #define GLUE2(a, b) a ## b
  #define GLUE(a, b) GLUE2(a, b)
  #define KDID(x) GLUE(x ## _, KD_DIM)
#else
  #define KDID(x) x
#endif
#define KDFUNC(x) KDID(x)

/* Build a tree from an array of data, of size N*D*sizeof(real).
   "options" is a bitfield of kd_build_options values. */
kdtree_t* KDFUNC(kdtree_build)
	 (kdtree_t* kd, void *data, int N, int D, int Nleaf,
	  int treetype, unsigned int options);

/* Range seach */
kdtree_qres_t* KDFUNC(kdtree_rangesearch)(const kdtree_t *kd, const void *pt, double maxd2);

kdtree_qres_t* KDFUNC(kdtree_rangesearch_nosort)(const kdtree_t *kd, const void *pt, double maxd2);

kdtree_qres_t* KDFUNC(kdtree_rangesearch_options)(const kdtree_t *kd, const void *pt, double maxd2, int options);

kdtree_qres_t* KDFUNC(kdtree_rangesearch_options_reuse)(const kdtree_t *kd, kdtree_qres_t* res, const void *pt, double maxd2, int options);

#if !defined(KD_DIM)
#undef KD_DIM_GENERIC
#endif

#endif

