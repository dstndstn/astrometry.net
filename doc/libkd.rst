
libkd documentation
===================

C API
-----

.. highlight:: c

.. c:function:: kdtree_t* kdtree_build(kdtree_t* kd, void *data, int N, int D, int Nleaf, int treetype, unsigned int options);

    Build a tree from an array of data, of size N*D*sizeof(data_item).
    
    *kd*: NULL to allocate a new *kdtree_t* structure, or the address
       of the structure in which to store the result.
    
    *data*: your N x D-dimensional data, stored in N-major direction:
       data[n*D + d] is the address of data item "n", dimension "d".
       If 3-dimensional data, eg, order is x0,y0,z0,x1,y1,z1,x2,y2,z2.
    
    *N*: number of vectors
    
    *D*: dimensionality of vectors
    
    *Nleaf*: number of element in a kd-tree leaf node.  Typical value
       would be about 32.
    
    *treetype*:
      * if your data are doubles, *KDTT_DOUBLE*
      * if your data are floats,    *KDTT_FLOAT*

    For fancier options, see *kd_tree_types*.
    
    *options*: bitfield of *kd_build_options* values.  Specify one of:
      * *KD_BUILD_BBOX*: keep a full bounding-box at each node;
      * *KD_BUILD_SPLIT*: just keep the split dimension and value at each node.
    
    see *kd_build_options* for additional fancy stuff.

    NOTE that this function will *permute* the contents of the *data* array!
    
    When you're done with your tree, be sure to *kdtree_free()* it.
    
    Example:

    .. code-block:: c

       double mydata[] = { 1,1, 2,2, 3,3, 4,4, 5,5, 6,6, 7,7, 8,8 };
       int D = 2;
       int N = sizeof(mydata) / (D * sizeof(double));
       kdtree_t* kd = kdtree_build(NULL, mydata, N, D, 4, KDTT_DOUBLE, KD_BUILD_BBOX);
       kdtree_print(kd);
       kdtree_free(kd);



.. c:function:: void kdtree_free(kdtree_t *kd);

    Frees the given *kd*.  By default, the *kd->data* is NOT freed.
    Set *kd->free_data = 1* to free the data when *kdtree_free()* is called.


Python API
----------




Code Internals
--------------

