// See util/ctmf.c for copyright & license
#ifndef CTMF_H
#define CTMF_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief Constant-time median filtering
 *
 * This function does a median filtering of an 8-bit image. The source image is
 * processed as if it was padded with zeros. The median kernel is square with
 * odd dimensions. Images of arbitrary size may be processed.
 *
 * To process multi-channel images, you must call this function multiple times,
 * changing the source and destination adresses and steps such that each channel
 * is processed as an independent single-channel image.
 *
 * Processing images of arbitrary bit depth is not supported.
 *
 * The computing time is O(1) per pixel, independent of the radius of the
 * filter. The algorithm's initialization is O(r*width), but it is negligible.
 * Memory usage is simple: it will be as big as the cache size, or smaller if
 * the image is small. For efficiency, the histograms' bins are 16-bit wide.
 * This may become too small and lead to overflow as \a r increases.
 *
 * \param src           Source image data.
 * \param dst           Destination image data. Must be preallocated.
 * \param width         Image width, in pixels.
 * \param height        Image height, in pixels.
 * \param src_step      Distance between adjacent pixels on the same column in
 *                      the source image, in bytes.
 * \param dst_step      Distance between adjacent pixels on the same column in
 *                      the destination image, in bytes.
 * \param r             Median filter radius. The kernel will be a 2*r+1 by
 *                      2*r+1 square.
 * \param cn            Number of channels. For example, a grayscale image would
 *                      have cn=1 while an RGB image would have cn=3.
 * \param memsize       Maximum amount of memory to use, in bytes. Set this to
 *                      the size of the L2 cache, then vary it slightly and
 *                      measure the processing time to find the optimal value.
 *                      For example, a 512 kB L2 cache would have
 *                      memsize=512*1024 initially.
 */
void ctmf(
        const unsigned char* src, unsigned char* dst,
        int width, int height,
        int src_step_row, int dst_step_row,
        int r, int channels, unsigned long memsize
        );

#ifdef __cplusplus
}
#endif

#endif
