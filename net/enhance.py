import numpy as np

def pixel_ranks(img, get_argsort=False):
    '''
    Computes the rankings of the given pixels.

    Returns an array of integers, 0 to len(img)-1, ranking the pixels.

    If get_argsort is True, returns the tuple of (ranking, I), where
    *I* is the argsort of img.
    '''
    II = np.argsort(img)
    rankimg = np.empty_like(II)
    rankimg[II] = np.arange(len(II))
    if get_argsort:
        return rankimg,II
    return rankimg

class EnhanceImage(object):
    '''
    A simple implementation of the 'Enhance!' algorithm
    (Lang, Hogg, & Schoelkopf, AI-STATS 2014)
    '''

    def __init__(self, npix, nbands, random=True, smallweight=1e-3):
        '''
        Create a new EnhanceImage object, with *npix* pixels and
        *nbands* bands (eg, 3 for RGB images).

        If *random* is True, initializes randomly.

        Initializes the weights to *smallweight*.
        '''
        # We store flat arrays
        self.enhW = np.empty(npix, np.float32)
        self.enhW[:] = smallweight
        self.enhI = np.zeros((npix, nbands), np.float32)
        if random:
            for b in range(nbands):
                self.enhI[:,b] = np.random.permutation(npix) / float(npix)

    def update(self, mask, img, weightFactor=1., addRandom=True):
        '''
        Updates this EnhanceImage with the given new image *img*.

        *mask* describes the pixels in this EnhanceImage corresponding
         to the pixels in *img*; this must be a numpy index array or
         boolean array to select pixels than overlap; ie, the mask
         describes _valid_, _overlapping_ pixels.

         *img* must be "flattened", having shape (npix, nbands).

         *img* must be floating-point with values between 0 and 1.

         If *addRandom* is true, a small amount of uniform noise is
         added to break ties between pixels.
        '''
        wenh = self.enhW[mask]
        npix,nbands = self.enhI.shape

        assert(len(img.shape) == 2)
        # Number of pixels in image == number of pixels within masked
        # region of this EnhanceImage.
        assert(img.shape[0] == len(wenh))
        assert(img.shape[1] == nbands)
        if addRandom:
            # Image range is as expected.
            assert(np.min(img) >= 0.)
            assert(np.max(img) <= 1.)

        for b in range(nbands):
            # Pull out this band of the image (R,G,B)
            imgb = img[:,b]
            if addRandom:
                imgb = imgb + np.random.uniform(0., 1./256.,
                                                size=imgb.shape)
            # Rank the image pixels
            imgrank = pixel_ranks(imgb)
            # Pull out the masked region
            enh = self.enhI[mask, b]
            # Rank the 'enhanced' pixels
            enhrank,EI = pixel_ranks(enh, get_argsort=True)
            # Compute composite ("consensus") rank.
            rank = ( ((enhrank * wenh) + (imgrank * weightFactor))
                        / (wenh + weightFactor) )
            # The "consensus" "ranks" need not be integers... re-rank
            # them.
            rank = pixel_ranks(rank)
            # Permute the "enhance" pixels using this new ranking.
            enhnew = enh[EI[rank]]
            self.enhI[mask,b] = enhnew
        # Update the weights
        self.enhW[mask] += 1.

    def stretch_to_match(self, img):
        '''
        Stretches this enhanced image to match the tone of the given
        *img*.  If *shape* is given, *shape = (H,W)*, reshapes the
        result to have shape *(H,W,#bands)*.  The result will have the
        same datatype as *img*.
        '''

        npix,nbands = self.enhI.shape
        assert(len(img.shape) == 3)
        stretch = np.zeros((npix, nbands), img.dtype)
        for b in range(nbands):
            imgb = img[:,:,b].ravel()
            I = np.argsort(imgb)
            # enhI should already be histogram-equalized (ie, ~
            # floating-point pixel ranks).  Scale to indices in the
            # (argsorted) imgb lookup table.
            EI = np.floor(self.enhI[:,b] * len(I)).astype(int)
            assert(np.all(EI >= 0))
            assert(np.all(EI < len(I)))
            stretch[:,b] = imgb[I[EI]]
        return stretch

                


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import pylab as plt
    import sys

    import os
    from astrometry.util.util import *
    from astrometry.util.resample import *
    from astrometry.util.plotutils import *
    import fitsio

    ra,dec = 83.8, -1.1
    pixscale = 100./3600.
    W,H = 500,500
    targetwcs = Tan(ra, dec, W/2.+0.5, H/2.+0.5,
                    -pixscale, 0., 0., pixscale, float(W), float(H))
    enhance = EnhanceImage(W*H, 3)

    ps = PlotSequence('en')
    
    urls = ['http://bbc.astrometry.net/new_fits_file/2',
            'http://bbc.astrometry.net/new_fits_file/4',
            'http://bbc.astrometry.net/new_fits_file/3',
            ]
    images = []
    for i,url in enumerate(urls):
        fn = 'orion-%i.fits' % (i+1)
        if not os.path.exists(fn):
            cmd = 'wget -O %s %s' % (fn, url)
            print(cmd)
            os.system(cmd)
        images.append(fn)
    
    for imgfn,wcsfn in zip(images, images):

        wcs = Sip(wcsfn)
        img = fitsio.read(imgfn)
        print('img shape', img.shape)
        three,imh,imw = img.shape
        imx = np.zeros((imh,imw,three), np.float32)
        for i in range(3):
            imx[:,:,i] = img[i,:,:] / 256.
        img = imx
        
        # Compute index arrays Yo,Xo,Yi,Xi (x and y pixel coordinates, input and output)
        # for nearest-neighbor resampling from wcs to targetwcs
        try:
            Yo,Xo,Yi,Xi,nil = resample_with_wcs(targetwcs, wcs, [], 3)
        except NoOverlapError:
            print('No actual overlap')
            continue
        print(len(Yo), 'resampled pixels')
        if len(Yo) == 0:
            continue

        # create the resampled image
        resampled_img = np.zeros((H,W,3), img.dtype)
        resampled_mask = np.zeros((H,W), bool)
        for band in range(3):
            resampled_img[Yo, Xo, band] = img[Yi, Xi, band]
        resampled_mask[Yo,Xo] = True

        plt.clf()
        plt.subplot(1,2,1)
        dimshow(resampled_img)
        plt.subplot(1,2,2)
        dimshow(resampled_mask)
        ps.savefig()
        
        # feed it to "enhance"
        enhance.update(resampled_mask.ravel(), img[Yi,Xi,:])

        plt.clf()
        dimshow(enhance.enhI.reshape((H,W,3)))
        ps.savefig()

    sys.exit(0)
    
    # test stretching
    img = plt.imread('demo/apod1.jpg')
    (H,W,B) = img.shape
    print('Image', img.shape, img.dtype)

    imx = np.sqrt(img.astype(np.float32) / 255.)

    enhance = EnhanceImage(H*W, B)
    enhance.update(np.ones((H*W), bool), imx.reshape((-1,B)))

    en = enhance.enhI.reshape((H,W,B))
    
    plt.clf()
    plt.imshow(en, interpolation='nearest', origin='lower')
    plt.savefig('en.png')

    stretch = enhance.stretch_to_match(img).reshape((H,W,B))
    
    plt.clf()
    plt.imshow(stretch, interpolation='nearest', origin='lower')
    plt.savefig('stretch1.png')
    
    img2 = plt.imread('demo/apod3.jpg')
    
    stretch2 = enhance.stretch_to_match(img2).reshape((H,W,B))
    
    plt.clf()
    plt.imshow(stretch2, interpolation='nearest', origin='lower')
    plt.savefig('stretch2.png')
    
