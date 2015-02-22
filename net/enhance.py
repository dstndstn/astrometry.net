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
                enhI[:,b] = np.random.permutation(npix) / float(npix)

    def update(self, mask, img, weightFactor=1., addRandom=True):
        '''
        Updates this EnhanceImage with the given new image *img*.

        *mask* describes the pixels in this EnhanceImage corresponding
         to the pixels in *img*.

         *img* must be "flattened", having shape (npix, nbands).

         *img* must be floating-point with values between 0 and 1.

         If *addRandom* is true, a small amount of uniform noise is
         added to break ties between pixels.
        '''
        wenh = self.enhW[mask]
        npix,nbands = self.enhI.shape

        assert(len(img.shape) == 2)
        assert(img.shape[1] == nbands)
        assert(len(wenh) == img.shape[0])
        assert(np.min(img) >= 0.)
        assert(np.max(img) <= 1.)

        for b in range(nbands):
            imgb = img[:,b]
            if addRandom:
                imgb = imgb + np.random.uniform(0., 1., size=imgb.shape)
            imgrank = pixel_ranks(imgb)
            enh = self.enhI[mask, b]
            enhrank,EI = pixel_ranks(enh, get_argsort=True)
            rank = ( ((enhrank * wenh) + (imgrank * weightFactor))
                        / (wenh + weightFactor) )
            rank = pixel_ranks(rank)
            enhnew = enh[EI[rank]]
            self.enhI[mask,b] = enhnew
        self.enhW[mask] += 1.

