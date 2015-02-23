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
                imgb = imgb + np.random.uniform(0., 1., size=imgb.shape)
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

