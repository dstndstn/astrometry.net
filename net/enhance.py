import numpy as np

def random_init(H, W, bands):
    enhI = np.zeros((H,W,bands), np.float32)
    enhW = np.zeros((H,W), np.float32) + 1e-3
    for b in range(bands):
        enhI[:,:,b] = np.random.permutation(W*H) / float(W*H)
    return enhI, enhW

def pixel_ranks(img, get_argsort=False):
    II = np.argsort(img)
    rankimg = np.empty_like(II)
    rankimg[II] = np.arange(len(II))
    if get_argsort:
        return rankimg,II
    return rankimg

def update(enhI, enhW, enhM, img, weightFactor=1.):
    rankimg = pixel_ranks(img)
    wenh = enhW[enhM]
    H,W,bands = enhI.shape
    for b in range(bands):
        enh = enhI[:,:,b][enhM]
        rankenh,EI = pixel_ranks(enh, get_argsort=True)
        rank = ( ((rankenh * wenh) + (rankimg * weightFactor))
                    / (wenh + weightFactor) )
        rank = pixel_ranks(rank)
        Enew = enh[EI[rank]]
        enhI[:,:,b][enhM] = Enew
    enhW[enhM] += 1.
    return enhI, enhW

