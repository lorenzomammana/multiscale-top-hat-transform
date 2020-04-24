from scipy.ndimage import grey_opening, grey_closing
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer


def square_closing(img, db, bb):
    return grey_closing(grey_opening(img, structure=db), structure=bb)


def square_opening(img, db, bb):
    return grey_opening(grey_closing(img, structure=db), structure=bb)


def top_hat(img, db, bb):
    NWTH = img - np.minimum(img, square_closing(img, db, bb))
    NBTH = np.maximum(img, square_opening(img, db, bb)) - img

    return [NWTH, NBTH]


def multiscale_top_hat(img, nw, nl, nm, ns, n):
    NWTH_out = np.zeros_like(img)
    NBTH_out = np.zeros_like(img)

    for i in range(n):
        bb = np.zeros([nl + ns * i, nl + ns * i])
        db = np.pad(np.zeros([nw + ns * i, nw + ns * i]), [nm, nm])

        single_scale_top_hat = top_hat(img, db, bb)
        NWTH_out = np.maximum(NWTH_out, single_scale_top_hat[0])
        NBTH_out = np.maximum(NBTH_out, single_scale_top_hat[1])

    return [NWTH_out, NBTH_out]


if __name__ == '__main__':
    image = io.imread('01.jpg')
    image = np.array(image[:, :, 0]).astype(int)

    ax = plt.hist(image.ravel(), bins=256)
    plt.show()

    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()
    nW = 5
    nL = 5
    nM = 2
    nS = 11
    n = 9

    start = timer()
    [NWTH, NBTH] = multiscale_top_hat(image, nW, nL, nM, nS, n)
    end = timer()

    print(end - start)
    out = image * 0.2 + 5 * NWTH - 3 * NBTH
    out[out > 255] = 255
    out[out < 0] = 0

    ax = plt.hist(out.ravel(), bins=256)
    plt.show()
    plt.imshow(out, cmap='gray', vmin=0, vmax=255)
    plt.show()
