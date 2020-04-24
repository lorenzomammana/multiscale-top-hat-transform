from morphology_cupy import *
from skimage import io
import matplotlib.pyplot as plt
import cupy as cp
from timeit import default_timer as timer


def square_closing(img, db, bb):
    return grey_closing_cuda(grey_opening_cuda(img, db), bb)


def square_opening(img, db, bb):
    return grey_opening_cuda(grey_closing_cuda(img, db), bb)


def top_hat(img, db, bb):
    NWTH = img - cp.minimum(img, square_closing(img, db, bb))
    NBTH = cp.maximum(img, square_opening(img, db, bb)) - img

    return [NWTH, NBTH]


def multiscale_top_hat(img, nw, nl, nm, ns, n):
    NWTH_out = cp.zeros_like(img)
    NBTH_out = cp.zeros_like(img)

    for i in range(n):
        bb = nl + ns * i
        db = nw + ns * i + 2 * nm

        single_scale_top_hat = top_hat(img, db, bb)
        NWTH_out = cp.maximum(NWTH_out, single_scale_top_hat[0])
        NBTH_out = cp.maximum(NBTH_out, single_scale_top_hat[1])

    return [NWTH_out, NBTH_out]


if __name__ == '__main__':
    image = io.imread('01.jpg')
    image = np.array(image[:, :, 0]).astype(int)

    ax = plt.hist(image.ravel(), bins=256)
    plt.show()

    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()

    image = cp.array(image)
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
    out = cp.asnumpy(out)

    ax = plt.hist(out.ravel(), bins=256)
    plt.show()
    plt.imshow(out, cmap='gray', vmin=0, vmax=255)
    plt.show()
