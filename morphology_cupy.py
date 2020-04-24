import cupy as cp
import numpy as np

with open('morphology.cu', 'r') as f:
    code = f.read()

module = cp.RawModule(code=code)
dilation_cuda = module.get_function('dilation')
erosion_cuda = module.get_function('erosion')


def apply_morphology(image, p, operation, morph_func):
    [img, window_size, reconstruction_shape, pad_size, n_window,
     out, required_blocks, thread_per_block] = prepare_morph(image, p, operation)

    # 4 = sizeof(int)
    morph_func((required_blocks,), (2, thread_per_block),
               (img, out, p, window_size, n_window, img.shape[0]), shared_mem=2 * n_window * p * 4)

    out = out.reshape(reconstruction_shape)[:, pad_size:-pad_size].transpose()
    [out, window_size, reconstruction_shape, pad_size, n_window,
     out2, required_blocks, thread_per_block] = prepare_morph(out, p, operation)

    morph_func((required_blocks,), (2, thread_per_block),
               (out, out2, p, window_size, n_window, out.shape[0]), shared_mem=2 * n_window * p * 4)
    out2 = out2.reshape(reconstruction_shape)[:, pad_size:-pad_size].transpose()
    return out2


def prepare_morph(img, p, operation):
    window_size = 2 * p - 1

    pad_size = int((p - 1) / 2)

    if operation == 'dilation':
        pad_value = 0
    else:
        pad_value = 255

    img = cp.pad(img, ((0, 0), (pad_size, pad_size)), constant_values=pad_value)

    reconstruction_shape = (img.shape[0], img.shape[1])
    img = img.reshape(-1)
    n_window = int(np.floor(img.shape[0] / p))
    out = cp.zeros_like(img)
    required_padding = (p - np.mod(img.shape[0], 2 * p - 1))

    if required_padding > 0:
        img = cp.pad(img, (0, required_padding), constant_values=pad_value)

    required_blocks = int((n_window / 512) + 1)

    original_num_window = n_window
    if n_window > 512:
        thread_per_block = 512
        n_window = 512
    else:
        thread_per_block = n_window

    if 2 * n_window * p * 4 > dilation_cuda.max_dynamic_shared_size_bytes:
        max_window = int(np.floor(dilation_cuda.max_dynamic_shared_size_bytes / (2 * p * 4)))
        required_blocks = int((original_num_window / max_window) + 1)
        n_window = max_window
        thread_per_block = max_window

    return [img, window_size, reconstruction_shape, pad_size, n_window, out, required_blocks, thread_per_block]


def grey_dilation_cuda(image, p):
    return apply_morphology(image, p, 'dilation', dilation_cuda)


def grey_erosion_cuda(image, p):
    return apply_morphology(image, p, 'erosion', erosion_cuda)


def grey_opening_cuda(image, p):
    return grey_dilation_cuda(grey_erosion_cuda(image, p), p)


def grey_closing_cuda(image, p):
    return grey_erosion_cuda(grey_dilation_cuda(image, p), p)


def grey_top_hat_cuda(image, p):
    NWTH = image - cp.minimum(grey_opening_cuda(image, p), image)
    NBTH = cp.maximum(image, grey_closing_cuda(image, p)) - image

    return [NWTH, NBTH]

