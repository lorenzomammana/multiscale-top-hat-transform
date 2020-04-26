Python + CUDA implementation of "Infrared image enhancement through contrast enhancement by using multiscale new top-hat transform" Xiangzhi Bai, Fugen Zhou, Bindang Xue (2011).

Using the same parameters contained in the paper and an image of dimension 460x630 the algorithm requires less than 100ms to run, while the Scipy implementation requires more than 2 minutes.

The CUDA implementation of the morphological operators is based on https://github.com/lorenzomammana/morphological-operators-cupy.
