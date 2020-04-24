#include <cuda_runtime.h>

extern "C"
{
    __global__ void dilation(int * src, int * dst, int p, int window_size, int n_window, int image_shape)
    {
        extern __shared__ int smem[];
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int bx = blockIdx.x;

        if (tx == 0)
        {
            for (int i = p - 1; i >= 0; i--)
            {
                if (i == p - 1)
                    smem[ty * p + i] = src[bx * p * n_window + ty * p + i];
                else
                    smem[ty * p + i] = max(src[bx * p * n_window + ty * p + i], smem[ty * p + (i + 1)]);
            }
        }
        else
        {
            for (int i = 0; i <= p - 1; i++)
            {
                if (i == 0)
                    smem[n_window * p + (ty * p) + i] = src[bx * p * n_window + ty * p + (i + p - 1)];
                else
                    smem[n_window * p + (ty * p) + i] = max(src[bx * p *n_window + ty * p + (i + p - 1)],
                                                 smem[n_window * p + (ty * p) + (i - 1)]);
            }
        }
        __syncthreads();

        if (tx == 0)
        {
            for (int i = 0; i < p; i++)
            {
                // Skip first p-1 / 2 because of padding
                int original_index = bx * p * n_window + ty * p + i + ((p - 1)/2);

                if (original_index < image_shape)
                {
                    dst[original_index] = max(smem[ty * p + i], smem[n_window * p + (ty * p) + i]);
                }
            }
        }

    }
}

extern "C"
{
    __global__ void erosion(int * src, int * dst, int p, int window_size, int n_window, int image_shape)
    {
        extern __shared__ int smem[];
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int bx = blockIdx.x;

        if (tx == 0)
        {
            for (int i = p - 1; i >= 0; i--)
            {
                if (i == p - 1)
                    smem[ty * p + i] = src[bx * p * n_window + ty * p + i];
                else
                    smem[ty * p + i] = min(src[bx * p * n_window + ty * p + i], smem[ty * p + (i + 1)]);
            }
        }
        else
        {
            for (int i = 0; i <= p - 1; i++)
            {
                if (i == 0)
                    smem[n_window * p + (ty * p) + i] = src[bx * p * n_window + ty * p + (i + p - 1)];
                else
                    smem[n_window * p + (ty * p) + i] = min(src[bx * p *n_window + ty * p + (i + p - 1)],
                                                 smem[n_window * p + (ty * p) + (i - 1)]);
            }
        }
        __syncthreads();

        if (tx == 0)
        {
            for (int i = 0; i < p; i++)
            {
                int original_index = bx * p * n_window + ty * p + i + ((p - 1)/2);

                if (original_index < image_shape)
                {
                    dst[original_index] = min(smem[ty * p + i], smem[n_window * p + (ty * p) + i]);
                }
            }
        }

    }
}