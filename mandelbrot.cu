#include <cuComplex.h>

#define CUDA_KERNEL_LOOP_x(i,n) \
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; \
        i < (n); \
        i += blockDim.x * gridDim.x)

#define CUDA_KERNEL_LOOP_y(j,m) \
    for(int j = blockIdx.y * blockDim.y + threadIdx.y; \
        j < (m); \
        j += blockDim.y * gridDim.y)

__device__ int mandelbrot(cuDoubleComplex c, int threshold)
{
    cuDoubleComplex z = make_cuDoubleComplex(0, 0);
    for(int i = 0; i < threshold; i++)
    {
        z = cuCadd(cuCmul(z, z), c);
        if(cuCabs(z) > 2)
            return i;
    }
    return 0;
}

__global__ void mandelbrot_set(double xmin, double xmax, double ymin, double ymax,
                               int xn, int yn, int threshold, int *atlas)
{
    CUDA_KERNEL_LOOP_y(j, yn)
    {
        CUDA_KERNEL_LOOP_x(i, xn)
        {
            double cx = xmin + i * (xmax - xmin) / xn;
            double cy = ymin + j * (ymax - ymin) / yn;
            cuDoubleComplex c = make_cuDoubleComplex(cx, cy);
            atlas[j * xn + i] = mandelbrot(c, threshold);
        }
    }
}
