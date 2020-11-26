#include <stdio.h>

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

int main()
{
    int *host_atlas = nullptr;
    int *device_atlas = nullptr;

    host_atlas = (int*)malloc(1920 * 1080 * sizeof(int));
    cudaMalloc((void**) &device_atlas, 1920 * 1080 * sizeof(int));
    dim3 d(16, 16, 1);
    mandelbrot_set<<<d, d>>>(-0.748768, -0.748718, 0.0650619375, 0.0650900625, 1920, 1080, 2048, device_atlas);
    cudaMemcpy(host_atlas, device_atlas, 1920 * 1080 * sizeof(int), cudaMemcpyDeviceToHost);

    FILE *fp = fopen("MathPic.ppm","wb");
    int max = 0;
    fprintf(fp, "P6\n%d %d\n255\n", 1920, 1080);
    for(int i = 0; i < 1920 * 1080; i++)
    {
        if(host_atlas[i] > max)
            max = host_atlas[i];
    }
    for(int i = 0; i < 1920 * 1080; i++)
    {
        char c = host_atlas[i] * 255 / max;
        fwrite(&c, 1, 1, fp);
        fwrite(&c, 1, 1, fp);
        fwrite(&c, 1, 1, fp);
    }
    fclose(fp);

    return 0;
}
