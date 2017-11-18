#include <unistd.h>
#include <cublas_v2.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <stdlib.h>

#define DSIZE 1000000
// nTPB should be a power-of-2
#define nTPB 1024
#define MAX_KERNEL_BLOCKS 30
#define MAX_BLOCKS ((DSIZE/nTPB)+1)
#define MIN(a,b) ((a>b)?b:a)
#define FLOAT_MIN -1.0f

#include <time.h>
#include <sys/time.h>

unsigned long long dtime_usec(unsigned long long prev){
#define USECPSEC 1000000ULL
    timeval tv1;
    gettimeofday(&tv1,0);
    return ((tv1.tv_sec * USECPSEC)+tv1.tv_usec) - prev;
}

__device__ volatile float blk_vals[MAX_BLOCKS];
__device__ volatile int   blk_idxs[MAX_BLOCKS];
__device__ int   blk_num = 0;

__global__ void max_idx_kernel(const float *data, const int dsize, int *result) {
    int max = -1;
    int maxi;
    for (int i = 0; i< dsize; ++i) {
        if (data[i] > max) {
            max = data[i];
            maxi = i;
        }
    }
    *result = maxi;
}

int main(){

    int nrElements = DSIZE;
    float *d_vector, *h_vector;
    h_vector = new float[DSIZE];
    for (int i = 0; i < DSIZE; i++) h_vector[i] = rand()/(float)RAND_MAX;
    h_vector[10] = 10;  // create definite max element
    cublasHandle_t my_handle;
    cublasStatus_t my_status = cublasCreate(&my_handle);
    cudaMalloc(&d_vector, DSIZE*sizeof(float));
    cudaMemcpy(d_vector, h_vector, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    int max_index = 0;
    int *d_max_index;
    cudaMalloc(&d_max_index, sizeof(int));
    unsigned long long dtime = dtime_usec(0);
    max_idx_kernel<<<1, 1>>>(d_vector, DSIZE, d_max_index);
    cudaDeviceSynchronize();
    dtime = dtime_usec(dtime);
    std::cout << "kernel time: " << dtime/(float)USECPSEC;
    cudaMemcpy(&max_index, d_max_index, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << " max index: " << max_index << std::endl;

    return 0;
}
