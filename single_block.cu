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
__device__ volatile float   vals[nTPB];
__device__ volatile int idxs[nTPB];


__global__ void max_idx_kernel(const float *data, const int dsize, int *result) {
    int idx = threadIdx.x;
    float my_val = FLOAT_MIN;
    int my_idx = -1;
    while (idx < dsize) {
        if (data[idx] > my_val) {
            my_val = data[idx];
            my_idx = idx;
        }
        idx += blockDim.x;
    }
    vals[threadIdx.x] = my_val;
    idxs[threadIdx.x] = my_idx;

    __syncthreads();
    for (int i = (nTPB>>1); i > 0; i>>=1) { // i = 512, 265, ...
        if (threadIdx.x < i) { // threadIdx.x is between 0 to 1023
            if (vals[threadIdx.x] < vals[threadIdx.x + i]) {
                vals[threadIdx.x] = vals[threadIdx.x+i];
                idxs[threadIdx.x] = idxs[threadIdx.x+i];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *result = idxs[0];
    }
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
    max_idx_kernel<<<1, nTPB>>>(d_vector, DSIZE, d_max_index);
    cudaDeviceSynchronize();
    dtime = dtime_usec(dtime);
    std::cout << "kernel time: " << dtime/(float)USECPSEC;
    cudaMemcpy(&max_index, d_max_index, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << " max index: " << max_index << std::endl;

    return 0;
}
