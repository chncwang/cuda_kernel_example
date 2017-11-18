#include <unistd.h>
#include <cublas_v2.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <stdlib.h>

#define DSIZE 10000
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

template <typename T>
__global__ void max_idx_kernel(const T *data, const int dsize, int *result){

    __shared__ volatile T   vals[nTPB];
    __shared__ volatile int idxs[nTPB];
    __shared__ volatile int last_block;
    int idx = threadIdx.x+blockDim.x*blockIdx.x; // idx is the current data's index
    last_block = 0;
    T   my_val = FLOAT_MIN;
    int my_idx = -1;
    // sweep from global memory
    while (idx < dsize){ // idx is in the array
        if (data[idx] > my_val) {
            my_val = data[idx];
            my_idx = idx; // find the max data
        }
        idx += blockDim.x*gridDim.x; // if the array is very long
    }
    // populate shared memory
    vals[threadIdx.x] = my_val;
    idxs[threadIdx.x] = my_idx; // load to the shared memory
    __syncthreads();
    // sweep in shared memory
    for (int i = (nTPB>>1); i > 0; i>>=1) { // i = 512, 265, ...
        if (threadIdx.x < i) { // threadIdx.x is between 0 to 1023
            if (vals[threadIdx.x] < vals[threadIdx.x + i]) {
                vals[threadIdx.x] = vals[threadIdx.x+i];
                idxs[threadIdx.x] = idxs[threadIdx.x+i];
            }
        }
        __syncthreads();
    }
    // perform block-level reduction
    if (!threadIdx.x) { // threadIdx.x is 0 , and there stores the max value
        blk_vals[blockIdx.x] = vals[0];
        blk_idxs[blockIdx.x] = idxs[0]; // store per block's max value into 
        if (atomicAdd(&blk_num, 1) == gridDim.x - 1) // then I am the last block
            last_block = 1;
    }
    __syncthreads();
    if (last_block){
        idx = threadIdx.x;
        my_val = FLOAT_MIN;
        my_idx = -1;
        while (idx < gridDim.x){
            if (blk_vals[idx] > my_val) {
                my_val = blk_vals[idx];
                my_idx = blk_idxs[idx];
            }
            idx += blockDim.x;
        }
        // populate shared memory
        vals[threadIdx.x] = my_val;
        idxs[threadIdx.x] = my_idx;
        __syncthreads();
        // sweep in shared memory
        for (int i = (nTPB>>1); i > 0; i>>=1){
            if (threadIdx.x < i) {
                if (vals[threadIdx.x] < vals[threadIdx.x + i]) {
                    vals[threadIdx.x] = vals[threadIdx.x+i];
                    idxs[threadIdx.x] = idxs[threadIdx.x+i];
                }
            }
            __syncthreads();}
        if (!threadIdx.x)
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
    unsigned long long dtime = dtime_usec(0);
    int *d_max_index;
    max_idx_kernel<<<MIN(MAX_KERNEL_BLOCKS, ((DSIZE+nTPB-1)/nTPB)), nTPB>>>(d_vector, DSIZE, d_max_index);
    dtime = dtime_usec(dtime);
    cudaMemcpy(&max_index, d_max_index, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "kernel time: " << dtime/(float)USECPSEC << " max index: " << max_index << std::endl;

    return 0;
}
