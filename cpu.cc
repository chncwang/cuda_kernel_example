#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define DSIZE 1000000
unsigned long long dtime_usec(unsigned long long prev){
#define USECPSEC 1000000ULL
    timeval tv1;
    gettimeofday(&tv1,0);
    return ((tv1.tv_sec * USECPSEC)+tv1.tv_usec) - prev;
}

void max_idx_kernel(const float *data, const int dsize, int *result) {
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
    int max_index = 0;
    unsigned long long dtime = dtime_usec(0);
    max_idx_kernel(h_vector, DSIZE, &max_index);
    dtime = dtime_usec(dtime);
    std::cout << "kernel time: " << dtime/(float)USECPSEC << " max index: " << max_index << std::endl;

    return 0;
}
