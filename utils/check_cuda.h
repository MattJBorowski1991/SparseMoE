#ifndef CHECK_CUDA_H
#define CHECK_CUDA_H

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if(err != cudaSuccess){ \
        fprintf(stderr, "CUDA Error at: %s: %d: %s \n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)


#endif