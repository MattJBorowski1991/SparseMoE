#include <cuda_runtime.h>
#include "../include/config.h"
#include "../include/moe_args.h"
#include "../utils/check_cuda.h"
#include <stdio.h>

// Each kernel file other than unfused must define its kernel e.g.: #define MOE_KERNEL baseline.cu

#ifndef MOE_LAUNCH
#define MOE_LAUNCH(args)    \
    do {    \
        dim3 threads(THREADS_PER_WARP * WARPS_PER_BLOCK);   \
        dim3 blocks( (d_model + WMMA_N * WARP_TILES_X - 1) / (WMMA_N * WARP_TILES_X), (N + WMMA_M * WARP_TILES_Y - 1) / (WMMA_M * WARP_TILES_Y), args.num_batches );    \
        CHECK_CUDA(cudaFuncSetAttribute((const void*)MOE_KERNEL, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, 102400)); \
        MOE_KERNEL<<<blocks, threads>>>(args);   \
    } while(0)
#endif


extern "C" void solve(MoEArgs args){

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    // Ensure the launcher sets the per-kernel capacity flag according to the kernel's declaration
#ifdef MOE_USES_CAPACITY
    args.use_capacity = (MOE_USES_CAPACITY != 0);
#endif

    MOE_LAUNCH(args);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    printf("Total duration via cudaEventRecord: %.3f ms\n", ms);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}