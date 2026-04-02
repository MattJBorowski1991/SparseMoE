#include <math.h>

#ifndef CONFIG_H
#define CONFIG_H

constexpr int N = 256; // prefill: profile for p50 prompt length (~N=256) , then p95 prompt length (~N=2048), then p=99 (~N=4096); decode: profile for N=1
constexpr int d_model = 4096;
constexpr int h = 32;
static_assert( d_model % 32 == 0, "d_model must be divisible by number of heads h");

constexpr int num_batches = 4;

constexpr int num_experts = 32;
constexpr int k = 4;

constexpr int THREADS_PER_WARP = 32;
static_assert( k < THREADS_PER_WARP, "chosen experts cannot exceed 32");

// --- Tiling and launch constants ---
//WMMA int8: use only 16x16x16, 32x8x16, 8x32x16.

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WMMA_K_FP16 16
#define PAD 0

#define WARP_TILES_X 4
#define WARP_TILES_Y 2
#define WARPS_PER_BLOCK (WARP_TILES_X * WARP_TILES_Y)

// Up projection for the per-expert first MLP layer
constexpr int up_proj_dim = 4;

constexpr float capacity_factor = 0.8f;


constexpr int NSTREAMS = 4;

#endif // CONFIG_H