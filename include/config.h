#include <math.h>

#ifndef CONFIG_H
#define CONFIG_H

constexpr int N = 1024;
constexpr int d_model = 4096;
constexpr int h = 32;
static_assert( d_model % 32 == 0, "d_model must be divisible by number of heads h");

constexpr int num_batches = 4;

constexpr int num_experts = 64;
constexpr int k = 4;

constexpr int THREADS_PER_WARP = 32;
static_assert( k < THREADS_PER_WARP, "chosen experts cannot exceed 32");

constexpr int up_proj_dim = 4;

// For baseline.cu only: 
constexpr int max_tokens_per_expert = 100;


constexpr int NSTREAMS = 4;

#endif // CONFIG_H