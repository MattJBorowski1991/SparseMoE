#include <math.h>

#ifndef CONFIG_H
#define CONFIG_H

constexpr int N = 1024;
constexpr int d_model = 4096;
constexpr int h = 32;
static_assert( d_model % 32, "d_model must be divisible by number of heads h");

constexpr int num_batches = 32;

constexpr int num_experts = 64;
constexpr int top_k = 4;

constexpr int NSTREAMS = 4;

#endif // CONFIG_H