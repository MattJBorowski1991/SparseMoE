#include "include/config.h"
#include "include/moe_args.h"
#include <mma.h>
using namespace nvcuda;
#include <cuda_fp8.h>
#include <stdio.h>
#include <assert.h>
#include <cstdint>

#define MOE_KERNEL capacity
#define MOE_USES_CAPACITY 1

constexpr int PTX_MMA_K = 32;  // m16n8k32 for fp8

// ---------------------------------------------------------------------------
// FP8 e4m3 PTX mma.sync:
//   shape:   m16n8k32
//   A:       16x32 e4m3, row-major  → 4x uint32_t per thread (packed, 4 bytes each)
//   B:       32x8  e4m3, col-major  → 2x uint32_t per thread (packed)
//   C/D:     16x8  f32             → 4x float per thread
//   Two ops per warp tile to cover m16n16k32
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// ptx_mma_m16n8k32_e4m3: one m16n8k32 fp8e4m3->fp32 mma.sync
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void ptx_mma_m16n8k32_e4m3(
    const uint32_t a[4],
    const uint32_t b[2],
    float          c[4]
){
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),
           "r"(b[0]),  "r"(b[1])
    );
}

// ---------------------------------------------------------------------------
// load_a_m16k32_fp8: load A tile (16x32 fp8, row-major) into 4 uint32 registers
//   identical byte layout to int8 version — fp8 is also 1 byte per element
//   a[0]: row (lane/4)*2+0, cols  0..15  packed as 4 bytes → uint32
//   a[1]: row (lane/4)*2+0, cols 16..31  packed
//   a[2]: row (lane/4)*2+1, cols  0..15  packed
//   a[3]: row (lane/4)*2+1, cols 16..31  packed
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void load_a_m16k32_fp8(
    const __nv_fp8_e4m3* tile,  // [WMMA_M][PTX_MMA_K + PAD]
    int                  ldA,   // = PTX_MMA_K + PAD
    uint32_t             a[4]
){
    int lane = threadIdx.x % 32;
    int row0 = (lane / 4) * 2;
    int row1 = row0 + 1;
    int col  = (lane % 4) * 4;  // 4 consecutive fp8 → 1 uint32

    a[0] = *reinterpret_cast<const uint32_t*>(&tile[row0 * ldA + col]);
    a[2] = *reinterpret_cast<const uint32_t*>(&tile[row1 * ldA + col]);
    a[1] = *reinterpret_cast<const uint32_t*>(&tile[row0 * ldA + col + 16]);
    a[3] = *reinterpret_cast<const uint32_t*>(&tile[row1 * ldA + col + 16]);
}

// ---------------------------------------------------------------------------
// load_b_m8k32_fp8: load B tile (32x8 fp8, col-major) into 2 uint32 registers
//   identical byte layout to int8 version
//   b[0]: k-rows  0.. 3, col owned by this thread  (pack4)
//   b[1]: k-rows 16..19, col owned by this thread
//   (PTX m16n8k32 fp8 B layout matches int8 exactly — 1 byte/elem)
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void load_b_m8k32_fp8(
    const __nv_fp8_e4m3* tile,       // [PTX_MMA_K][WMMA_N + PAD]
    int                  ldB,        // = WMMA_N + PAD
    int                  n_col_base, // 0 or 8
    uint32_t             b[2]
){
    int lane = threadIdx.x % 32;
    int col  = n_col_base + (lane / 4) % 2;

    auto pack4 = [&](int k0) -> uint32_t {
        uint32_t v = 0;
        v |= ((uint32_t)(uint8_t)tile[(k0+0)*ldB + col]);
        v |= ((uint32_t)(uint8_t)tile[(k0+1)*ldB + col]) << 8;
        v |= ((uint32_t)(uint8_t)tile[(k0+2)*ldB + col]) << 16;
        v |= ((uint32_t)(uint8_t)tile[(k0+3)*ldB + col]) << 24;
        return v;
    };

    b[0] = pack4(0);
    b[1] = pack4(16);
}

// ---------------------------------------------------------------------------
// wmma_db_fp8: double-buffered fp8 GEMM via PTX mma.sync
//   two m16n8k32 ops per warp tile → covers m16n16k32
//   accumulator is float (fp32)
//   output written directly to C_global as float (no int32 scratch needed)
// ---------------------------------------------------------------------------
template<bool calculatePerExpert>
static __device__ __forceinline__ void wmma_db_fp8(
    const __nv_fp8_e4m3* A,
    const __nv_fp8_e4m3* B,
    float*               C_global,  // [M, N] float scratch
    int M, int N, int K,
    int tile_row_in,
    int tile_col_in
){
    assert((K % PTX_MMA_K) == 0);

    int batch   = blockIdx.z;
    int tid     = threadIdx.x;
    int warp_id = tid / THREADS_PER_WARP;
    int lane_id = tid % THREADS_PER_WARP;

    const __nv_fp8_e4m3* A_batch = A + batch * M * K;
    const __nv_fp8_e4m3* B_batch = B;  // weights not batched

    const __nv_fp8_e4m3* A_e;
    const __nv_fp8_e4m3* B_e;
    int tile_row_local = tile_row_in;

    if constexpr (calculatePerExpert) {
        const int rows_per_expert = M / num_experts;
        const int expert_id       = tile_row_in / rows_per_expert;
        tile_row_local            = tile_row_in % rows_per_expert;
        A_e = A_batch + expert_id * rows_per_expert * K;
        B_e = B_batch + expert_id * K * N;
    } else {
        A_e = A_batch;
        B_e = B_batch;
    }

    // fp8 smem tiles — same byte size as int8 tiles (1 byte/element)
    __shared__ __align__(16) __nv_fp8_e4m3 As8[2][WARPS_PER_BLOCK][WMMA_M][PTX_MMA_K + PAD];
    __shared__ __align__(16) __nv_fp8_e4m3 Bs8[2][WARPS_PER_BLOCK][PTX_MMA_K][WMMA_N + PAD];

    // fp32 accumulators: two m16n8k32 ops → D0 (left n8), D1 (right n8)
    float D0[4] = {0.f, 0.f, 0.f, 0.f};
    float D1[4] = {0.f, 0.f, 0.f, 0.f};

    int buf = 0;

    // --- initial load ---
    for (int i = lane_id; i < WMMA_M * PTX_MMA_K; i += 32) {
        int row = i / PTX_MMA_K, col = i % PTX_MMA_K;
        As8[buf][warp_id][row][col] = A_e[(tile_row_local + row) * K + col];
    }
    for (int i = lane_id; i < PTX_MMA_K * WMMA_N; i += 32) {
        int row = i / WMMA_N, col = i % WMMA_N;
        Bs8[buf][warp_id][row][col] = B_e[row * N + (tile_col_in + col)];
    }
    __syncthreads();

    uint32_t a[4], b0[2], b1[2];
    load_a_m16k32_fp8(&As8[buf][warp_id][0][0], PTX_MMA_K + PAD, a);
    load_b_m8k32_fp8 (&Bs8[buf][warp_id][0][0], WMMA_N + PAD, 0, b0);
    load_b_m8k32_fp8 (&Bs8[buf][warp_id][0][0], WMMA_N + PAD, 8, b1);

    // --- main loop ---
    for (int k_off = PTX_MMA_K; k_off < K; k_off += PTX_MMA_K) {
        int next = 1 - buf;

        // cp.async: fp8 is 1 byte, 16 bytes = 16 elements per issue
        for (int i = lane_id * 16; i < WMMA_M * PTX_MMA_K; i += 32 * 16) {
            int row = i / PTX_MMA_K, col = i % PTX_MMA_K;
            char*       dst = (char*)&As8[next][warp_id][row][col];
            const char* src = (const char*)&A_e[(tile_row_local + row) * K + (k_off + col)];
            unsigned smem_ptr = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(smem_ptr), "l"(src));
        }
        for (int i = lane_id * 16; i < PTX_MMA_K * WMMA_N; i += THREADS_PER_WARP * 16) {
            int row = i / WMMA_N, col = i % WMMA_N;
            char*       dst = (char*)&Bs8[next][warp_id][row][col];
            const char* src = (const char*)&B_e[(k_off + row) * N + (tile_col_in + col)];
            unsigned smem_ptr = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(smem_ptr), "l"(src));
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        // compute on current tile
        ptx_mma_m16n8k32_e4m3(a, b0, D0);
        ptx_mma_m16n8k32_e4m3(a, b1, D1);

        buf = next;
        load_a_m16k32_fp8(&As8[buf][warp_id][0][0], PTX_MMA_K + PAD, a);
        load_b_m8k32_fp8 (&Bs8[buf][warp_id][0][0], WMMA_N + PAD, 0, b0);
        load_b_m8k32_fp8 (&Bs8[buf][warp_id][0][0], WMMA_N + PAD, 8, b1);
    }

    // compute last tile
    ptx_mma_m16n8k32_e4m3(a, b0, D0);
    ptx_mma_m16n8k32_e4m3(a, b1, D1);

    // --- store float accumulators to C_global ---
    // m16n8 output layout: each thread owns 2 rows x 1 col per n8 block
    // row0 = (lane/4)*2, row1 = row0+1, col = lane%4
    {
        int row0      = (lane_id / 4) * 2;
        int row1      = row0 + 1;
        int col_left  = lane_id % 4;
        int col_right = lane_id % 4;

        float* base = C_global + (blockIdx.z * M + tile_row_in) * N + tile_col_in;

        base[(row0) * N + col_left]         = D0[0];
        base[(row0) * N + col_left  + 4]    = D0[1];
        base[(row1) * N + col_left]         = D0[2];
        base[(row1) * N + col_left  + 4]    = D0[3];

        base[(row0) * N + 8 + col_right]    = D1[0];
        base[(row0) * N + 8 + col_right + 4]= D1[1];
        base[(row1) * N + 8 + col_right]    = D1[2];
        base[(row1) * N + 8 + col_right + 4]= D1[3];
    }
}

// ---------------------------------------------------------------------------
// wmma_db_fp16: router matmul — unchanged from int8 version
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void wmma_db_fp16(
    float alpha,
    const half* A,
    const half* B,
    float* C,
    int M, int N, int K
){
    assert( (M % WMMA_M == 0) && (N % WMMA_N == 0) && (K % WMMA_K_FP16 == 0) );
    int batch = blockIdx.z;

    const half* A_batch = A + batch * M * K;
    const half* B_batch = B;
    float*      C_batch = C + batch * M * N;

    int tid     = threadIdx.x;
    int warp_id = tid / THREADS_PER_WARP;
    int lane_id = tid % THREADS_PER_WARP;

    int warp_tile_row = warp_id / WARP_TILES_X;
    int warp_tile_col = warp_id % WARP_TILES_X;
    const int tile_row = blockIdx.y * (WMMA_M * WARP_TILES_Y) + warp_tile_row * WMMA_M;
    const int tile_col = blockIdx.x * (WMMA_N * WARP_TILES_X) + warp_tile_col * WMMA_N;
    if (tile_row >= M || tile_col >= N) return;

    __shared__ __align__(16) half As[2][WARPS_PER_BLOCK][WMMA_M][WMMA_K_FP16 + PAD];
    __shared__ __align__(16) half Bs[2][WARPS_PER_BLOCK][WMMA_K_FP16][WMMA_N + PAD];

    wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K_FP16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K_FP16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K_FP16, float>                 c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int buf = 0;

    for (int i = lane_id; i < WMMA_M * WMMA_K_FP16; i += 32) {
        int row = i / WMMA_K_FP16, col = i % WMMA_K_FP16;
        As[buf][warp_id][row][col] = A_batch[(tile_row + row) * K + col];
    }
    for (int i = lane_id; i < WMMA_K_FP16 * WMMA_N; i += 32) {
        int row = i / WMMA_N, col = i % WMMA_N;
        Bs[buf][warp_id][row][col] = B_batch[row * N + (tile_col + col)];
    }
    __syncthreads();

    wmma::load_matrix_sync(a_frag, &As[buf][warp_id][0][0], WMMA_K_FP16 + PAD);
    wmma::load_matrix_sync(b_frag, &Bs[buf][warp_id][0][0], WMMA_N + PAD);

    for (int k_off = WMMA_K_FP16; k_off < K; k_off += WMMA_K_FP16) {
        int next = 1 - buf;

        for (int i = lane_id * 8; i < WMMA_M * WMMA_K_FP16; i += 32 * 8) {
            int row = i / WMMA_K_FP16, col = i % WMMA_K_FP16;
            char*       dst      = (char*)&As[next][warp_id][row][col];
            const char* src      = (const char*)&A_batch[(tile_row + row) * K + (k_off + col)];
            unsigned    smem_ptr = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(smem_ptr), "l"(src));
        }
        for (int i = lane_id * 8; i < WMMA_K_FP16 * WMMA_N; i += THREADS_PER_WARP * 8) {
            int row = i / WMMA_N, col = i % WMMA_N;
            char*       dst      = (char*)&Bs[next][warp_id][row][col];
            const char* src      = (const char*)&B_batch[(k_off + row) * N + (tile_col + col)];
            unsigned    smem_ptr = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(smem_ptr), "l"(src));
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        buf = next;
        wmma::load_matrix_sync(a_frag, &As[buf][warp_id][0][0], WMMA_K_FP16 + PAD);
        wmma::load_matrix_sync(b_frag, &Bs[buf][warp_id][0][0], WMMA_N + PAD);
    }
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    float* c_dst = C_batch + tile_row * N + tile_col;
    for (int i = 0; i < c_frag.num_elements; ++i) c_frag.x[i] = alpha * c_frag.x[i];
    wmma::store_matrix_sync(c_dst, c_frag, N, wmma::mem_row_major);
}

// ---------------------------------------------------------------------------
// quantize_and_assign_per_expert_inputs: fp16 → fp8 e4m3
//   replaces int8 version — clamp to e4m3 range [-448, 448]
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void quantize_and_assign_per_expert_inputs(
    const half*           __restrict__ input,
    const int*            __restrict__ expert_counts,
    const int*            __restrict__ expert_token_ids,
    __nv_fp8_e4m3*        __restrict__ per_expert_wmma_inputs,
    float                              scale_input_act,
    int                                CAP
){
    const int batch = blockIdx.z;
    const half*         input_b            = input           + batch * N * d_model;
    const int*          expert_counts_b    = expert_counts   + batch * num_experts;
    const int*          expert_token_ids_b = expert_token_ids+ batch * num_experts * CAP;
    __nv_fp8_e4m3*      per_expert_b       = per_expert_wmma_inputs + batch * num_experts * CAP * d_model;

    const int tid     = threadIdx.x;
    const int warp_id = tid / THREADS_PER_WARP;
    const int lane_id = tid % THREADS_PER_WARP;
    const int row_id  = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row_id >= num_experts * CAP) return;

    const int expert_id = row_id / CAP;
    const int slot      = row_id % CAP;
    const int row_base  = row_id * d_model;

    // e4m3 max representable = 448.0
    constexpr float FP8_E4M3_MAX = 448.0f;

    if (slot < expert_counts_b[expert_id]) {
        const int token_id = expert_token_ids_b[expert_id * CAP + slot];
        if (token_id >= 0 && token_id < N) {
            const int in_base = token_id * d_model;
            for (int col = lane_id; col < d_model; col += THREADS_PER_WARP) {
                float val = __half2float(input_b[in_base + col]) / scale_input_act;
                val = fminf(fmaxf(val, -FP8_E4M3_MAX), FP8_E4M3_MAX);
                per_expert_b[row_base + col] = __nv_fp8_e4m3(val);
            }
        } else {
            for (int col = lane_id; col < d_model; col += THREADS_PER_WARP)
                per_expert_b[row_base + col] = __nv_fp8_e4m3(0.f);
        }
    } else {
        for (int col = lane_id; col < d_model; col += THREADS_PER_WARP)
            per_expert_b[row_base + col] = __nv_fp8_e4m3(0.f);
    }
}

// ---------------------------------------------------------------------------
// silu_and_requant_fp8:
//   dequant float up+gate (already fp32 from mma accum) → SiLU → requant fp8
//   vastly simpler than int8 version: no int32→float cast needed,
//   accumulators are already float, just apply scale and clamp
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void silu_and_requant_fp8(
    const float*   __restrict__ up_fp32,        // [num_experts, CAP, 4*d_model]
    const float*   __restrict__ gate_fp32,      // [num_experts, CAP, 4*d_model]
    __nv_fp8_e4m3* __restrict__ out_fp8,        // [num_experts, CAP, 4*d_model]
    float scale_input_act,
    float scale_up_w,
    float scale_gate_w,
    float scale_mid_act,
    int   total_size
){
    const int batch         = blockIdx.z;
    const float* up_b       = up_fp32   + batch * total_size;
    const float* gate_b     = gate_fp32 + batch * total_size;
    __nv_fp8_e4m3* out_b    = out_fp8   + batch * total_size;

    const int block_linear  = blockIdx.y * gridDim.x + blockIdx.x;
    const int global_tid    = block_linear * blockDim.x + threadIdx.x;
    const int global_stride = gridDim.x * gridDim.y * blockDim.x;

    // accum is already float, scales just compound
    const float dequant_up   = scale_input_act * scale_up_w;
    const float dequant_gate = scale_input_act * scale_gate_w;
    const float inv_scale_mid = 1.0f / scale_mid_act;
    constexpr float FP8_E4M3_MAX = 448.0f;

    for (int idx = global_tid; idx < total_size; idx += global_stride) {
        const float up_f   = up_b[idx]   * dequant_up;
        const float gate_f = gate_b[idx] * dequant_gate;
        const float silu   = gate_f / (1.0f + __expf(-gate_f));
        const float fused  = up_f * silu * inv_scale_mid;
        out_b[idx] = __nv_fp8_e4m3(fminf(fmaxf(fused, -FP8_E4M3_MAX), FP8_E4M3_MAX));
    }
}

// ---------------------------------------------------------------------------
// top_k_gating — unchanged
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void top_k_gating(
    const float* logits,
    int*   selected_expert_indices,
    float* selected_expert_weights,
    float* max_vals,
    int*   max_idxs
){
    int batch = blockIdx.z;
    const float* logits_batch              = logits + batch * N * num_experts;
    int*   selected_expert_indices_b       = selected_expert_indices + batch * N * k;
    float* selected_expert_weights_b       = selected_expert_weights + batch * N * k;

    const int tid      = threadIdx.x;
    const int warp_id  = tid / THREADS_PER_WARP;
    const int lane_id  = tid % THREADS_PER_WARP;
    const int token_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (token_id >= N) return;

    float* warp_max_vals = max_vals + warp_id * k;
    int*   warp_max_idxs = max_idxs + warp_id * k;
    const float* logits_row = logits_batch + token_id * num_experts;

    if (lane_id == 0) {
        for (int i = 0; i < k; ++i) { warp_max_vals[i] = -1e20f; warp_max_idxs[i] = -1; }
        for (int logit_id = 0; logit_id < num_experts; ++logit_id) {
            float val = logits_row[logit_id];
            if (val > warp_max_vals[k - 1]) {
                warp_max_vals[k - 1] = val; warp_max_idxs[k - 1] = logit_id;
                for (int i = k - 1; i > 0 && warp_max_vals[i] > warp_max_vals[i - 1]; --i) {
                    float tv = warp_max_vals[i-1]; warp_max_vals[i-1] = warp_max_vals[i]; warp_max_vals[i] = tv;
                    int   ti = warp_max_idxs[i-1]; warp_max_idxs[i-1] = warp_max_idxs[i]; warp_max_idxs[i] = ti;
                }
            }
        }
        float max_val = warp_max_vals[0], sum_of_exps = 0.0f;
        for (int l = 0; l < k; ++l) sum_of_exps += expf(warp_max_vals[l] - max_val);
        for (int l = 0; l < k; ++l) {
            selected_expert_indices_b[token_id * k + l] = warp_max_idxs[l];
            selected_expert_weights_b[token_id * k + l] = expf(warp_max_vals[l] - max_val) / (sum_of_exps + 1e-10f);
        }
    }
}

// ---------------------------------------------------------------------------
// build_per_expert_buffers — unchanged
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void build_per_expert_buffers(
    const int*   __restrict__ selected_expert_indices,
    const float* __restrict__ selected_expert_weights,
    int*         __restrict__ expert_counts,
    int*         __restrict__ expert_token_ids,
    float*       __restrict__ expert_token_weights,
    int CAP
){
    const int batch      = blockIdx.z;
    const int* sel_idx_b = selected_expert_indices + batch * N * k;
    const float* sel_w_b = selected_expert_weights + batch * N * k;
    int*   counts_b      = expert_counts           + batch * num_experts;
    int*   tok_ids_b     = expert_token_ids        + batch * num_experts * CAP;
    float* tok_w_b       = expert_token_weights    + batch * num_experts * CAP;

    const int tid = threadIdx.x, warp_id = tid / THREADS_PER_WARP, lane_id = tid % THREADS_PER_WARP;
    const int warp_linear = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int warp_stride = gridDim.x  * WARPS_PER_BLOCK;

    for (int route_id = warp_linear; route_id < N * k; route_id += warp_stride) {
        if (lane_id == 0) {
            const int token_id  = route_id / k;
            const int expert_id = sel_idx_b[route_id];
            if (expert_id >= 0 && expert_id < num_experts) {
                const int slot = atomicAdd(&counts_b[expert_id], 1);
                if (slot < CAP) {
                    tok_ids_b[expert_id * CAP + slot] = token_id;
                    tok_w_b  [expert_id * CAP + slot] = sel_w_b[route_id];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// clamp_expert_counts — unchanged
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void clamp_expert_counts(
    int* __restrict__ expert_counts, int CAP
){
    const int batch         = blockIdx.z;
    int* counts_b           = expert_counts + batch * num_experts;
    const int global_tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_stride = gridDim.x   * blockDim.x;
    for (int idx = global_tid; idx < num_experts; idx += global_stride) {
        if (counts_b[idx] > CAP) counts_b[idx] = CAP;
    }
}

// ---------------------------------------------------------------------------
// combine: fp32 accum already — just apply scale and weighted sum
//   simpler than int8: no int32 cast, input is already float
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void combine(
    const float* __restrict__ input,               // float down_proj output [num_experts, CAP, d_model]
    const int*   __restrict__ expert_token_ids,
    const float* __restrict__ expert_token_weights,
    const int*   __restrict__ expert_counts,
    float*                    final_output,
    float scale_mid_act,
    float scale_down_w,
    int   CAP
){
    const int batch           = blockIdx.z;
    const float dequant       = scale_mid_act * scale_down_w;
    const int rows_per_expert = CAP;

    const float* input_b   = input              + batch * num_experts * rows_per_expert * d_model;
    const int*   tok_ids_b = expert_token_ids   + batch * num_experts * rows_per_expert;
    const float* tok_w_b   = expert_token_weights + batch * num_experts * rows_per_expert;
    const int*   counts_b  = expert_counts       + batch * num_experts;
    float*       final_b   = final_output        + batch * N * d_model;

    const int tid     = threadIdx.x;
    const int warp_id = tid / THREADS_PER_WARP;
    const int lane_id = tid % THREADS_PER_WARP;
    const int row_id  = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row_id >= num_experts * rows_per_expert) return;

    const int expert_id = row_id / rows_per_expert;
    const int slot      = row_id % rows_per_expert;
    if (slot >= counts_b[expert_id]) return;

    const int token_id = tok_ids_b[expert_id * rows_per_expert + slot];
    if (token_id < 0 || token_id >= N) return;

    const float route_weight    = tok_w_b[expert_id * rows_per_expert + slot];
    const int   expert_row_base = row_id   * d_model;
    const int   token_row_base  = token_id * d_model;

    for (int col = lane_id; col < d_model; col += THREADS_PER_WARP) {
        float val = input_b[expert_row_base + col] * dequant;
        atomicAdd(&final_b[token_row_base + col], route_weight * val);
    }
}

// ---------------------------------------------------------------------------
// zero_final_output_and_expert_counts — unchanged
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void zero_final_output_and_expert_counts(
    float* __restrict__ final_output,
    int*   __restrict__ expert_counts
){
    const int batch = blockIdx.z;
    if (blockIdx.y != 0) return;

    float* final_b  = final_output  + batch * N * d_model;
    int*   counts_b = expert_counts + batch * num_experts;

    const int global_tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_stride = gridDim.x  * blockDim.x;

    for (int idx = global_tid; idx < num_experts; idx += global_stride)
        counts_b[idx] = 0;
    for (int idx = global_tid; idx < N * d_model; idx += global_stride)
        final_b[idx] = 0.0f;
}

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------
__global__ void capacity(MoEArgs args){

    const half*          input                    = args.input;
    const half*          router_weights           = args.router_weights;
    const __nv_fp8_e4m3* expert_up_proj_weights   = args.expert_up_proj_weights_fp8;
    const __nv_fp8_e4m3* expert_gate_proj_weights = args.expert_gate_proj_weights_fp8;
    const __nv_fp8_e4m3* expert_down_proj_weights = args.expert_down_proj_weights_fp8;

    const float scale_up_w      = args.scale_up_w;
    const float scale_gate_w    = args.scale_gate_w;
    const float scale_down_w    = args.scale_down_w;
    const float scale_input_act = args.scale_input_act;
    const float scale_mid_act   = args.scale_mid_act;

    float*          expert_logits           = args.expert_logits;
    int*            selected_expert_indices = args.selected_expert_indices;
    float*          selected_expert_weights = args.selected_expert_weights;
    int*            expert_counts           = args.expert_counts;
    int*            expert_token_ids        = args.expert_token_ids;
    float*          expert_token_weights    = args.expert_token_weights;
    __nv_fp8_e4m3*  per_expert_wmma_inputs  = args.per_expert_wmma_inputs_fp8;
    __nv_fp8_e4m3*  hidden_mlp_fp8          = args.hidden_mlp_layer_1_out_fp8;
    float*          final_output            = args.final_output;

    // fp32 scratch: accumulators are already float — reuse same host allocations
    // (sizeof(float) == sizeof(int32_t), no realloc needed vs int8 version)
    float* up_fp32   = args.hidden_mlp_layer_1_out;   // repurposed as float scratch
    float* gate_fp32 = args.hidden_mlp_gate_out;      // repurposed as float scratch
    float* down_fp32 = args.hidden_mlp_layer_2_out;   // repurposed as float scratch

    int CAP_raw = (int)ceilf((float)N * (float)k / (float)num_experts * capacity_factor);
    int CAP     = ((CAP_raw + WMMA_M - 1) / WMMA_M) * WMMA_M;

    // --- Stage 0 ---
    zero_final_output_and_expert_counts(final_output, expert_counts);

    // --- Stage 1: router GEMM fp16 ---
    wmma_db_fp16(1.0f, input, router_weights, expert_logits, N, num_experts, d_model);

    // --- Stage 2: top-k gating ---
    __shared__ float max_vals[WARPS_PER_BLOCK * k];
    __shared__ int   max_indices[WARPS_PER_BLOCK * k];
    top_k_gating(expert_logits, selected_expert_indices, selected_expert_weights, max_vals, max_indices);
    __syncthreads();

    // --- Stage 3: build per-expert token lists ---
    build_per_expert_buffers(selected_expert_indices, selected_expert_weights,
                             expert_counts, expert_token_ids, expert_token_weights, CAP);
    __syncthreads();

    clamp_expert_counts(expert_counts, CAP);
    __syncthreads();

    // --- Stage 4: quantize fp16 activations → fp8 per-expert tiles ---
    quantize_and_assign_per_expert_inputs(input, expert_counts, expert_token_ids,
                                          per_expert_wmma_inputs, scale_input_act, CAP);
    __syncthreads();

    // --- Stage 5: up_proj and gate_proj fp8 GEMMs → float scratch ---
    {
        int tid           = threadIdx.x;
        int warp_id       = tid / THREADS_PER_WARP;
        int warp_tile_row = warp_id / WARP_TILES_X;
        int warp_tile_col = warp_id % WARP_TILES_X;
        const int M       = num_experts * CAP;
        const int Nw      = up_proj_dim * d_model;
        const int K       = d_model;
        const int tile_row = blockIdx.y * (WMMA_M * WARP_TILES_Y) + warp_tile_row * WMMA_M;
        const int tile_col = blockIdx.x * (WMMA_N * WARP_TILES_X) + warp_tile_col * WMMA_N;

        if (tile_row < M && tile_col < Nw) {
            wmma_db_fp8<true>(per_expert_wmma_inputs, expert_up_proj_weights,
                              up_fp32,   M, Nw, K, tile_row, tile_col);
            wmma_db_fp8<true>(per_expert_wmma_inputs, expert_gate_proj_weights,
                              gate_fp32, M, Nw, K, tile_row, tile_col);
        }
    }
    __syncthreads();

    // --- Stage 6: fused dequant + SiLU + requant → fp8 ---
    silu_and_requant_fp8(up_fp32, gate_fp32, hidden_mlp_fp8,
                         scale_input_act, scale_up_w, scale_gate_w, scale_mid_act,
                         num_experts * CAP * up_proj_dim * d_model);
    __syncthreads();

    // --- Stage 7: down_proj fp8 GEMM → float scratch ---
    {
        int tid           = threadIdx.x;
        int warp_id       = tid / THREADS_PER_WARP;
        int warp_tile_row = warp_id / WARP_TILES_X;
        int warp_tile_col = warp_id % WARP_TILES_X;
        const int M       = num_experts * CAP;
        const int Nd      = d_model;
        const int K       = up_proj_dim * d_model;
        const int tile_row = blockIdx.y * (WMMA_M * WARP_TILES_Y) + warp_tile_row * WMMA_M;
        const int tile_col = blockIdx.x * (WMMA_N * WARP_TILES_X) + warp_tile_col * WMMA_N;

        if (tile_row < M && tile_col < Nd) {
            wmma_db_fp8<true>(hidden_mlp_fp8, expert_down_proj_weights,
                              down_fp32, M, Nd, K, tile_row, tile_col);
        }
    }
    __syncthreads();

    // --- Stage 8: weighted combine → fp32 final output ---
    combine(down_fp32, expert_token_ids, expert_token_weights, expert_counts,
            final_output, scale_mid_act, scale_down_w, CAP);
}

#include "launcher.h"