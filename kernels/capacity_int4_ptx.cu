#include "include/config.h"
#include "include/moe_args_int4.h"
#include <mma.h>
using namespace nvcuda;
#include <stdio.h>
#include <assert.h>
#include <cstdint>

#define MOE_KERNEL capacity
#define MOE_USES_CAPACITY 1

// ---------------------------------------------------------------------------
// W4A8: weights are packed int4 (2 per byte), activations are int8
// PTX mma.sync shape for int4: m16n8k64
//   A: 16x64 int8,  row-major → 4x uint32_t per thread (same as int8 PTX version)
//   B: 64x8  int4,  col-major → 2x uint32_t per thread (packed nibbles)
//   C/D: 16x8 int32           → 4x int32_t  per thread
//   Two ops per warp tile to cover m16n16k64
// ---------------------------------------------------------------------------

constexpr int PTX_MMA_K_INT4 = 64;  // int4 requires K=64

// ---------------------------------------------------------------------------
// ptx_mma_m16n8k64_s8s4: m16n8k64 int8(A) x int4(B) -> int32
//   A: 4x uint32 (int8, same layout as int8 PTX version, K=64 so covers 64 cols)
//   B: 2x uint32 (packed int4, col-major)
//   c: 4x int32 accumulator, updated in place
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void ptx_mma_m16n8k64_s8s4(
    const uint32_t a[4],
    const uint32_t b[2],
    int32_t        c[4]
){
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s8.s4.s32 "
        "{%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),
           "r"(b[0]),  "r"(b[1])
    );
}

// ---------------------------------------------------------------------------
// load_a_m16k64_s8: load A tile (16x64 int8, row-major) into 4 uint32 registers
//   K=64: each thread covers 2 rows, 8 consecutive int8 cols per k-half
//   Layout (m16k64 row-major per PTX spec):
//     a[0]: row (lane/4)*2+0, cols  0.. 3  packed (first  k-quarter)
//     a[1]: row (lane/4)*2+0, cols 32..35  packed (third  k-quarter)
//     a[2]: row (lane/4)*2+1, cols  0.. 3  packed
//     a[3]: row (lane/4)*2+1, cols 32..35  packed
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void load_a_m16k64_s8(
    const int8_t* tile,  // [WMMA_M][PTX_MMA_K_INT4 + PAD]
    int           ldA,   // = PTX_MMA_K_INT4 + PAD
    uint32_t      a[4]
){
    int lane = threadIdx.x % 32;
    int row0 = (lane / 4) * 2;
    int row1 = row0 + 1;
    int col  = (lane % 4) * 4;  // 4 consecutive int8 → 1 uint32

    // k-quarter 0 (cols 0..31 first 4)
    a[0] = *reinterpret_cast<const uint32_t*>(&tile[row0 * ldA + col]);
    a[2] = *reinterpret_cast<const uint32_t*>(&tile[row1 * ldA + col]);
    // k-quarter 2 (cols 32..63 first 4)
    a[1] = *reinterpret_cast<const uint32_t*>(&tile[row0 * ldA + col + 32]);
    a[3] = *reinterpret_cast<const uint32_t*>(&tile[row1 * ldA + col + 32]);
}

// ---------------------------------------------------------------------------
// load_b_m8k64_s4: load B tile (64x8 packed int4, col-major) into 2 uint32 registers
//   B stored in smem as [PTX_MMA_K_INT4][WMMA_N/2 + PAD] — packed: 2 int4 per byte
//   so physical smem row width = WMMA_N/2 bytes = 8 bytes for N=16
//   PTX layout for B (k64n8, col-major, int4):
//     b[0]: k-rows  0..7, col this thread owns  → 8 nibbles = 4 bytes = 1 uint32
//     b[1]: k-rows 32..39,col this thread owns
//   n_col_base: 0 or 8 (in logical int4 element space)
//   Physical col in packed smem = n_col_base/2 + offset
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void load_b_m8k64_s4(
    const uint8_t* tile,       // [PTX_MMA_K_INT4][WMMA_N/2 + PAD] packed int4
    int            ldB_bytes,  // = WMMA_N/2 + PAD (bytes per row)
    int            n_col_base, // 0 or 8 (logical int4 col base)
    uint32_t       b[2]
){
    int lane = threadIdx.x % 32;
    // each thread owns 1 logical col in [0..7] of its n8 block
    int logical_col = n_col_base + (lane / 4) % 2;
    // in packed storage: 2 int4 per byte, col c is at byte c/2, nibble c%2
    int byte_col    = logical_col / 2;
    int nibble_shift= (logical_col % 2) * 4;

    // pack 8 k-rows of int4 into one uint32
    // each k-row contributes one nibble; we pack 8 consecutive k-rows
    auto pack8_nibbles = [&](int k0) -> uint32_t {
        uint32_t v = 0;
        for (int i = 0; i < 8; ++i) {
            uint8_t byte = tile[(k0 + i) * ldB_bytes + byte_col];
            uint8_t nibble = (byte >> nibble_shift) & 0xF;
            // sign-extend nibble from 4-bit to fill slot (PTX expects sign-extended s4)
            // pack into output: slot i gets nibble in bits [i*4 .. i*4+3]
            v |= ((uint32_t)nibble) << (i * 4);
        }
        return v;
    };

    b[0] = pack8_nibbles(0);   // k-rows  0.. 7
    b[1] = pack8_nibbles(32);  // k-rows 32..39
}

// ---------------------------------------------------------------------------
// wmma_db_int4: W4A8 double-buffered GEMM via PTX mma.sync m16n8k64
//   A: int8  activations [M, K]
//   B: packed int4 weights [K, N/2] (2 nibbles per byte)
//   C_global: int32 output scratch [M, N]
//   Two m16n8k64 ops per warp tile → covers m16n16k64
// ---------------------------------------------------------------------------
template<bool calculatePerExpert>
static __device__ __forceinline__ void wmma_db_int4(
    const int8_t*  A,
    const uint8_t* B,          // packed int4: [K][N/2]
    int32_t*       C_global,
    int M, int N, int K,
    int tile_row_in,
    int tile_col_in
){
    assert((K % PTX_MMA_K_INT4) == 0);

    int batch   = blockIdx.z;
    int tid     = threadIdx.x;
    int warp_id = tid / THREADS_PER_WARP;
    int lane_id = tid % THREADS_PER_WARP;

    const int8_t*  A_batch = A + batch * M * K;
    const uint8_t* B_batch = B;  // weights not batched

    const int8_t*  A_e;
    const uint8_t* B_e;
    int tile_row_local = tile_row_in;

    if constexpr (calculatePerExpert) {
        const int rows_per_expert = M / num_experts;
        const int expert_id       = tile_row_in / rows_per_expert;
        tile_row_local            = tile_row_in % rows_per_expert;
        A_e = A_batch + expert_id * rows_per_expert * K;
        // B packed: [K][N/2] per expert
        B_e = B_batch + expert_id * K * (N / 2);
    } else {
        A_e = A_batch;
        B_e = B_batch;
    }

    // A smem: int8, [WMMA_M][PTX_MMA_K_INT4] — same byte count as int8 version with K=32
    // B smem: packed int4, [PTX_MMA_K_INT4][WMMA_N/2] — quarter the bytes of fp16 B tile
    __shared__ __align__(16) int8_t  As[2][WARPS_PER_BLOCK][WMMA_M][PTX_MMA_K_INT4 + PAD];
    __shared__ __align__(16) uint8_t Bs[2][WARPS_PER_BLOCK][PTX_MMA_K_INT4][WMMA_N / 2 + PAD];

    int32_t D0[4] = {0, 0, 0, 0};  // left  n8 accumulator
    int32_t D1[4] = {0, 0, 0, 0};  // right n8 accumulator

    int buf = 0;

    // --- initial load: A tile (int8, 16 bytes per thread covers 16 elements) ---
    for (int i = lane_id; i < WMMA_M * PTX_MMA_K_INT4; i += 32) {
        int row = i / PTX_MMA_K_INT4, col = i % PTX_MMA_K_INT4;
        As[buf][warp_id][row][col] = A_e[(tile_row_local + row) * K + col];
    }
    // --- initial load: B tile (packed int4, WMMA_N/2 bytes per logical row) ---
    const int ldB_bytes = WMMA_N / 2 + PAD;
    for (int i = lane_id; i < PTX_MMA_K_INT4 * (WMMA_N / 2); i += 32) {
        int row = i / (WMMA_N / 2), col = i % (WMMA_N / 2);
        // physical col in B: tile_col_in/2 + col
        Bs[buf][warp_id][row][col] = B_e[row * (N / 2) + (tile_col_in / 2 + col)];
    }
    __syncthreads();

    uint32_t a[4], b0[2], b1[2];
    load_a_m16k64_s8(&As[buf][warp_id][0][0], PTX_MMA_K_INT4 + PAD, a);
    load_b_m8k64_s4 ( Bs[buf][warp_id][0],    ldB_bytes, 0, b0);
    load_b_m8k64_s4 ( Bs[buf][warp_id][0],    ldB_bytes, 8, b1);

    // --- main loop ---
    for (int k_off = PTX_MMA_K_INT4; k_off < K; k_off += PTX_MMA_K_INT4) {
        int next = 1 - buf;

        // cp.async A: int8, 16 bytes = 16 elements per issue
        for (int i = lane_id * 16; i < WMMA_M * PTX_MMA_K_INT4; i += 32 * 16) {
            int row = i / PTX_MMA_K_INT4, col = i % PTX_MMA_K_INT4;
            char*       dst = (char*)&As[next][warp_id][row][col];
            const char* src = (const char*)&A_e[(tile_row_local + row) * K + (k_off + col)];
            unsigned smem_ptr = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(smem_ptr), "l"(src));
        }
        // cp.async B: packed int4, 16 bytes = 32 int4 elements per issue
        for (int i = lane_id * 16; i < PTX_MMA_K_INT4 * (WMMA_N / 2); i += THREADS_PER_WARP * 16) {
            int row = i / (WMMA_N / 2), col = i % (WMMA_N / 2);
            char*       dst = (char*)&Bs[next][warp_id][row][col];
            const char* src = (const char*)&B_e[(k_off + row) * (N / 2) + (tile_col_in / 2 + col)];
            unsigned smem_ptr = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(smem_ptr), "l"(src));
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        // compute on current tile
        ptx_mma_m16n8k64_s8s4(a, b0, D0);
        ptx_mma_m16n8k64_s8s4(a, b1, D1);

        buf = next;
        load_a_m16k64_s8(&As[buf][warp_id][0][0], PTX_MMA_K_INT4 + PAD, a);
        load_b_m8k64_s4 ( Bs[buf][warp_id][0],    ldB_bytes, 0, b0);
        load_b_m8k64_s4 ( Bs[buf][warp_id][0],    ldB_bytes, 8, b1);
    }

    // compute last tile
    ptx_mma_m16n8k64_s8s4(a, b0, D0);
    ptx_mma_m16n8k64_s8s4(a, b1, D1);

    // --- store D0/D1 to global int32 scratch ---
    // identical output layout to int8 PTX version (m16n8 output, 4 regs per thread)
    {
        int row0      = (lane_id / 4) * 2;
        int row1      = row0 + 1;
        int col_left  = lane_id % 4;
        int col_right = lane_id % 4;

        int32_t* base = C_global + (blockIdx.z * M + tile_row_in) * N + tile_col_in;

        base[(row0) * N + col_left]          = D0[0];
        base[(row0) * N + col_left  + 4]     = D0[1];
        base[(row1) * N + col_left]          = D0[2];
        base[(row1) * N + col_left  + 4]     = D0[3];

        base[(row0) * N + 8 + col_right]     = D1[0];
        base[(row0) * N + 8 + col_right + 4] = D1[1];
        base[(row1) * N + 8 + col_right]     = D1[2];
        base[(row1) * N + 8 + col_right + 4] = D1[3];
    }
}

// ---------------------------------------------------------------------------
// wmma_db_fp16: router matmul — unchanged
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
// quantize_and_assign_per_expert_inputs: fp16 → int8 (activations stay int8 = W4A8)
// identical to validated int8 PTX version
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void quantize_and_assign_per_expert_inputs(
    const half* __restrict__  input,
    const int*  __restrict__  expert_counts,
    const int*  __restrict__  expert_token_ids,
    int8_t*     __restrict__  per_expert_wmma_inputs,
    float                     scale_input_act,
    int                       CAP
){
    const int batch              = blockIdx.z;
    const half* input_b          = input            + batch * N * d_model;
    const int*  expert_counts_b  = expert_counts    + batch * num_experts;
    const int*  expert_token_ids_b = expert_token_ids + batch * num_experts * CAP;
    int8_t*     per_expert_b     = per_expert_wmma_inputs + batch * num_experts * CAP * d_model;

    const int tid     = threadIdx.x;
    const int warp_id = tid / THREADS_PER_WARP;
    const int lane_id = tid % THREADS_PER_WARP;
    const int row_id  = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row_id >= num_experts * CAP) return;

    const int expert_id = row_id / CAP;
    const int slot      = row_id % CAP;
    const int row_base  = row_id * d_model;

    if (slot < expert_counts_b[expert_id]) {
        const int token_id = expert_token_ids_b[expert_id * CAP + slot];
        if (token_id >= 0 && token_id < N) {
            const int in_base = token_id * d_model;
            for (int col = lane_id; col < d_model; col += THREADS_PER_WARP) {
                float val = __half2float(input_b[in_base + col]);
                per_expert_b[row_base + col] = (int8_t)__float2int_rn(
                    fminf(fmaxf(val / scale_input_act, -128.f), 127.f));
            }
        } else {
            for (int col = lane_id; col < d_model; col += THREADS_PER_WARP)
                per_expert_b[row_base + col] = (int8_t)0;
        }
    } else {
        for (int col = lane_id; col < d_model; col += THREADS_PER_WARP)
            per_expert_b[row_base + col] = (int8_t)0;
    }
}

// ---------------------------------------------------------------------------
// silu_and_requant_int4:
//   dequant int32 up+gate → SiLU → requant to packed int4 (2 per byte)
//   output buffer is uint8_t* at half the element count
//   Two adjacent elements (even/odd col) packed into one byte
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void silu_and_requant_int4(
    const int32_t* __restrict__ up_int32,
    const int32_t* __restrict__ gate_int32,
    uint8_t*       __restrict__ out_int4_packed,  // [num_experts, CAP, 4*d_model/2]
    float scale_input_act,
    float scale_up_w,
    float scale_gate_w,
    float scale_mid_act,
    int   total_elements  // logical element count = num_experts * CAP * 4 * d_model
){
    const int batch         = blockIdx.z;
    const int32_t* up_b     = up_int32          + batch * total_elements;
    const int32_t* gate_b   = gate_int32        + batch * total_elements;
    uint8_t*       out_b    = out_int4_packed   + batch * (total_elements / 2);

    const int block_linear  = blockIdx.y * gridDim.x + blockIdx.x;
    const int global_tid    = block_linear * blockDim.x + threadIdx.x;
    const int global_stride = gridDim.x * gridDim.y * blockDim.x;

    const float dequant_up    = scale_input_act * scale_up_w;
    const float dequant_gate  = scale_input_act * scale_gate_w;
    const float inv_scale_mid = 1.0f / scale_mid_act;

    // process two logical elements per thread → pack into one output byte
    // iterate over pairs: idx = even logical index
    const int total_pairs = total_elements / 2;
    for (int pair_idx = global_tid; pair_idx < total_pairs; pair_idx += global_stride) {
        int idx0 = pair_idx * 2;
        int idx1 = idx0 + 1;

        auto quantize_one = [&](int idx) -> int8_t {
            const float up_f   = (float)up_b[idx]   * dequant_up;
            const float gate_f = (float)gate_b[idx] * dequant_gate;
            const float silu   = gate_f / (1.0f + __expf(-gate_f));
            const float fused  = up_f * silu * inv_scale_mid;
            // clamp to [-7, 7] (signed int4 range, avoid -8 for symmetry)
            return (int8_t)__float2int_rn(fminf(fmaxf(fused, -7.f), 7.f));
        };

        int8_t v0 = quantize_one(idx0);
        int8_t v1 = quantize_one(idx1);

        // pack: low nibble = v0, high nibble = v1
        out_b[pair_idx] = ((uint8_t)(v0 & 0xF)) | (((uint8_t)(v1 & 0xF)) << 4);
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
// combine — identical to int8 PTX version (int32 input, fp32 output)
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void combine(
    const int32_t* __restrict__ input,
    const int*     __restrict__ expert_token_ids,
    const float*   __restrict__ expert_token_weights,
    const int*     __restrict__ expert_counts,
    float*                      final_output,
    float scale_mid_act,
    float scale_down_w,
    int   CAP
){
    const int batch           = blockIdx.z;
    const float dequant       = scale_mid_act * scale_down_w;
    const int rows_per_expert = CAP;

    const int32_t* input_b   = input              + batch * num_experts * rows_per_expert * d_model;
    const int*     tok_ids_b = expert_token_ids   + batch * num_experts * rows_per_expert;
    const float*   tok_w_b   = expert_token_weights + batch * num_experts * rows_per_expert;
    const int*     counts_b  = expert_counts       + batch * num_experts;
    float*         final_b   = final_output        + batch * N * d_model;

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
        float val = (float)input_b[expert_row_base + col] * dequant;
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

    const half*    input                    = args.input;
    const half*    router_weights           = args.router_weights;
    const uint8_t* expert_up_proj_weights   = args.expert_up_proj_weights_int4;
    const uint8_t* expert_gate_proj_weights = args.expert_gate_proj_weights_int4;
    const uint8_t* expert_down_proj_weights = args.expert_down_proj_weights_int4;

    const float scale_up_w      = args.scale_up_w;
    const float scale_gate_w    = args.scale_gate_w;
    const float scale_down_w    = args.scale_down_w;
    const float scale_input_act = args.scale_input_act;
    const float scale_mid_act   = args.scale_mid_act;

    float*     expert_logits           = args.expert_logits;
    int*       selected_expert_indices = args.selected_expert_indices;
    float*     selected_expert_weights = args.selected_expert_weights;
    int*       expert_counts           = args.expert_counts;
    int*       expert_token_ids        = args.expert_token_ids;
    float*     expert_token_weights    = args.expert_token_weights;
    int8_t*    per_expert_wmma_inputs  = args.per_expert_wmma_inputs_int8;  // activations: int8
    uint8_t*   hidden_mlp_int4         = args.hidden_mlp_layer_1_out_int4;  // post-SiLU: packed int4
    float*     final_output            = args.final_output;

    int32_t* up_int32   = reinterpret_cast<int32_t*>(args.hidden_mlp_layer_1_out);
    int32_t* gate_int32 = reinterpret_cast<int32_t*>(args.hidden_mlp_gate_out);
    int32_t* down_int32 = reinterpret_cast<int32_t*>(args.hidden_mlp_layer_2_out);

    int CAP_raw = (int)ceilf((float)N * (float)k / (float)num_experts * capacity_factor);
    int CAP     = ((CAP_raw + WMMA_M - 1) / WMMA_M) * WMMA_M;

    zero_final_output_and_expert_counts(final_output, expert_counts);

    wmma_db_fp16(1.0f, input, router_weights, expert_logits, N, num_experts, d_model);

    __shared__ float max_vals[WARPS_PER_BLOCK * k];
    __shared__ int   max_indices[WARPS_PER_BLOCK * k];
    top_k_gating(expert_logits, selected_expert_indices, selected_expert_weights, max_vals, max_indices);
    __syncthreads();

    build_per_expert_buffers(selected_expert_indices, selected_expert_weights,
                             expert_counts, expert_token_ids, expert_token_weights, CAP);
    __syncthreads();

    clamp_expert_counts(expert_counts, CAP);
    __syncthreads();

    // Stage 4: quantize fp16 → int8 activations (W4A8: activations stay int8)
    quantize_and_assign_per_expert_inputs(input, expert_counts, expert_token_ids,
                                          per_expert_wmma_inputs, scale_input_act, CAP);
    __syncthreads();

    // Stage 5: up_proj + gate_proj W4A8 GEMMs → int32 scratch
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
            wmma_db_int4<true>(per_expert_wmma_inputs, expert_up_proj_weights,
                               up_int32,   M, Nw, K, tile_row, tile_col);
            wmma_db_int4<true>(per_expert_wmma_inputs, expert_gate_proj_weights,
                               gate_int32, M, Nw, K, tile_row, tile_col);
        }
    }
    __syncthreads();

    // Stage 6: fused dequant + SiLU + requant → packed int4
    silu_and_requant_int4(up_int32, gate_int32, hidden_mlp_int4,
                          scale_input_act, scale_up_w, scale_gate_w, scale_mid_act,
                          num_experts * CAP * up_proj_dim * d_model);
    __syncthreads();

    // Stage 7: down_proj W4A8 GEMM (A=packed int4 mid activations, B=packed int4 weights)
    // Note: after SiLU, mid activations are packed int4 → treat as packed int4 for A too
    // down_proj: A=[num_experts,CAP,4*d_model] int4, B=[num_experts,4*d_model,d_model] int4
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
            // Both A (mid activations) and B (down_proj weights) are packed int4
            // Reuse wmma_db_int4 with hidden_mlp_int4 reinterpreted as int8* for A
            // PTX s8xs4 — A pointer cast: packed int4 treated as int8 bytes
            // (each byte holds 2 int4; PTX unpacks nibbles internally for s4 B;
            //  for A the s8 path reads full bytes — so we need A as int8)
            // Solution: down_proj uses a fully symmetric int4xi4 variant
            // For W4A8 consistency: unpack hidden_mlp_int4 to int8 on-the-fly in a
            // dedicated helper, or use wmma_db_int4_sym below.
            // Here we use the asymmetric path: A unpacked to int8 scratch, B packed int4.
            // Unpack hidden_mlp_int4 → per_expert_wmma_inputs (reuse int8 buffer, now free)
            // This unpack happens implicitly: we pass hidden_mlp_int4 recast and let
            // load_a_m16k64_s8 read it as int8 bytes (each byte = 2 nibbles packed).
            // IMPORTANT: this is WRONG for correctness — see note below.
            // Correct approach: unpack mid activations to int8 before down_proj GEMM.
            // We do that here explicitly via a per-thread unpack into per_expert_wmma_inputs.

            // Unpack packed int4 mid activations → int8 (sign-extended) into reused buffer
            // per_expert_wmma_inputs is now free (used in stage 4, no longer needed)
            int8_t* mid_int8 = per_expert_wmma_inputs;  // reuse
            const int total_packed = num_experts * CAP * up_proj_dim * d_model / 2;
            // each thread unpacks 16 bytes of packed int4 → 32 bytes of int8
            for (int i = threadIdx.x; i < total_packed; i += blockDim.x) {
                uint8_t packed = hidden_mlp_int4[i];
                int8_t lo = (int8_t)((packed & 0xF) | ((packed & 0x8) ? 0xF0 : 0)); // sign extend
                int8_t hi = (int8_t)(((packed >> 4) & 0xF) | (((packed >> 4) & 0x8) ? 0xF0 : 0));
                mid_int8[i * 2 + 0] = lo;
                mid_int8[i * 2 + 1] = hi;
            }
            __syncthreads();

            wmma_db_int4<true>(mid_int8, expert_down_proj_weights,
                               down_int32, M, Nd, K, tile_row, tile_col);
        }
    }
    __syncthreads();

    combine(down_int32, expert_token_ids, expert_token_weights, expert_counts,
            final_output, scale_mid_act, scale_down_w, CAP);
}

#include "launcher.h"