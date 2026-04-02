#include "include/config.h"
#include "include/moe_args.h"
#include <mma.h>
using namespace nvcuda;
#include <stdio.h>
#include <assert.h>
#include <cstdint>

#define MOE_KERNEL capacity
#define MOE_USES_CAPACITY 1

// ---------------------------------------------------------------------------
// wmma_db_fp16: router matmul only — stays fp16 (router is not bandwidth bottleneck)
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void wmma_db_fp16(
    float alpha,
    const half* A,
    const half* B,
    float* C,
    int M, int N, int K
){
    assert( (M % WMMA_M == 0) && (N % WMMA_N == 0) && (K % WMMA_K == 0) );
    int batch = blockIdx.z;

    const half* A_batch = A + batch * M * K;
    const half* B_batch = B;
    float*      C_batch = C + batch * M * N;

    int tid        = threadIdx.x;
    int warp_id    = tid / THREADS_PER_WARP;
    int lane_id    = tid % THREADS_PER_WARP;

    int warp_tile_row = warp_id / WARP_TILES_X;
    int warp_tile_col = warp_id % WARP_TILES_X;
    const int tile_row = blockIdx.y * (WMMA_M * WARP_TILES_Y) + warp_tile_row * WMMA_M;
    const int tile_col = blockIdx.x * (WMMA_N * WARP_TILES_X) + warp_tile_col * WMMA_N;
    if (tile_row >= M || tile_col >= N) return;

    __shared__ __align__(16) half As[2][WARPS_PER_BLOCK][WMMA_M][WMMA_K + PAD];
    __shared__ __align__(16) half Bs[2][WARPS_PER_BLOCK][WMMA_K][WMMA_N + PAD];

    wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>                 c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int buf = 0;

    for (int i = lane_id; i < WMMA_M * WMMA_K; i += 32) {
        int row = i / WMMA_K, col = i % WMMA_K;
        As[buf][warp_id][row][col] = A_batch[(tile_row + row) * K + col];
    }
    for (int i = lane_id; i < WMMA_K * WMMA_N; i += 32) {
        int row = i / WMMA_N, col = i % WMMA_N;
        Bs[buf][warp_id][row][col] = B_batch[row * N + (tile_col + col)];
    }
    __syncthreads();

    wmma::load_matrix_sync(a_frag, &As[buf][warp_id][0][0], WMMA_K + PAD);
    wmma::load_matrix_sync(b_frag, &Bs[buf][warp_id][0][0], WMMA_N + PAD);

    for (int k_off = WMMA_K; k_off < K; k_off += WMMA_K) {
        int next = 1 - buf;

        for (int i = lane_id * 8; i < WMMA_M * WMMA_K; i += 32 * 8) {
            int row = i / WMMA_K, col = i % WMMA_K;
            char*       dst      = (char*)&As[next][warp_id][row][col];
            const char* src      = (const char*)&A_batch[(tile_row + row) * K + (k_off + col)];
            unsigned    smem_ptr = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(smem_ptr), "l"(src));
        }
        for (int i = lane_id * 8; i < WMMA_K * WMMA_N; i += THREADS_PER_WARP * 8) {
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
        wmma::load_matrix_sync(a_frag, &As[buf][warp_id][0][0], WMMA_K + PAD);
        wmma::load_matrix_sync(b_frag, &Bs[buf][warp_id][0][0], WMMA_N + PAD);
    }
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    float* c_dst = C_batch + tile_row * N + tile_col;
    for (int i = 0; i < c_frag.num_elements; ++i) c_frag.x[i] = alpha * c_frag.x[i];
    wmma::store_matrix_sync(c_dst, c_frag, N, wmma::mem_row_major);
}

// ---------------------------------------------------------------------------
// wmma_db_int8: int8 x int8 -> int32 double-buffered GEMM
//   A: int8 [M, K]   B: int8 [K, N]   C_out: int32 smem tile (never goes to global)
//   The caller receives the int32 accumulator fragment directly for fused post-processing.
//   Returns via c_frag (passed by reference).
// ---------------------------------------------------------------------------
template<bool calculatePerExpert>
static __device__ __forceinline__ void wmma_db_int8(
    const int8_t* A,
    const int8_t* B,
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t>& c_frag,
    int M, int N, int K,
    int tile_row_in,   // pre-computed tile row (absolute, into A)
    int tile_col_in    // pre-computed tile col (absolute, into B)
){
    // NOTE: K must be multiple of WMMA_K=32 for int8
    int batch   = blockIdx.z;
    int tid     = threadIdx.x;
    int warp_id = tid / THREADS_PER_WARP;
    int lane_id = tid % THREADS_PER_WARP;

    const int8_t* A_batch = A + batch * M * K;
    const int8_t* B_batch = B; // weights not batched

    const int8_t* A_e;
    const int8_t* B_e;
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

    // int8 shared memory tiles — half the smem vs fp16
    __shared__ __align__(16) int8_t As8[2][WARPS_PER_BLOCK][WMMA_M][WMMA_K + PAD];
    __shared__ __align__(16) int8_t Bs8[2][WARPS_PER_BLOCK][WMMA_K][WMMA_N + PAD];

    wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> b_frag;
    wmma::fill_fragment(c_frag, 0);

    int buf = 0;

    // Initial load (scalar, no cp.async)
    for (int i = lane_id; i < WMMA_M * WMMA_K; i += 32) {
        int row = i / WMMA_K, col = i % WMMA_K;
        As8[buf][warp_id][row][col] = A_e[(tile_row_local + row) * K + col];
    }
    for (int i = lane_id; i < WMMA_K * WMMA_N; i += 32) {
        int row = i / WMMA_N, col = i % WMMA_N;
        Bs8[buf][warp_id][row][col] = B_e[row * N + (tile_col_in + col)];
    }
    __syncthreads();

    wmma::load_matrix_sync(a_frag, &As8[buf][warp_id][0][0], WMMA_K + PAD);
    wmma::load_matrix_sync(b_frag, &Bs8[buf][warp_id][0][0], WMMA_N + PAD);

    for (int k_off = WMMA_K; k_off < K; k_off += WMMA_K) {
        int next = 1 - buf;

        // cp.async next int8 tiles — 16 bytes per issue, int8 so 16 elements per thread
        for (int i = lane_id * 16; i < WMMA_M * WMMA_K; i += 32 * 16) {
            int row = i / WMMA_K, col = i % WMMA_K;
            char*       dst = (char*)&As8[next][warp_id][row][col];
            const char* src = (const char*)&A_e[(tile_row_local + row) * K + (k_off + col)];
            unsigned smem_ptr = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(smem_ptr), "l"(src));
        }
        for (int i = lane_id * 16; i < WMMA_K * WMMA_N; i += THREADS_PER_WARP * 16) {
            int row = i / WMMA_N, col = i % WMMA_N;
            char*       dst = (char*)&Bs8[next][warp_id][row][col];
            const char* src = (const char*)&B_e[(k_off + row) * N + (tile_col_in + col)];
            unsigned smem_ptr = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(smem_ptr), "l"(src));
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        buf = next;
        wmma::load_matrix_sync(a_frag, &As8[buf][warp_id][0][0], WMMA_K + PAD);
        wmma::load_matrix_sync(b_frag, &Bs8[buf][warp_id][0][0], WMMA_N + PAD);
    }
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    // c_frag (int32) returned to caller — no global store here
}

// ---------------------------------------------------------------------------
// quantize_input_to_int8:
//   Converts fp16 input activations → int8 per_expert_wmma_inputs in one pass.
//   Replaces assign_per_expert_wmma_inputs + does quantization.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void quantize_and_assign_per_expert_inputs(
    const half* __restrict__  input,
    const int*  __restrict__  expert_counts,
    const int*  __restrict__  expert_token_ids,
    int8_t*     __restrict__  per_expert_wmma_inputs,
    float                     scale_input_act,
    int                       CAP
){
    const int batch = blockIdx.z;
    const half* input_b              = input + batch * N * d_model;
    const int*  expert_counts_b      = expert_counts + batch * num_experts;
    const int*  expert_token_ids_b   = expert_token_ids + batch * num_experts * CAP;
    int8_t*     per_expert_b         = per_expert_wmma_inputs + batch * num_experts * CAP * d_model;

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
// silu_and_requant:
//   Fused: dequant int32 up+gate results → SiLU(up)*gate → requant to int8.
//   Replaces fp32_to_fp16. Writes directly to hidden_mlp_layer_1_out_int8.
//   Called after up_proj and gate_proj wmma_db_int8 results are stored as int32
//   into two temporary smem tiles — but since we can't hold two full int32 tiles
//   across the whole [num_experts, CAP, 4*d_model] space in smem, we write
//   the int32 results to global as int32 (reusing the now-freed fp32 buffer
//   pointers, cast), then fuse here.
//   Strategy: pass int32 global scratch for up and gate, fuse in this kernel.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void silu_and_requant(
    const int32_t* __restrict__ up_int32,       // [num_experts, CAP, 4*d_model]
    const int32_t* __restrict__ gate_int32,     // [num_experts, CAP, 4*d_model]
    int8_t*        __restrict__ out_int8,        // [num_experts, CAP, 4*d_model]
    float scale_input_act,
    float scale_up_w,
    float scale_gate_w,
    float scale_mid_act,
    int   total_size
){
    const int batch          = blockIdx.z;
    const int up_size_batch  = total_size; // per batch
    const int32_t* up_b      = up_int32   + batch * up_size_batch;
    const int32_t* gate_b    = gate_int32 + batch * up_size_batch;
    int8_t*        out_b     = out_int8   + batch * up_size_batch;

    const int block_linear  = blockIdx.y * gridDim.x + blockIdx.x;
    const int global_tid    = block_linear * blockDim.x + threadIdx.x;
    const int global_stride = gridDim.x * gridDim.y * blockDim.x;

    const float dequant_up   = scale_input_act * scale_up_w;
    const float dequant_gate = scale_input_act * scale_gate_w;
    const float inv_scale_mid = 1.0f / scale_mid_act;

    for (int idx = global_tid; idx < total_size; idx += global_stride) {
        const float up_f   = (float)up_b[idx]   * dequant_up;
        const float gate_f = (float)gate_b[idx] * dequant_gate;
        const float silu   = gate_f / (1.0f + __expf(-gate_f));  // SiLU on gate
        const float fused  = up_f * silu;
        out_b[idx] = (int8_t)__float2int_rn(fminf(fmaxf(fused * inv_scale_mid, -128.f), 127.f));
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
    const float* logits_batch                 = logits + batch * N * num_experts;
    int*         selected_expert_indices_b    = selected_expert_indices + batch * N * k;
    float*       selected_expert_weights_b    = selected_expert_weights + batch * N * k;

    const int tid     = threadIdx.x;
    const int warp_id = tid / THREADS_PER_WARP;
    const int lane_id = tid % THREADS_PER_WARP;
    const int token_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (token_id >= N) return;

    float* warp_max_vals = max_vals  + warp_id * k;
    int*   warp_max_idxs = max_idxs  + warp_id * k;
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
            selected_expert_indices_b[token_id * k + l]  = warp_max_idxs[l];
            selected_expert_weights_b[token_id * k + l]  = expf(warp_max_vals[l] - max_val) / (sum_of_exps + 1e-10f);
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
    const int batch = blockIdx.z;
    const int* sel_idx_b  = selected_expert_indices  + batch * N * k;
    const float* sel_w_b  = selected_expert_weights  + batch * N * k;
    int*   counts_b       = expert_counts            + batch * num_experts;
    int*   tok_ids_b      = expert_token_ids         + batch * num_experts * CAP;
    float* tok_w_b        = expert_token_weights     + batch * num_experts * CAP;

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
static __device__ __forceinline__ void clamp_expert_counts(int* __restrict__ expert_counts, int CAP){
    const int batch  = blockIdx.z;
    int* counts_b    = expert_counts + batch * num_experts;
    const int global_tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_stride = gridDim.x  * blockDim.x;
    for (int idx = global_tid; idx < num_experts; idx += global_stride) {
        if (counts_b[idx] > CAP) counts_b[idx] = CAP;
    }
}

// ---------------------------------------------------------------------------
// combine: dequantize int32 down_proj result and accumulate into final output.
//   scale applied here — no requant needed (final output is fp32).
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void combine(
    const int32_t* __restrict__ input,              // int32 down_proj output [num_experts, CAP, d_model]
    const int*     __restrict__ expert_token_ids,
    const float*   __restrict__ expert_token_weights,
    const int*     __restrict__ expert_counts,
    float*                      final_output,
    float scale_mid_act,
    float scale_down_w,
    int   CAP
){
    const int batch = blockIdx.z;
    const float dequant = scale_mid_act * scale_down_w;
    const int rows_per_expert = CAP;

    const int32_t* input_b       = input              + batch * num_experts * rows_per_expert * d_model;
    const int*     tok_ids_b     = expert_token_ids   + batch * num_experts * rows_per_expert;
    const float*   tok_w_b       = expert_token_weights + batch * num_experts * rows_per_expert;
    const int*     counts_b      = expert_counts       + batch * num_experts;
    float*         final_b       = final_output        + batch * N * d_model;

    const int tid     = threadIdx.x;
    const int warp_id = tid / THREADS_PER_WARP;
    const int lane_id = tid % THREADS_PER_WARP;
    const int row_id  = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row_id >= num_experts * rows_per_expert) return;

    const int expert_id = row_id / rows_per_expert;
    const int slot      = row_id % rows_per_expert;
    if (slot >= counts_b[expert_id]) return;

    const int token_id    = tok_ids_b[expert_id * rows_per_expert + slot];
    if (token_id < 0 || token_id >= N) return;

    const float route_weight    = tok_w_b[expert_id * rows_per_expert + slot];
    const int   expert_row_base = row_id  * d_model;
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

    const half*    input                   = args.input;
    const half*    router_weights          = args.router_weights;
    const int8_t*  expert_up_proj_weights  = args.expert_up_proj_weights_int8;
    const int8_t*  expert_gate_proj_weights= args.expert_gate_proj_weights_int8;
    const int8_t*  expert_down_proj_weights= args.expert_down_proj_weights_int8;

    const float scale_up_w        = args.scale_up_w;
    const float scale_gate_w      = args.scale_gate_w;
    const float scale_down_w      = args.scale_down_w;
    const float scale_input_act   = args.scale_input_act;
    const float scale_mid_act     = args.scale_mid_act;

    float*     expert_logits          = args.expert_logits;
    int*       selected_expert_indices= args.selected_expert_indices;
    float*     selected_expert_weights= args.selected_expert_weights;
    int*       expert_counts          = args.expert_counts;
    int*       expert_token_ids       = args.expert_token_ids;
    float*     expert_token_weights   = args.expert_token_weights;
    int8_t*    per_expert_wmma_inputs = args.per_expert_wmma_inputs_int8;
    int8_t*    hidden_mlp_int8        = args.hidden_mlp_layer_1_out_int8; // post SiLU, int8
    float*     final_output           = args.final_output;

    // Scratch int32 buffers: reuse host-allocated storage.
    // Host must allocate two int32 buffers of size [num_batches, num_experts, CAP, 4*d_model]
    // and one of size [num_batches, num_experts, CAP, d_model] and pass via args (see note).
    // For clarity cast from the host-side int32* fields (add to MoEArgs if needed):
    int32_t* up_int32   = reinterpret_cast<int32_t*>(args.hidden_mlp_layer_1_out);  // repurposed
    int32_t* gate_int32 = reinterpret_cast<int32_t*>(args.hidden_mlp_gate_out);     // repurposed
    int32_t* down_int32 = reinterpret_cast<int32_t*>(args.hidden_mlp_layer_2_out);  // repurposed

    int CAP_raw = (int)ceilf((float)N * (float)k / (float)num_experts * capacity_factor);
    int CAP     = ((CAP_raw + WMMA_M - 1) / WMMA_M) * WMMA_M;

    // --- Stage 0: zero buffers ---
    zero_final_output_and_expert_counts(final_output, expert_counts);

    // --- Stage 1: router GEMM (fp16, small matrix, not bandwidth bottleneck) ---
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

    // --- Stage 4: quantize fp16 input activations → int8 per-expert tiles ---
    quantize_and_assign_per_expert_inputs(input, expert_counts, expert_token_ids,
                                          per_expert_wmma_inputs, scale_input_act, CAP);
    __syncthreads();

    // --- Stage 5: up_proj and gate_proj int8 GEMMs → int32 scratch ---
    {
        int tid      = threadIdx.x;
        int warp_id  = tid / THREADS_PER_WARP;
        int warp_tile_row = warp_id / WARP_TILES_X;
        int warp_tile_col = warp_id % WARP_TILES_X;
        const int M  = num_experts * CAP;
        const int Nw = up_proj_dim * d_model;
        const int K  = d_model;
        const int tile_row = blockIdx.y * (WMMA_M * WARP_TILES_Y) + warp_tile_row * WMMA_M;
        const int tile_col = blockIdx.x * (WMMA_N * WARP_TILES_X) + warp_tile_col * WMMA_N;

        if (tile_row < M && tile_col < Nw) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> up_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> gate_frag;

            wmma_db_int8<true>(per_expert_wmma_inputs, expert_up_proj_weights,
                               up_frag,   M, Nw, K, tile_row, tile_col);
            wmma_db_int8<true>(per_expert_wmma_inputs, expert_gate_proj_weights,
                               gate_frag, M, Nw, K, tile_row, tile_col);

            // Store int32 fragments to global scratch (int32, same footprint as old fp32 buffers)
            int32_t* up_dst   = up_int32   + (blockIdx.z * M + tile_row) * Nw + tile_col;
            int32_t* gate_dst = gate_int32 + (blockIdx.z * M + tile_row) * Nw + tile_col;
            wmma::store_matrix_sync(up_dst,   up_frag,   Nw, wmma::mem_row_major);
            wmma::store_matrix_sync(gate_dst, gate_frag, Nw, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // --- Stage 6: fused dequant + SiLU + requant → int8 ---
    silu_and_requant(up_int32, gate_int32, hidden_mlp_int8,
                     scale_input_act, scale_up_w, scale_gate_w, scale_mid_act,
                     num_experts * CAP * up_proj_dim * d_model);
    __syncthreads();

    // --- Stage 7: down_proj int8 GEMM → int32 scratch ---
    {
        int tid      = threadIdx.x;
        int warp_id  = tid / THREADS_PER_WARP;
        int warp_tile_row = warp_id / WARP_TILES_X;
        int warp_tile_col = warp_id % WARP_TILES_X;
        const int M  = num_experts * CAP;
        const int Nd = d_model;
        const int K  = up_proj_dim * d_model;
        const int tile_row = blockIdx.y * (WMMA_M * WARP_TILES_Y) + warp_tile_row * WMMA_M;
        const int tile_col = blockIdx.x * (WMMA_N * WARP_TILES_X) + warp_tile_col * WMMA_N;

        if (tile_row < M && tile_col < Nd) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> down_frag;
            wmma_db_int8<true>(hidden_mlp_int8, expert_down_proj_weights,
                               down_frag, M, Nd, K, tile_row, tile_col);
            int32_t* down_dst = down_int32 + (blockIdx.z * M + tile_row) * Nd + tile_col;
            wmma::store_matrix_sync(down_dst, down_frag, Nd, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // --- Stage 8: dequant + weighted combine → fp32 final output ---
    combine(down_int32, expert_token_ids, expert_token_weights, expert_counts,
            final_output, scale_mid_act, scale_down_w, CAP);
}

#include "launcher.h"