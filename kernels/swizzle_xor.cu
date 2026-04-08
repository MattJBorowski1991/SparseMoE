#include "include/config.h"
#include "include/moe_args.h"
#include <mma.h>                //Enables Tensor Core instructions
using namespace nvcuda;         // Brings wmma:: into scope
#include <stdio.h>
#include <assert.h>

#define MOE_KERNEL swizzle_xor
// capacity variant requires capacity-aware allocations
#define MOE_USES_CAPACITY 1
// Swizzle at 16B chunk granularity to preserve cp.async destination alignment.
#define SWIZZLE_CHUNK_MASK 0x1

template<bool calculatePerExpert>
static __device__ __forceinline__ void wmma_db(
    float alpha,
    const half* A,
    const half* B,
    float* C,
    int M, int N, int K
)
{   
    assert( (M % WMMA_M == 0) && (N % WMMA_N == 0) && (K % WMMA_K == 0) );
    int batch = blockIdx.z;

    const half* A_batch = A + batch * M * K;
    const half* B_batch = B; //we don't batch B as here the trained weights come in for all wmma we use: both in router layer and then MLPs
    float* C_batch = C + batch * M * N;

    int tid = threadIdx.x;
    int warp_id = tid / THREADS_PER_WARP;
    int lane_id = tid % THREADS_PER_WARP;

    const half* A_e;
    const half* B_e;
    float* C_e;

    int warp_tile_row = warp_id / WARP_TILES_X;
    int warp_tile_col = warp_id % WARP_TILES_X;
    const int tile_row = blockIdx.y * (WMMA_M * WARP_TILES_Y) + warp_tile_row * WMMA_M;
    const int tile_col = blockIdx.x * (WMMA_N * WARP_TILES_X) + warp_tile_col * WMMA_N;
    if (tile_row >= M || tile_col >= N) return;

    int tile_row_local = tile_row;

    if constexpr (calculatePerExpert){ // mini wmma's per-expert after the routing layer
        const int row_id = tile_row;
        const int rows_per_expert = M / num_experts;
        const int expert_id = row_id / rows_per_expert;
        tile_row_local = row_id % rows_per_expert;

        A_e = A_batch + expert_id * rows_per_expert * K;
        B_e = B_batch + expert_id * K * N;
        C_e = C_batch + expert_id * rows_per_expert * N;
    }else{
        A_e = A_batch;
        B_e = B_batch;
        C_e = C_batch;
    }

    //********DOUBLE BUFFER START ********

    // Per-warp double buffers in shared memory
    __shared__ __align__(16) half As[2][WARPS_PER_BLOCK][WMMA_M][WMMA_K + PAD];
    __shared__ __align__(16) half Bs[2][WARPS_PER_BLOCK][WMMA_K][WMMA_N + PAD];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int compute_buf = 0;
    int stage_buf = 1;

    // Initial tile load: raw cooperative loads (no cp.async)
    for (int i = lane_id; i < WMMA_M * WMMA_K; i += 32) {
        int row = i / WMMA_K;
        int col = i % WMMA_K;
        As[compute_buf][warp_id][row][col] = A_e[(tile_row_local + row) * K + col];
    }
    for (int i = lane_id; i < WMMA_K * WMMA_N; i += 32) {
        int row = i / WMMA_N;
        int col = i % WMMA_N;
        Bs[compute_buf][warp_id][row][col] = B_e[row * N + (tile_col + col)];
    }
    __syncthreads(); 

    wmma::load_matrix_sync(a_frag, &As[compute_buf][warp_id][0][0], WMMA_K + PAD);
    wmma::load_matrix_sync(b_frag, &Bs[compute_buf][warp_id][0][0], WMMA_N + PAD);

    // Main loop: overlap load (to/from buffers) with compute (in fragments)
    for (int k = WMMA_K; k < K; k += WMMA_K){

        // Load next tile into the next buffer
        for (int i = lane_id * 8; i < WMMA_M * WMMA_K; i += 32 * 8) {
            int row = i / WMMA_K;
            int col = i % WMMA_K;
            int col_chunk = col >> 3; // 8 half = 16B chunk
            int col_chunk_swz = col_chunk ^ (row & SWIZZLE_CHUNK_MASK);
            int col_swz = col_chunk_swz << 3;

            char* dst = (char*)&As[stage_buf][warp_id][row][col_swz];
            const char* src = (const char*)&A_e[(tile_row_local + row) * K + (k + col)];

            unsigned smem_addr = __cvta_generic_to_shared(dst);  // 32-bit shared addr
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                        :: "r"(smem_addr), "l"(src));
        }

        for (int i = lane_id * 8; i < WMMA_K * WMMA_N; i += THREADS_PER_WARP * 8) {
            int row = i / WMMA_N;
            int col = i % WMMA_N;
            int col_chunk = col >> 3; // 8 half = 16B chunk
            int col_chunk_swz = col_chunk ^ (row & SWIZZLE_CHUNK_MASK);
            int col_swz = col_chunk_swz << 3;

            char* dst = (char*)&Bs[stage_buf][warp_id][row][col_swz];
            const char* src = (const char*)&B_e[(k + row) * N + (tile_col + col)];

            unsigned smem_addr = __cvta_generic_to_shared(dst);
            // .cg bypasses L1 for weights (too large to benefit from L1 caching)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                        :: "r"(smem_addr), "l"(src));
        }

        asm volatile("cp.async.commit_group;");
        // No __syncthreads() needed: each warp has its own buffer slot (warp_id-indexed)
        // and cp.async.wait_group 0 already fences this warp's async copies.

        // compute current tile
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        asm volatile("cp.async.wait_group 0;");

        // Unswizzle staged tiles back to linear layout expected by wmma::load_matrix_sync.
        for (int i = lane_id; i < WMMA_M * WMMA_K; i += THREADS_PER_WARP) {
            int row = i / WMMA_K;
            int col = i % WMMA_K;
            int col_chunk = col >> 3;
            int col_chunk_swz = col_chunk ^ (row & SWIZZLE_CHUNK_MASK);
            int col_swz = (col_chunk_swz << 3) | (col & 0x7);
            As[compute_buf][warp_id][row][col] = As[stage_buf][warp_id][row][col_swz];
        }
        for (int i = lane_id; i < WMMA_K * WMMA_N; i += THREADS_PER_WARP) {
            int row = i / WMMA_N;
            int col = i % WMMA_N;
            int col_chunk = col >> 3;
            int col_chunk_swz = col_chunk ^ (row & SWIZZLE_CHUNK_MASK);
            int col_swz = (col_chunk_swz << 3) | (col & 0x7);
            Bs[compute_buf][warp_id][row][col] = Bs[stage_buf][warp_id][row][col_swz];
        }

        wmma::load_matrix_sync(a_frag, &As[compute_buf][warp_id][0][0], WMMA_K + PAD);
        wmma::load_matrix_sync(b_frag, &Bs[compute_buf][warp_id][0][0], WMMA_N + PAD);
    }

    //compute last tile 
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    //********DOUBLE BUFFER END ********

    // Apply alpha and beta*C
    float* c_dst = C_e + tile_row_local * N + tile_col;
    
    for (int i = 0; i < c_frag.num_elements; ++i) c_frag.x[i] = alpha * c_frag.x[i];
    
    wmma::store_matrix_sync(c_dst, c_frag, N, wmma::mem_row_major);
}

static __device__ __forceinline__ void fp32_to_fp16(
    const float* __restrict__ up_input,     // hidden_mlp_layer_1_out
    const float* __restrict__ gate_input,   // hidden_mlp_gate_out
    half* __restrict__ output,              // hidden_mlp_layer_1_out_fp16
    int total_size                          // num_experts * ( CAP * (up_proj_dim * d_model) )
){  
    const int batch = blockIdx.z;

    int CAP_raw = (int)ceilf((float)N * (float)k / (float)num_experts * capacity_factor);
    int CAP = ((CAP_raw + WMMA_M - 1) / WMMA_M) * WMMA_M;

    const float* up_input_b = up_input + batch * num_experts * CAP * (up_proj_dim * d_model);
    const float* gate_input_b = gate_input + batch * num_experts * CAP * (up_proj_dim * d_model);
    half* output_b = output + batch *  num_experts * CAP * (up_proj_dim * d_model);

    const int block_linear = blockIdx.y * gridDim.x + blockIdx.x;
    const int global_tid = block_linear * blockDim.x + threadIdx.x;
    const int global_stride = gridDim.x * gridDim.y * blockDim.x;

    for (int idx = global_tid; idx < total_size; idx += global_stride) {
        const float gate = gate_input_b[idx];
        const float silu_gate = gate / (1.0f + __expf(-gate));
        output_b[idx] = __float2half(up_input_b[idx] * silu_gate);
    }
}


static __device__ __forceinline__ void top_k_gating(
    const float* logits,                // [num_tokens, num_experts] = [num_batches * N, num_experts]
    int* selected_expert_indices,        // [num_tokens, k]
    float* selected_expert_weights      // [num_tokens, k]
){
    int batch = blockIdx.z;

    const float* logits_batch = logits + batch * N * num_experts; // N * num_experts
    int* selected_expert_indices_batch = selected_expert_indices + batch * N * k;
    float* selected_expert_weights_batch = selected_expert_weights + batch * N * k;

    const int tid = threadIdx.x;

    const int warp_id = tid / THREADS_PER_WARP;
    const int lane_id = tid % THREADS_PER_WARP;

    const int token_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (token_id >= N) return;

    const float* logits_row = logits_batch + token_id * num_experts;

    if (lane_id == 0) {
        float warp_max_vals[k];
        int warp_max_idxs[k];

        #pragma unroll
        for (int i = 0; i < k; ++i) {
            warp_max_vals[i] = -1e20f;
            warp_max_idxs[i] = -1;
        }

        #pragma unroll
        for (int logit_id = 0; logit_id < num_experts; ++logit_id) {
            float val = logits_row[logit_id];
            if (val > warp_max_vals[k - 1]) {
                warp_max_vals[k - 1] = val;
                warp_max_idxs[k - 1] = logit_id;

                #pragma unroll
                for (int i = k - 1; i > 0 && warp_max_vals[i] > warp_max_vals[i - 1]; --i) {
                    float tmp_v = warp_max_vals[i - 1];
                    warp_max_vals[i - 1] = warp_max_vals[i];
                    warp_max_vals[i] = tmp_v;

                    int tmp_i = warp_max_idxs[i - 1];
                    warp_max_idxs[i - 1] = warp_max_idxs[i];
                    warp_max_idxs[i] = tmp_i;
                }
            }
        }

        float max_val = warp_max_vals[0];
        float exp_vals[k];
        float sum_of_exps = 0.0f;

        #pragma unroll
        for (int l = 0; l < k; ++l) {
            // Compute each exponential once and reuse it for normalization.
            float e = __expf(warp_max_vals[l] - max_val);
            exp_vals[l] = e;
            sum_of_exps += e;
        }

        float inv_sum = __fdividef(1.0f, sum_of_exps + 1e-10f);
        #pragma unroll
        for (int l = 0; l < k; ++l) {
            selected_expert_indices_batch[token_id * k + l] = warp_max_idxs[l];
            selected_expert_weights_batch[token_id * k + l] = exp_vals[l] * inv_sum;
        }
    }

}

// // after the kernel below is executed:
// 1. m_e: expert_counts[e] gives the number m_e
// 2. [m_e, d_model]:   per_expert_wmma_inputs from indices (base) to (base + m_e * d_model) 
//                      give [m_e, d_model] for that expert, where base = e * N * d_model
// 
static __device__ __forceinline__ void build_per_expert_buffers(
    const int* __restrict__ selected_expert_indices,         // [N, k]
    const float* __restrict__ selected_expert_weights,       // [N, k]
    int* __restrict__ expert_counts,                 // [num_experts]
    int* __restrict__ expert_token_ids,              // [num_experts, max_tokens_per_expert=N]
    float* __restrict__ expert_token_weights         // [num_experts, max_tokens_per_expert=N]
){

    const int batch = blockIdx.z;

    int CAP_raw = (int)ceilf((float)N * (float)k / (float)num_experts * capacity_factor);
    int CAP = ((CAP_raw + WMMA_M - 1) / WMMA_M) * WMMA_M;

    const int* selected_expert_indices_b = selected_expert_indices + batch * N * k;
    const float* selected_expert_weights_b = selected_expert_weights + batch * N * k;
    int* expert_counts_b = expert_counts + batch * num_experts;
    int* expert_token_ids_b = expert_token_ids + batch * num_experts * CAP;
    float* expert_token_weights_b = expert_token_weights + batch * num_experts * CAP;

    const int tid = threadIdx.x;
    const int warp_id = tid / THREADS_PER_WARP;
    const int lane_id = tid % THREADS_PER_WARP;

    const int warp_linear = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int warp_stride = gridDim.x * WARPS_PER_BLOCK;
    const int total_routes = N * k;

    for (int route_id = warp_linear; route_id < total_routes; route_id += warp_stride) {
        if (lane_id == 0) {
            const int token_id = route_id / k;
            const int expert_id = selected_expert_indices_b[route_id];

            if (expert_id >= 0 && expert_id < num_experts) {
                const int slot = atomicAdd(&expert_counts_b[expert_id], 1);

                if (slot < CAP) { // drop overflow policy 
                    const int out_idx = expert_id * CAP + slot;
                    expert_token_ids_b[out_idx] = token_id;
                    expert_token_weights_b[out_idx] = selected_expert_weights_b[route_id];
                }
            }
        }
    }

}


static __device__ __forceinline__ void clamp_expert_counts(
    int* __restrict__ expert_counts
){
    const int batch = blockIdx.z;

    int CAP_raw = (int)ceilf((float)N * (float)k / (float)num_experts * capacity_factor);
    int CAP = ((CAP_raw + WMMA_M - 1) / WMMA_M) * WMMA_M;

    int* expert_counts_b = expert_counts + batch * num_experts;

    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_stride = gridDim.x * blockDim.x;

    for (int idx = global_tid; idx < num_experts; idx += global_stride) {
        int v = expert_counts_b[idx];
        if (v > CAP) expert_counts_b[idx] = CAP;
    }
}


static __device__ __forceinline__ void assign_per_expert_wmma_inputs(
    const half* __restrict__ input,                  // [N, d_model]
    const int* __restrict__ expert_counts,           // [num_experts]
    const int* __restrict__ expert_token_ids,        // [num_experts, N]
    half* __restrict__ per_expert_wmma_inputs        // [num_experts, N, d_model]
){  
    const int batch = blockIdx.z;

    const half* input_b = input + batch * N * d_model;
    const int* expert_counts_b = expert_counts + batch * num_experts;
    int CAP_raw = (int)ceilf((float)N * (float)k / (float)num_experts * capacity_factor);
    int CAP = ((CAP_raw + WMMA_M - 1) / WMMA_M) * WMMA_M;

    const int* expert_token_ids_b = expert_token_ids + batch * num_experts * CAP;
    half* per_expert_wmma_inputs_b = per_expert_wmma_inputs + batch * num_experts * CAP * d_model;

    const int tid = threadIdx.x;
    const int warp_id = tid / THREADS_PER_WARP;
    const int lane_id = tid % THREADS_PER_WARP;


    const int row_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int total_rows = num_experts * CAP;
    if (row_id >= total_rows) return;

    const int expert_id = row_id / CAP;
    const int slot = row_id % CAP;
    const int row_base = row_id * d_model;
    int elems_per_batch = num_experts * CAP * d_model;
    int token_ids_per_batch = num_experts * CAP;

    if (slot < expert_counts_b[expert_id]) {
        int token_idx = expert_id * CAP + slot;
        if (token_idx < 0 || token_idx >= token_ids_per_batch) {
            return;
        }

        const int token_id = expert_token_ids_b[token_idx];
        if (token_id >= 0 && token_id < N) {
            const int in_base = token_id * d_model;
            for (int col = lane_id; col < d_model; col += THREADS_PER_WARP) {
                int dst_idx = row_base + col;
                if (dst_idx < 0 || dst_idx >= elems_per_batch) {
                    return;
                }
                per_expert_wmma_inputs_b[dst_idx] = input_b[in_base + col];
            }
        } else {
            for (int col = lane_id; col < d_model; col += THREADS_PER_WARP) {
                int dst_idx = row_base + col;
                if (dst_idx < 0 || dst_idx >= elems_per_batch) {
                    return;
                }
                per_expert_wmma_inputs_b[dst_idx] = __float2half(0.0f);
            }
        }
    } else {
        for (int col = lane_id; col < d_model; col += THREADS_PER_WARP) {
            int dst_idx = row_base + col;
            if (dst_idx < 0 || dst_idx >= elems_per_batch) {
                return;
            }
            per_expert_wmma_inputs_b[dst_idx] = __float2half(0.0f);
        }
    }
}


static __device__ __forceinline__ void combine(
    const float* __restrict__ input,                         // hidden_mlp_layer_2_out [num_experts, N, d_model]
    const int* __restrict__ expert_token_ids,                // [num_experts, N]
    const float* __restrict__ expert_token_weights,          // [num_experts, N]
    const int* __restrict__ expert_counts,                    // [num_experts]

    float* final_output                        // [N, d_model]
){ 
    const int batch = blockIdx.z;
    int CAP_raw = (int)ceilf((float)N * (float)k / (float)num_experts * capacity_factor);
    int CAP = ((CAP_raw + WMMA_M - 1) / WMMA_M) * WMMA_M;
    const int rows_per_expert = CAP;
    const int elems_per_expert = rows_per_expert * d_model;

    const float* input_b = input + batch * num_experts * elems_per_expert;
    const int* expert_token_ids_b = expert_token_ids + batch * num_experts * rows_per_expert;
    const float* expert_token_weights_b = expert_token_weights + batch * num_experts * rows_per_expert;
    const int* expert_counts_b = expert_counts + batch * num_experts;
    float* final_output_b = final_output + batch * N * d_model;

    const int tid = threadIdx.x;
    const int warp_id = tid / THREADS_PER_WARP;
    const int lane_id = tid % THREADS_PER_WARP;
    const int row_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    const int total_rows = num_experts * rows_per_expert;
    if (row_id >= total_rows) return;

    const int expert_id = row_id / rows_per_expert;
    const int slot = row_id % rows_per_expert;

    if (slot >= expert_counts_b[expert_id]) return;

    const int token_id = expert_token_ids_b[expert_id * rows_per_expert + slot];
    if (token_id < 0 || token_id >= N) return;

    const float route_weight = expert_token_weights_b[expert_id * rows_per_expert + slot];
    const int expert_row_base = row_id * d_model;
    const int token_row_base = token_id * d_model;

    for (int col = lane_id; col < d_model; col += THREADS_PER_WARP) {
        atomicAdd(&final_output_b[token_row_base + col], route_weight * input_b[expert_row_base + col]);
    }
}

static __device__ __forceinline__ void zero_final_output_and_expert_counts(
    float* __restrict__ final_output,    // [N, d_model]
    int* __restrict__ expert_counts      // [num_experts]
){
    const int batch = blockIdx.z;
    if (blockIdx.y != 0) return;

    float* final_output_b = final_output + batch * N * d_model;
    int* expert_counts_b = expert_counts + batch * num_experts;

    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_stride = gridDim.x * blockDim.x;

    for (int idx = global_tid; idx < num_experts; idx += global_stride) {
        expert_counts_b[idx] = 0;
    }

    const int output_size = N * d_model;
    for (int idx = global_tid; idx < output_size; idx += global_stride) {
        final_output_b[idx] = 0.0f;
    }
}



__global__ void swizzle_xor(MoEArgs args){

    // **** list of kernel arguments: start ****
    // (provided via struct MoEArgs from include/moe_args.h for reusability)

    const half* __restrict__ input = args.input;                // [N, d_model]

    // model weights
    const half* __restrict__ router_weights = args.router_weights;                     // [d_model, num_experts]
    const half* __restrict__ expert_up_proj_weights = args.expert_up_proj_weights;     // [num_experts, d_model, 4 * d_model]
    const half* __restrict__ expert_gate_proj_weights = args.expert_gate_proj_weights; // [num_experts, d_model, 4 * d_model]
    const half* __restrict__ expert_down_proj_weights = args.expert_down_proj_weights; // [num_experts, 4 * d_model, d_model]

    //intermediate buffers, pre-allocated by host
    float* __restrict__ expert_logits = args.expert_logits;                            // [N, num_experts] = output of the routing layer
    int* __restrict__ selected_expert_indices = args.selected_expert_indices;          // [N, k] = each token's selected experts
    float* __restrict__ selected_expert_weights = args.selected_expert_weights;        // [N, k] = weights of each token's selected experts
    int* __restrict__ expert_counts = args.expert_counts;                              // [num_experts]
    int* __restrict__ expert_token_ids = args.expert_token_ids;                        // [num_experts, N]
    float* __restrict__ expert_token_weights = args.expert_token_weights;              // [num_experts, N]
    half* __restrict__ per_expert_wmma_inputs = args.per_expert_wmma_inputs;           // [num_experts, N, d_model]

    float* __restrict__ hidden_mlp_layer_1_out = args.hidden_mlp_layer_1_out;          // [num_experts, N, 4 * d_model]
    float* __restrict__ hidden_mlp_gate_out = args.hidden_mlp_gate_out;                // [num_experts, N, 4 * d_model]
    half* __restrict__ hidden_mlp_layer_1_out_fp16 = args.hidden_mlp_layer_1_out_fp16; // [num_experts, N, 4 * d_model]
    float* __restrict__ hidden_mlp_layer_2_out = args.hidden_mlp_layer_2_out;          // [num_experts, N, d_model]

    float* __restrict__ final_output = args.final_output;                               // [N, d_model]

    // **** list of kernel arguments: end ****


    zero_final_output_and_expert_counts(final_output, expert_counts);

    wmma_db<false>(1.0f, input, router_weights, expert_logits, N, num_experts, d_model);

    top_k_gating(expert_logits, selected_expert_indices, selected_expert_weights);
    __syncthreads(); // ensure selected_expert_indices/weights are globally visible before build_per_expert_buffers reads them

    build_per_expert_buffers(selected_expert_indices, selected_expert_weights, expert_counts, expert_token_ids, expert_token_weights); 
    __syncthreads(); // ensure expert_token_ids/weights and expert_counts are visible to this block

    // Clamp expert counts to CAP to avoid out-of-bounds reads/writes in subsequent kernels
    clamp_expert_counts(expert_counts);
    __syncthreads();

    assign_per_expert_wmma_inputs(input, expert_counts, expert_token_ids, per_expert_wmma_inputs);

    // compute capacity and use it for per-expert buffer sizes
    int CAP_kernel_raw = (int)ceilf((float)N * (float)k / (float)num_experts * capacity_factor);
    int CAP_kernel = ((CAP_kernel_raw + WMMA_M - 1) / WMMA_M) * WMMA_M;

    wmma_db<true>(1.0f, per_expert_wmma_inputs, expert_up_proj_weights, hidden_mlp_layer_1_out, num_experts * CAP_kernel, up_proj_dim * d_model, d_model);
    wmma_db<true>(1.0f, per_expert_wmma_inputs, expert_gate_proj_weights, hidden_mlp_gate_out, num_experts * CAP_kernel, up_proj_dim * d_model, d_model);

    fp32_to_fp16(hidden_mlp_layer_1_out, hidden_mlp_gate_out, hidden_mlp_layer_1_out_fp16, num_experts * CAP_kernel * up_proj_dim * d_model);
    __syncthreads(); // ensure fp32->fp16 conversion is complete before second GEMM

    wmma_db<true>(1.0f, hidden_mlp_layer_1_out_fp16, expert_down_proj_weights, hidden_mlp_layer_2_out, num_experts * CAP_kernel, d_model, up_proj_dim * d_model);
    __syncthreads(); // ensure hidden_mlp_layer_2_out is fully written before combine reads it

    combine(hidden_mlp_layer_2_out, expert_token_ids, expert_token_weights, expert_counts, final_output);

}


#include "launcher.h"