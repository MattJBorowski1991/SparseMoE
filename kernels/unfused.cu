#include "include/config.h"
#include "include/moe_args.h"
#include "utils/check_cuda.h"
#include <mma.h>
#include <stdio.h>
using namespace nvcuda;

extern "C" __global__ void zero_final_output_and_expert_counts_kernel(float* final_output, int* expert_counts) {
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

extern "C" __global__ void wmma_db_kernel(const half* A, const half* B, float* C, int M, int N_, int K, int calculatePerExpert) {
    int batch = blockIdx.z;

    const half* A_batch = A + batch * M * K;
    const half* B_batch = B;
    float* C_batch = C + batch * M * N_;

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
    if (tile_row >= M || tile_col >= N_) return;

    int tile_row_local = tile_row;

    if (calculatePerExpert) {
        const int row_id = tile_row;
        const int rows_per_expert = M / num_experts;
        const int expert_id = row_id / rows_per_expert;
        tile_row_local = row_id % rows_per_expert;

        A_e = A_batch + expert_id * rows_per_expert * K;
        B_e = B_batch + expert_id * K * N_;
        C_e = C_batch + expert_id * rows_per_expert * N_;
    } else {
        A_e = A_batch;
        B_e = B_batch;
        C_e = C_batch;
    }

    __shared__ __align__(16) half As[2][WARPS_PER_BLOCK][WMMA_M][WMMA_K + PAD];
    __shared__ __align__(16) half Bs[2][WARPS_PER_BLOCK][WMMA_K][WMMA_N + PAD];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int buf = 0;

    for (int i = lane_id; i < WMMA_M * WMMA_K; i += 32) {
        int row = i / WMMA_K;
        int col = i % WMMA_K;
        As[buf][warp_id][row][col] = A_e[(tile_row_local + row) * K + col];
    }
    for (int i = lane_id; i < WMMA_K * WMMA_N; i += 32) {
        int row = i / WMMA_N;
        int col = i % WMMA_N;
        Bs[buf][warp_id][row][col] = B_e[row * N_ + (tile_col + col)];
    }
    __syncthreads(); 

    wmma::load_matrix_sync(a_frag, &As[buf][warp_id][0][0], WMMA_K + PAD);
    wmma::load_matrix_sync(b_frag, &Bs[buf][warp_id][0][0], WMMA_N + PAD);

    for (int k = WMMA_K; k < K; k += WMMA_K){
        int next = 1 - buf;

        for (int i = lane_id * 8; i < WMMA_M * WMMA_K; i += 32 * 8) {
            int row = i / WMMA_K;
            int col = i % WMMA_K;

            char* dst = (char*)&As[next][warp_id][row][col];
            const char* src = (const char*)&A_e[(tile_row_local + row) * K + (k + col)];

            unsigned smem_addr = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                        :: "r"(smem_addr), "l"(src));
        }

        for (int i = lane_id * 8; i < WMMA_K * WMMA_N; i += THREADS_PER_WARP * 8) {
            int row = i / WMMA_N;
            int col = i % WMMA_N;

            char* dst = (char*)&Bs[next][warp_id][row][col];
            const char* src = (const char*)&B_e[(k + row) * N_ + (tile_col + col)];

            unsigned smem_addr = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                        :: "r"(smem_addr), "l"(src));
        }

        asm volatile("cp.async.commit_group;");

        __syncthreads();
        

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        asm volatile("cp.async.wait_group 0;");

        buf = next;
        wmma::load_matrix_sync(a_frag, &As[buf][warp_id][0][0], WMMA_K + PAD);
        wmma::load_matrix_sync(b_frag, &Bs[buf][warp_id][0][0], WMMA_N + PAD);
    }

    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    float* c_dst = C_e + tile_row_local * N_ + tile_col;
    
    for (int i = 0; i < c_frag.num_elements; ++i) c_frag.x[i] = 1.0f * c_frag.x[i];
    
    wmma::store_matrix_sync(c_dst, c_frag, N_, wmma::mem_row_major);
}

extern "C" __global__ void top_k_gating_kernel(const float* logits, int* selected_expert_indices, float* selected_expert_weights, float* max_vals, int* max_idxs) {
    int batch = blockIdx.z;

    const float* logits_batch = logits + batch * N * num_experts;
    int* selected_expert_indices_batch = selected_expert_indices + batch * N * k;
    float* selected_expert_weights_batch = selected_expert_weights + batch * N * k;

    const int tid = threadIdx.x;

    const int warp_id = tid / THREADS_PER_WARP;
    const int lane_id = tid % THREADS_PER_WARP;

    const int token_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (token_id >= N) return;

    float local_max_vals[k];
    int local_max_idxs[k];

    const float* logits_row = logits_batch + token_id * num_experts;

    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < k; ++i) {
            local_max_vals[i] = -1e20f;
            local_max_idxs[i] = -1;
        }

        #pragma unroll
        for (int logit_id = 0; logit_id < num_experts; ++logit_id) {
            float val = logits_row[logit_id];
            if (val > local_max_vals[k - 1]) {
                local_max_vals[k - 1] = val;
                local_max_idxs[k - 1] = logit_id;

                #pragma unroll
                for (int i = k - 1; i > 0 && local_max_vals[i] > local_max_vals[i - 1]; --i) {
                    float tmp_v = local_max_vals[i - 1];
                    local_max_vals[i - 1] = local_max_vals[i];
                    local_max_vals[i] = tmp_v;

                    int tmp_i = local_max_idxs[i - 1];
                    local_max_idxs[i - 1] = local_max_idxs[i];
                    local_max_idxs[i] = tmp_i;
                }
            }
        }

        float max_val = local_max_vals[0];
        float sum_of_exps = 0.0f;
        #pragma unroll
        for (int l = 0; l < k; ++l) {
            sum_of_exps += expf(local_max_vals[l] - max_val);
        }

        #pragma unroll
        for (int l = 0; l < k; ++l) {
            selected_expert_indices_batch[token_id * k + l] = local_max_idxs[l];
            selected_expert_weights_batch[token_id * k + l] = expf(local_max_vals[l] - max_val) / (sum_of_exps + 1e-10f);
        }
    }
}

extern "C" __global__ void build_per_expert_buffers_kernel(const int* selected_expert_indices, const float* selected_expert_weights, int* expert_counts, int* expert_token_ids, float* expert_token_weights) {
    const int batch = blockIdx.z;

    const int* selected_expert_indices_b = selected_expert_indices + batch * N * k;
    const float* selected_expert_weights_b = selected_expert_weights + batch * N * k;
    int* expert_counts_b = expert_counts + batch * num_experts;
    int* expert_token_ids_b = expert_token_ids + batch * num_experts * N;
    float* expert_token_weights_b = expert_token_weights + batch * num_experts * N;

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

                if (slot < N) {
                    const int out_idx = expert_id * N + slot;
                    expert_token_ids_b[out_idx] = token_id;
                    expert_token_weights_b[out_idx] = selected_expert_weights_b[route_id];
                }
            }
        }
    }
}

extern "C" __global__ void assign_per_expert_wmma_inputs_kernel(const half* input, const int* expert_counts, const int* expert_token_ids, half* per_expert_wmma_inputs) {
    const int batch = blockIdx.z;

    const half* input_b = input + batch * N * d_model;
    const int* expert_counts_b = expert_counts + batch * num_experts;
    const int* expert_token_ids_b = expert_token_ids + batch * num_experts * N;
    half* per_expert_wmma_inputs_b = per_expert_wmma_inputs + batch * num_experts * N * d_model;

    const int tid = threadIdx.x;
    const int warp_id = tid / THREADS_PER_WARP;
    const int lane_id = tid % THREADS_PER_WARP;

    const int row_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int total_rows = num_experts * N;
    if (row_id >= total_rows) return;

    const int expert_id = row_id / N;
    const int slot = row_id % N;
    const int row_base = row_id * d_model;

    if (slot < expert_counts_b[expert_id]) {
        const int token_id = expert_token_ids_b[expert_id * N + slot];
        if (token_id >= 0 && token_id < N) {
            const int in_base = token_id * d_model;
            for (int col = lane_id; col < d_model; col += THREADS_PER_WARP) {
                per_expert_wmma_inputs_b[row_base + col] = input_b[in_base + col];
            }
        } else {
            for (int col = lane_id; col < d_model; col += THREADS_PER_WARP) {
                per_expert_wmma_inputs_b[row_base + col] = __float2half(0.0f);
            }
        }
    } else {
        for (int col = lane_id; col < d_model; col += THREADS_PER_WARP) {
            per_expert_wmma_inputs_b[row_base + col] = __float2half(0.0f);
        }
    }
}


extern "C" __global__ void fp32_to_fp16_kernel(const float* up_input, const float* gate_input, half* output, int total_size) {
    const int batch = blockIdx.z;

    const float* up_input_b = up_input + batch * num_experts * N * (up_proj_dim * d_model);
    const float* gate_input_b = gate_input + batch * num_experts * N * (up_proj_dim * d_model);
    half* output_b = output + batch *  num_experts * N * (up_proj_dim * d_model);

    const int block_linear = blockIdx.y * gridDim.x + blockIdx.x;
    const int global_tid = block_linear * blockDim.x + threadIdx.x;
    const int global_stride = gridDim.x * gridDim.y * blockDim.x;

    for (int idx = global_tid; idx < total_size; idx += global_stride) {
        const float gate = gate_input_b[idx];
        const float silu_gate = gate / (1.0f + __expf(-gate));
        output_b[idx] = __float2half(up_input_b[idx] * silu_gate);
    }
}

extern "C" __global__ void combine_kernel(const float* input, const int* expert_token_ids, const float* expert_token_weights, const int* expert_counts, float* final_output) {
    const int batch = blockIdx.z;
    const int rows_per_expert = N;
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

// Launcher that runs unfused kernels in sequence to mimic baseline workflow
extern "C" void solve(MoEArgs args){

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    dim3 threads(THREADS_PER_WARP * WARPS_PER_BLOCK);
    dim3 blocks_zero((N * d_model + threads.x - 1) / threads.x, 1, args.num_batches);
    dim3 blocks_router((num_experts + WMMA_N * WARP_TILES_X - 1) / (WMMA_N * WARP_TILES_X), (N + WMMA_M * WARP_TILES_Y - 1) / (WMMA_M * WARP_TILES_Y), args.num_batches);
    dim3 blocks_token((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, 1, args.num_batches);
    dim3 blocks_rows((num_experts * N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, 1, args.num_batches);
    dim3 blocks_up(( (up_proj_dim * d_model) + WMMA_N * WARP_TILES_X - 1) / (WMMA_N * WARP_TILES_X), (num_experts * N + WMMA_M * WARP_TILES_Y - 1) / (WMMA_M * WARP_TILES_Y), args.num_batches);
    dim3 blocks_down(( d_model + WMMA_N * WARP_TILES_X - 1) / (WMMA_N * WARP_TILES_X), (num_experts * N + WMMA_M * WARP_TILES_Y - 1) / (WMMA_M * WARP_TILES_Y), args.num_batches);
    dim3 blocks_convert((num_experts * N * up_proj_dim * d_model + threads.x - 1) / threads.x, 1, args.num_batches);


    zero_final_output_and_expert_counts_kernel<<<blocks_zero, threads>>>(args.final_output, args.expert_counts);

    wmma_db_kernel<<<blocks_router, threads>>>(args.input, args.router_weights, args.expert_logits, N, num_experts, d_model, 0);

    top_k_gating_kernel<<<blocks_token, threads>>>(args.expert_logits, args.selected_expert_indices, args.selected_expert_weights, nullptr, nullptr);

    build_per_expert_buffers_kernel<<<blocks_token, threads>>>(args.selected_expert_indices, args.selected_expert_weights, args.expert_counts, args.expert_token_ids, args.expert_token_weights);

    assign_per_expert_wmma_inputs_kernel<<<blocks_rows, threads>>>(args.input, args.expert_counts, args.expert_token_ids, args.per_expert_wmma_inputs);

    wmma_db_kernel<<<blocks_up, threads>>>(args.per_expert_wmma_inputs, args.expert_up_proj_weights, args.hidden_mlp_layer_1_out, num_experts * N, up_proj_dim * d_model, d_model, 1);
    wmma_db_kernel<<<blocks_up, threads>>>(args.per_expert_wmma_inputs, args.expert_gate_proj_weights, args.hidden_mlp_gate_out, num_experts * N, up_proj_dim * d_model, d_model, 1);

    fp32_to_fp16_kernel<<<blocks_convert, threads>>>(args.hidden_mlp_layer_1_out, args.hidden_mlp_gate_out, args.hidden_mlp_layer_1_out_fp16, num_experts * N * up_proj_dim * d_model);

    wmma_db_kernel<<<blocks_down, threads>>>(args.hidden_mlp_layer_1_out_fp16, args.expert_down_proj_weights, args.hidden_mlp_layer_2_out, num_experts * N, d_model, up_proj_dim * d_model, 1);

    combine_kernel<<<blocks_rows, threads>>>(args.hidden_mlp_layer_2_out, args.expert_token_ids, args.expert_token_weights, args.expert_counts, args.final_output);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    printf("Total duration via cudaEventRecord: %.3f ms\n", ms);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}
