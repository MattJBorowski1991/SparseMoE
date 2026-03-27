#include <cstring>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "data.h"
#include "../include/config.h"
#include "../utils/check_cuda.h"

void initialize_host_data(
    std::vector<half>& h_input,
    std::vector<float>& h_final_output,
    std::vector<half>& h_expert_up_proj_weights,
    std::vector<half>& h_expert_down_proj_weights
) {
    // Zero-initialize all host vectors
    std::fill(h_input.begin(), h_input.end(), __float2half(0.0f));
    std::fill(h_final_output.begin(), h_final_output.end(), 0.0f);
    std::fill(h_expert_up_proj_weights.begin(), h_expert_up_proj_weights.end(), __float2half(0.0f));
    std::fill(h_expert_down_proj_weights.begin(), h_expert_down_proj_weights.end(), __float2half(0.0f));
}

MoEArgs allocate_and_copy_to_device(
    const std::vector<half>& h_input,
    const std::vector<float>& h_final_output,
    const std::vector<half>& h_expert_up_proj_weights,
    const std::vector<half>& h_expert_down_proj_weights,
    bool use_capacity
) {
    MoEArgs args;
    args.num_batches = num_batches;
    args.use_capacity = use_capacity;

    // auto print_mem_info = [&](const char* name, size_t req){
    //     size_t free_bytes = 0, total_bytes = 0;
    //     cudaError_t e = cudaMemGetInfo(&free_bytes, &total_bytes);
    //     if (e == cudaSuccess) {
    //         printf("[MEM] %s: requesting %zu bytes (%.2f MiB), free=%zu bytes (%.2f MiB), total=%zu bytes (%.2f MiB)\n",
    //                name, req, req / 1024.0 / 1024.0, free_bytes, free_bytes / 1024.0 / 1024.0, total_bytes, total_bytes / 1024.0 / 1024.0);
    //     } else {
    //         printf("[MEM] %s: requesting %zu bytes (cudaMemGetInfo failed: %d)\n", name, req, (int)e);
    //     }
    // };


    

    // Allocate and copy host input data to device
    {
        size_t size = (size_t)num_batches * N * d_model * sizeof(half);
        
        CHECK_CUDA(cudaMalloc(&args.input, size));
        CHECK_CUDA(cudaMemcpy((void*)args.input, h_input.data(), size, cudaMemcpyHostToDevice));
    }

    // Allocate router_weights (initialized to zero on device)
    {
        size_t size = d_model * num_experts * sizeof(half);
        
        CHECK_CUDA(cudaMalloc(&args.router_weights, size));
        CHECK_CUDA(cudaMemset((void*)args.router_weights, 0, size));
    }

    // Allocate and copy expert_up_proj_weights
    {
        size_t size = (size_t)num_experts * d_model * (up_proj_dim * d_model) * sizeof(half);
        
        CHECK_CUDA(cudaMalloc(&args.expert_up_proj_weights, size));
        CHECK_CUDA(cudaMemcpy((void*)args.expert_up_proj_weights, h_expert_up_proj_weights.data(), size, cudaMemcpyHostToDevice));
    }

    // Allocate and copy expert_down_proj_weights
    {
        size_t size = (size_t)num_experts * (up_proj_dim * d_model) * d_model * sizeof(half);
        
        CHECK_CUDA(cudaMalloc(&args.expert_down_proj_weights, size));
        CHECK_CUDA(cudaMemcpy((void*)args.expert_down_proj_weights, h_expert_down_proj_weights.data(), size, cudaMemcpyHostToDevice));
    }

    // Determine per-expert capacity if requested
    int CAP = N;
    if (use_capacity) {
        int CAP_raw = (int)ceilf((float)N * (float)k / (float)num_experts * capacity_factor);
        CAP = ((CAP_raw + WMMA_M - 1) / WMMA_M) * WMMA_M; // pad up to the next wmma size so we can use tensor cores
    }
    args.cap = CAP;

    // Allocate intermediate buffers (device-only, will be initialized by kernel)
    {
        // expert_logits: [num_batches, N, num_experts]
        size_t size = (size_t)num_batches * N * num_experts * sizeof(float);
        
        CHECK_CUDA(cudaMalloc(&args.expert_logits, size));
    }

    {
        // selected_expert_indices: [num_batches, N, k]
        size_t size = (size_t)num_batches * N * k * sizeof(int);
        
        CHECK_CUDA(cudaMalloc(&args.selected_expert_indices, size));
    }

    {
        // selected_expert_weights: [num_batches, N, k]
        size_t size = (size_t)num_batches * N * k * sizeof(float);
        
        CHECK_CUDA(cudaMalloc(&args.selected_expert_weights, size));
    }

    {
        // expert_counts: [num_batches, num_experts]
        size_t size = (size_t)num_batches * num_experts * sizeof(int);
        
        CHECK_CUDA(cudaMalloc(&args.expert_counts, size));
    }

    {
        // expert_token_ids: [num_batches, num_experts, CAP]
        size_t size = (size_t)num_batches * num_experts * CAP * sizeof(int);
        
        CHECK_CUDA(cudaMalloc(&args.expert_token_ids, size));
    }

    {
        // expert_token_weights: [num_batches, num_experts, CAP]
        size_t size = (size_t)num_batches * num_experts * CAP * sizeof(float);
        
        CHECK_CUDA(cudaMalloc(&args.expert_token_weights, size));
    }

    {
        // per_expert_wmma_inputs: [num_batches, num_experts, CAP, d_model]
        size_t size = (size_t)num_batches * num_experts * CAP * d_model * sizeof(half);
        
        CHECK_CUDA(cudaMalloc(&args.per_expert_wmma_inputs, size));
    }

    {
        // hidden_mlp_layer_1_out: [num_batches, num_experts, CAP, 4 * d_model]
        size_t size = (size_t)num_batches * num_experts * CAP * (up_proj_dim * d_model) * sizeof(float);
        
        CHECK_CUDA(cudaMalloc(&args.hidden_mlp_layer_1_out, size));
    }

    {
        // hidden_mlp_layer_1_out_fp16: [num_batches, num_experts, CAP, 4 * d_model]
        size_t size = (size_t)num_batches * num_experts * CAP * (up_proj_dim * d_model) * sizeof(half);
        
        CHECK_CUDA(cudaMalloc(&args.hidden_mlp_layer_1_out_fp16, size));
    }

    {
        // hidden_mlp_layer_2_out: [num_batches, num_experts, CAP, d_model]
        size_t size = (size_t)num_batches * num_experts * CAP * d_model * sizeof(float);
        
        CHECK_CUDA(cudaMalloc(&args.hidden_mlp_layer_2_out, size));
    }

    {
        // final_output: [num_batches, N, d_model]
        size_t size = (size_t)num_batches * N * d_model * sizeof(float);
        
        CHECK_CUDA(cudaMalloc(&args.final_output, size));
    }

    return args;
}

void cleanup_device_data(MoEArgs& args) {
    // Free all device pointers
    if (args.input) CHECK_CUDA(cudaFree((void*)args.input));
    if (args.router_weights) CHECK_CUDA(cudaFree((void*)args.router_weights));
    if (args.expert_up_proj_weights) CHECK_CUDA(cudaFree((void*)args.expert_up_proj_weights));
    if (args.expert_down_proj_weights) CHECK_CUDA(cudaFree((void*)args.expert_down_proj_weights));
    if (args.expert_logits) CHECK_CUDA(cudaFree(args.expert_logits));
    if (args.selected_expert_indices) CHECK_CUDA(cudaFree(args.selected_expert_indices));
    if (args.selected_expert_weights) CHECK_CUDA(cudaFree(args.selected_expert_weights));
    if (args.expert_counts) CHECK_CUDA(cudaFree(args.expert_counts));
    if (args.expert_token_ids) CHECK_CUDA(cudaFree(args.expert_token_ids));
    if (args.expert_token_weights) CHECK_CUDA(cudaFree(args.expert_token_weights));
    if (args.per_expert_wmma_inputs) CHECK_CUDA(cudaFree(args.per_expert_wmma_inputs));
    if (args.hidden_mlp_layer_1_out) CHECK_CUDA(cudaFree(args.hidden_mlp_layer_1_out));
    if (args.hidden_mlp_layer_1_out_fp16) CHECK_CUDA(cudaFree(args.hidden_mlp_layer_1_out_fp16));
    if (args.hidden_mlp_layer_2_out) CHECK_CUDA(cudaFree(args.hidden_mlp_layer_2_out));
    if (args.final_output) CHECK_CUDA(cudaFree(args.final_output));
}
