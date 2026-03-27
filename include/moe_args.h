#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

struct MoEArgs {
    const half* input;                              // [N, d_model]
    // model weights
    const half* router_weights;                     // [d_model, num_experts]
    const half* expert_up_proj_weights;             // [num_experts, d_model, 4 * d_model]
    const half* expert_down_proj_weights;           // [num_experts, 4 * d_model, d_model]
    //intermediate buffers, pre-allocated by host
    float*  expert_logits;                          // [N, num_experts] = output of the routing layer
    int*    selected_expert_indices;                // [N, k] = each token's selected experts
    float*  selected_expert_weights;                // [N, k] = weights of each token's selected experts
    int*    expert_counts;                          // [num_experts]
    int*    expert_token_ids;                       // [num_experts, N]
    float*  expert_token_weights;                   // [num_experts, N]
    half* per_expert_wmma_inputs;                   // [num_experts, N, d_model]
    float* hidden_mlp_layer_1_out;                  // [num_experts, N, 4 * d_model]
    half* hidden_mlp_layer_1_out_fp16;
    float* hidden_mlp_layer_2_out;                  // [num_experts, N, d_model]
    //final output
    float* final_output;                                   // [N, d_model]
    //number of batches
    int num_batches;
};