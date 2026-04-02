#pragma once

#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

struct MoEArgs {
    const half* input;                              // [N, d_model]
    // model weights
    const half* router_weights;                     // [d_model, num_experts]
    const half* expert_up_proj_weights;             // [num_experts, d_model, 4 * d_model]
    const half* expert_gate_proj_weights;           // [num_experts, d_model, 4 * d_model]
    const half* expert_down_proj_weights;           // [num_experts, 4 * d_model, d_model]
    const int8_t* expert_up_proj_weights_int8;      // [num_experts, d_model, 4 * d_model]
    const int8_t* expert_gate_proj_weights_int8;    // [num_experts, d_model, 4 * d_model]
    const int8_t* expert_down_proj_weights_int8;    // [num_experts, 4 * d_model, d_model]
    const uint8_t* expert_up_proj_weights_int4;     // packed int4: [num_experts, d_model, (4 * d_model)/2]
    const uint8_t* expert_gate_proj_weights_int4;   // packed int4: [num_experts, d_model, (4 * d_model)/2]
    const uint8_t* expert_down_proj_weights_int4;   // packed int4: [num_experts, (4 * d_model), d_model/2]
    const __nv_fp8_e4m3* expert_up_proj_weights_fp8;    // [num_experts, d_model, 4 * d_model]
    const __nv_fp8_e4m3* expert_gate_proj_weights_fp8;  // [num_experts, d_model, 4 * d_model]
    const __nv_fp8_e4m3* expert_down_proj_weights_fp8;  // [num_experts, 4 * d_model, d_model]
    // scalar scales used by true-int8 capacity kernel
    float scale_up_w;
    float scale_gate_w;
    float scale_down_w;
    float scale_input_act;
    float scale_mid_act;
    //intermediate buffers, pre-allocated by host
    float*  expert_logits;                          // [N, num_experts] = output of the routing layer
    int*    selected_expert_indices;                // [N, k] = each token's selected experts
    float*  selected_expert_weights;                // [N, k] = weights of each token's selected experts
    int*    expert_counts;                          // [num_experts]
    int*    expert_token_ids;                       // [num_experts, N]
    float*  expert_token_weights;                   // [num_experts, N]
    half* per_expert_wmma_inputs;                   // [num_experts, N, d_model]
    int8_t* per_expert_wmma_inputs_int8;            // [num_experts, N, d_model]
    uint8_t* per_expert_wmma_inputs_int4;           // packed int4: [num_experts, N, d_model/2]
    __nv_fp8_e4m3* per_expert_wmma_inputs_fp8; // [num_experts, N, d_model]
    float* hidden_mlp_layer_1_out;                  // [num_experts, N, 4 * d_model]
    float* hidden_mlp_gate_out;                     // [num_experts, N, 4 * d_model]
    half* hidden_mlp_layer_1_out_fp16;
    int8_t* hidden_mlp_layer_1_out_int8;            // [num_experts, N, 4 * d_model]
    uint8_t* hidden_mlp_layer_1_out_int4;           // packed int4: [num_experts, N, (4 * d_model)/2]
    __nv_fp8_e4m3* hidden_mlp_layer_1_out_fp8; // [num_experts, N, 4 * d_model]
    float* hidden_mlp_layer_2_out;                  // [num_experts, N, d_model]
    //final output
    float* final_output;                                   // [N, d_model]
    //number of batches
    int num_batches;
    // capacity option (host computed). If true, `cap` holds the per-expert CAP value used by capacity variant.
    bool use_capacity;
    int cap;
};