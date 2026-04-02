#pragma once

#include <vector>
#include <cuda_fp16.h>
#include "../include/moe_args.h"

// Zero-initialize all host vectors
void initialize_host_data(
    std::vector<half>& h_input,
    std::vector<float>& h_final_output,
    std::vector<half>& h_expert_up_proj_weights,
    std::vector<half>& h_expert_gate_proj_weights,
    std::vector<half>& h_expert_down_proj_weights
);

// Allocate device memory and copy host data to device, return populated MoEArgs
MoEArgs allocate_and_copy_to_device(
    const std::vector<half>& h_input,
    const std::vector<half>& h_expert_up_proj_weights,
    const std::vector<half>& h_expert_gate_proj_weights,
    const std::vector<half>& h_expert_down_proj_weights,
    bool use_capacity
);

// Dedicated int8 path: skips fp16 up/down expert weight allocations and uploads int8+scales only.
MoEArgs allocate_and_copy_to_device_int8(
    const std::vector<half>& h_input,
    const std::vector<half>& h_expert_up_proj_weights,
    const std::vector<half>& h_expert_gate_proj_weights,
    const std::vector<half>& h_expert_down_proj_weights,
    bool use_capacity
);

// Dedicated int4 path: uploads int4-range values stored in int8 buffers and matching scales.
MoEArgs allocate_and_copy_to_device_int4(
    const std::vector<half>& h_input,
    const std::vector<half>& h_expert_up_proj_weights,
    const std::vector<half>& h_expert_gate_proj_weights,
    const std::vector<half>& h_expert_down_proj_weights,
    bool use_capacity
);

// Dedicated int4 PTX path: uploads packed+transposed int4 weights for capacity_int4_ptx.
MoEArgs allocate_and_copy_to_device_int4_ptx(
    const std::vector<half>& h_input,
    const std::vector<half>& h_expert_up_proj_weights,
    const std::vector<half>& h_expert_gate_proj_weights,
    const std::vector<half>& h_expert_down_proj_weights,
    bool use_capacity
);

// Dedicated fp8 path: uploads fp8 (e4m3) weights and fp8 activation buffers.
MoEArgs allocate_and_copy_to_device_fp8(
    const std::vector<half>& h_input,
    const std::vector<half>& h_expert_up_proj_weights,
    const std::vector<half>& h_expert_gate_proj_weights,
    const std::vector<half>& h_expert_down_proj_weights,
    bool use_capacity
);

// Free all device memory allocated in MoEArgs
void cleanup_device_data(MoEArgs& args);
