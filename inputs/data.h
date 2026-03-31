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
    const std::vector<float>& h_final_output,
    const std::vector<half>& h_expert_up_proj_weights,
    const std::vector<half>& h_expert_gate_proj_weights,
    const std::vector<half>& h_expert_down_proj_weights,
    bool use_capacity
);

// Free all device memory allocated in MoEArgs
void cleanup_device_data(MoEArgs& args);
