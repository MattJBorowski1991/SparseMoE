#include <cstring>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "data.h"
#include "../include/config.h"
#include "../utils/check_cuda.h"

static float quantize_symmetric_per_tensor_int8(
    const std::vector<half>& src,
    std::vector<int8_t>& qdst
) {
    qdst.resize(src.size());

    float max_abs = 0.0f;
    for (size_t i = 0; i < src.size(); ++i) {
        float v = fabsf(__half2float(src[i]));
        if (v > max_abs) max_abs = v;
    }

    float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    for (size_t i = 0; i < src.size(); ++i) {
        float x = __half2float(src[i]) / scale;
        float clamped = fminf(127.0f, fmaxf(-127.0f, x));
        int q = (int)nearbyintf(clamped);
        qdst[i] = (int8_t)q;
    }

    return scale;
}

static float quantize_symmetric_per_tensor_int4(
    const std::vector<half>& src,
    std::vector<int8_t>& qdst
) {
    qdst.resize(src.size());

    float max_abs = 0.0f;
    for (size_t i = 0; i < src.size(); ++i) {
        float v = fabsf(__half2float(src[i]));
        if (v > max_abs) max_abs = v;
    }

    float scale = (max_abs > 0.0f) ? (max_abs / 7.0f) : 1.0f;
    for (size_t i = 0; i < src.size(); ++i) {
        float x = __half2float(src[i]) / scale;
        float clamped = fminf(7.0f, fmaxf(-7.0f, x));
        int q = (int)nearbyintf(clamped);
        qdst[i] = (int8_t)q;
    }

    return scale;
}

static float quantize_symmetric_per_tensor_int4_packed(
    const std::vector<half>& src,
    std::vector<uint8_t>& qdst_packed
) {
    const size_t n = src.size();
    qdst_packed.assign((n + 1) / 2, 0);

    float max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float v = fabsf(__half2float(src[i]));
        if (v > max_abs) max_abs = v;
    }

    float scale = (max_abs > 0.0f) ? (max_abs / 7.0f) : 1.0f;
    for (size_t i = 0; i < n; ++i) {
        float x = __half2float(src[i]) / scale;
        float clamped = fminf(7.0f, fmaxf(-7.0f, x));
        int q = (int)nearbyintf(clamped);
        uint8_t nib = (uint8_t)(q & 0xF);
        const size_t byte_idx = i >> 1;
        if ((i & 1) == 0) {
            qdst_packed[byte_idx] = (qdst_packed[byte_idx] & 0xF0) | nib;
        } else {
            qdst_packed[byte_idx] = (qdst_packed[byte_idx] & 0x0F) | (uint8_t)(nib << 4);
        }
    }

    return scale;
}

static float quantize_symmetric_per_tensor_int4_packed_transposed(
    const std::vector<half>& src,
    std::vector<uint8_t>& qdst_packed_transposed,
    int experts,
    int K,
    int N
) {
    const size_t elems_per_expert = (size_t)K * (size_t)N;
    const size_t expected = (size_t)experts * elems_per_expert;
    if (src.size() != expected) {
        qdst_packed_transposed.clear();
        return 1.0f;
    }

    // Packed-transposed byte layout per expert: [N/2][K].
    // Each byte stores two adjacent N-column int4 values at fixed K.
    const size_t bytes_per_expert = (size_t)(N / 2) * (size_t)K;
    qdst_packed_transposed.assign((size_t)experts * bytes_per_expert, 0);

    float max_abs = 0.0f;
    for (size_t i = 0; i < src.size(); ++i) {
        float v = fabsf(__half2float(src[i]));
        if (v > max_abs) max_abs = v;
    }

    float scale = (max_abs > 0.0f) ? (max_abs / 7.0f) : 1.0f;

    for (int e = 0; e < experts; ++e) {
        const size_t src_base = (size_t)e * elems_per_expert;
        const size_t dst_base = (size_t)e * bytes_per_expert;
        for (int k_idx = 0; k_idx < K; ++k_idx) {
            for (int n_idx = 0; n_idx < N; ++n_idx) {
                const size_t src_idx = src_base + (size_t)k_idx * (size_t)N + (size_t)n_idx;
                float x = __half2float(src[src_idx]) / scale;
                float clamped = fminf(7.0f, fmaxf(-7.0f, x));
                int q = (int)nearbyintf(clamped);
                uint8_t nib = (uint8_t)(q & 0xF);

                const size_t byte_idx = dst_base + (size_t)(n_idx / 2) * (size_t)K + (size_t)k_idx;
                if ((n_idx & 1) == 0) {
                    qdst_packed_transposed[byte_idx] =
                        (qdst_packed_transposed[byte_idx] & 0xF0) | nib;
                } else {
                    qdst_packed_transposed[byte_idx] =
                        (qdst_packed_transposed[byte_idx] & 0x0F) | (uint8_t)(nib << 4);
                }
            }
        }
    }

    return scale;
}

static void quantize_per_tensor_fp8_e4m3(
    const std::vector<half>& src,
    std::vector<__nv_fp8_e4m3>& qdst
) {
    qdst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        qdst[i] = __nv_fp8_e4m3(__half2float(src[i]));
    }
}

static MoEArgs allocate_and_copy_to_device_impl(
    const std::vector<half>& h_input,
    const std::vector<half>& h_expert_up_proj_weights,
    const std::vector<half>& h_expert_gate_proj_weights,
    const std::vector<half>& h_expert_down_proj_weights,
    bool use_capacity,
    bool int8_only,
    bool fp8_only,
    bool int4_only,
    bool int4_ptx_layout
);

void initialize_host_data(
    std::vector<half>& h_input,
    std::vector<float>& h_final_output,
    std::vector<half>& h_expert_up_proj_weights,
    std::vector<half>& h_expert_gate_proj_weights,
    std::vector<half>& h_expert_down_proj_weights
) {
    // Zero-initialize all host vectors
    std::fill(h_input.begin(), h_input.end(), __float2half(0.0f));
    std::fill(h_final_output.begin(), h_final_output.end(), 0.0f);
    std::fill(h_expert_up_proj_weights.begin(), h_expert_up_proj_weights.end(), __float2half(0.0f));
    std::fill(h_expert_gate_proj_weights.begin(), h_expert_gate_proj_weights.end(), __float2half(0.0f));
    std::fill(h_expert_down_proj_weights.begin(), h_expert_down_proj_weights.end(), __float2half(0.0f));
}

MoEArgs allocate_and_copy_to_device(
    const std::vector<half>& h_input,
    const std::vector<half>& h_expert_up_proj_weights,
    const std::vector<half>& h_expert_gate_proj_weights,
    const std::vector<half>& h_expert_down_proj_weights,
    bool use_capacity
) {
    return allocate_and_copy_to_device_impl(
        h_input,
        h_expert_up_proj_weights,
        h_expert_gate_proj_weights,
        h_expert_down_proj_weights,
        use_capacity,
        false,
        false,
        false,
        false
    );
}

static MoEArgs allocate_and_copy_to_device_impl(
    const std::vector<half>& h_input,
    const std::vector<half>& h_expert_up_proj_weights,
    const std::vector<half>& h_expert_gate_proj_weights,
    const std::vector<half>& h_expert_down_proj_weights,
    bool use_capacity,
    bool int8_only,
    bool fp8_only,
    bool int4_only,
    bool int4_ptx_layout
) {
    MoEArgs args;
    std::memset(&args, 0, sizeof(MoEArgs));
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

    if (!int8_only && !fp8_only && !int4_only) {
        // Allocate and copy expert_up_proj_weights
        {
            size_t size = (size_t)num_experts * d_model * (up_proj_dim * d_model) * sizeof(half);

            CHECK_CUDA(cudaMalloc(&args.expert_up_proj_weights, size));
            CHECK_CUDA(cudaMemcpy((void*)args.expert_up_proj_weights, h_expert_up_proj_weights.data(), size, cudaMemcpyHostToDevice));
        }
    }

    if (!int8_only && !fp8_only && !int4_only) {
        // Allocate and copy expert_gate_proj_weights
        {
            size_t size = (size_t)num_experts * d_model * (up_proj_dim * d_model) * sizeof(half);

            CHECK_CUDA(cudaMalloc(&args.expert_gate_proj_weights, size));
            CHECK_CUDA(cudaMemcpy((void*)args.expert_gate_proj_weights, h_expert_gate_proj_weights.data(), size, cudaMemcpyHostToDevice));
        }
    }

    if (!int8_only && !fp8_only && !int4_only) {
        // Allocate and copy expert_down_proj_weights
        {
            size_t size = (size_t)num_experts * (up_proj_dim * d_model) * d_model * sizeof(half);

            CHECK_CUDA(cudaMalloc(&args.expert_down_proj_weights, size));
            CHECK_CUDA(cudaMemcpy((void*)args.expert_down_proj_weights, h_expert_down_proj_weights.data(), size, cudaMemcpyHostToDevice));
        }
    }

    if (int8_only || int4_only) {
        std::vector<int8_t> h_up_q;
        std::vector<int8_t> h_gate_q;
        std::vector<int8_t> h_down_q;
        std::vector<uint8_t> h_up_q_packed;
        std::vector<uint8_t> h_gate_q_packed;
        std::vector<uint8_t> h_down_q_packed;

        if (int4_only) {
            args.scale_up_w = quantize_symmetric_per_tensor_int4(h_expert_up_proj_weights, h_up_q);
            args.scale_gate_w = quantize_symmetric_per_tensor_int4(h_expert_gate_proj_weights, h_gate_q);
            args.scale_down_w = quantize_symmetric_per_tensor_int4(h_expert_down_proj_weights, h_down_q);
            if (int4_ptx_layout) {
                quantize_symmetric_per_tensor_int4_packed_transposed(
                    h_expert_up_proj_weights,
                    h_up_q_packed,
                    num_experts,
                    d_model,
                    up_proj_dim * d_model
                );
                quantize_symmetric_per_tensor_int4_packed_transposed(
                    h_expert_gate_proj_weights,
                    h_gate_q_packed,
                    num_experts,
                    d_model,
                    up_proj_dim * d_model
                );
                quantize_symmetric_per_tensor_int4_packed_transposed(
                    h_expert_down_proj_weights,
                    h_down_q_packed,
                    num_experts,
                    up_proj_dim * d_model,
                    d_model
                );
            } else {
                quantize_symmetric_per_tensor_int4_packed(h_expert_up_proj_weights, h_up_q_packed);
                quantize_symmetric_per_tensor_int4_packed(h_expert_gate_proj_weights, h_gate_q_packed);
                quantize_symmetric_per_tensor_int4_packed(h_expert_down_proj_weights, h_down_q_packed);
            }
        } else {
            args.scale_up_w = quantize_symmetric_per_tensor_int8(h_expert_up_proj_weights, h_up_q);
            args.scale_gate_w = quantize_symmetric_per_tensor_int8(h_expert_gate_proj_weights, h_gate_q);
            args.scale_down_w = quantize_symmetric_per_tensor_int8(h_expert_down_proj_weights, h_down_q);
        }

        // Placeholder activation scales; replace with calibrated values when available.
        args.scale_input_act = 1.0f;
        args.scale_mid_act = 1.0f;

        {
            size_t qsize = h_up_q.size() * sizeof(int8_t);
            CHECK_CUDA(cudaMalloc(&args.expert_up_proj_weights_int8, qsize));
            CHECK_CUDA(cudaMemcpy((void*)args.expert_up_proj_weights_int8, h_up_q.data(), qsize, cudaMemcpyHostToDevice));
        }
        {
            size_t qsize = h_gate_q.size() * sizeof(int8_t);
            CHECK_CUDA(cudaMalloc(&args.expert_gate_proj_weights_int8, qsize));
            CHECK_CUDA(cudaMemcpy((void*)args.expert_gate_proj_weights_int8, h_gate_q.data(), qsize, cudaMemcpyHostToDevice));
        }
        {
            size_t qsize = h_down_q.size() * sizeof(int8_t);
            CHECK_CUDA(cudaMalloc(&args.expert_down_proj_weights_int8, qsize));
            CHECK_CUDA(cudaMemcpy((void*)args.expert_down_proj_weights_int8, h_down_q.data(), qsize, cudaMemcpyHostToDevice));
        }

        if (int4_only) {
            {
                size_t qsize = h_up_q_packed.size() * sizeof(uint8_t);
                CHECK_CUDA(cudaMalloc(&args.expert_up_proj_weights_int4, qsize));
                CHECK_CUDA(cudaMemcpy((void*)args.expert_up_proj_weights_int4, h_up_q_packed.data(), qsize, cudaMemcpyHostToDevice));
            }
            {
                size_t qsize = h_gate_q_packed.size() * sizeof(uint8_t);
                CHECK_CUDA(cudaMalloc(&args.expert_gate_proj_weights_int4, qsize));
                CHECK_CUDA(cudaMemcpy((void*)args.expert_gate_proj_weights_int4, h_gate_q_packed.data(), qsize, cudaMemcpyHostToDevice));
            }
            {
                size_t qsize = h_down_q_packed.size() * sizeof(uint8_t);
                CHECK_CUDA(cudaMalloc(&args.expert_down_proj_weights_int4, qsize));
                CHECK_CUDA(cudaMemcpy((void*)args.expert_down_proj_weights_int4, h_down_q_packed.data(), qsize, cudaMemcpyHostToDevice));
            }
        }
    }

    if (fp8_only) {
        std::vector<__nv_fp8_e4m3> h_up_q;
        std::vector<__nv_fp8_e4m3> h_gate_q;
        std::vector<__nv_fp8_e4m3> h_down_q;

        quantize_per_tensor_fp8_e4m3(h_expert_up_proj_weights, h_up_q);
        quantize_per_tensor_fp8_e4m3(h_expert_gate_proj_weights, h_gate_q);
        quantize_per_tensor_fp8_e4m3(h_expert_down_proj_weights, h_down_q);

        // FP8 path does not require explicit linear dequant scales.
        args.scale_up_w = 1.0f;
        args.scale_gate_w = 1.0f;
        args.scale_down_w = 1.0f;
        args.scale_input_act = 1.0f;
        args.scale_mid_act = 1.0f;

        {
            size_t qsize = h_up_q.size() * sizeof(__nv_fp8_e4m3);
            CHECK_CUDA(cudaMalloc(&args.expert_up_proj_weights_fp8, qsize));
            CHECK_CUDA(cudaMemcpy((void*)args.expert_up_proj_weights_fp8, h_up_q.data(), qsize, cudaMemcpyHostToDevice));
        }
        {
            size_t qsize = h_gate_q.size() * sizeof(__nv_fp8_e4m3);
            CHECK_CUDA(cudaMalloc(&args.expert_gate_proj_weights_fp8, qsize));
            CHECK_CUDA(cudaMemcpy((void*)args.expert_gate_proj_weights_fp8, h_gate_q.data(), qsize, cudaMemcpyHostToDevice));
        }
        {
            size_t qsize = h_down_q.size() * sizeof(__nv_fp8_e4m3);
            CHECK_CUDA(cudaMalloc(&args.expert_down_proj_weights_fp8, qsize));
            CHECK_CUDA(cudaMemcpy((void*)args.expert_down_proj_weights_fp8, h_down_q.data(), qsize, cudaMemcpyHostToDevice));
        }
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

    if (!int8_only && !fp8_only && !int4_only) {
        // per_expert_wmma_inputs: [num_batches, num_experts, CAP, d_model]
        size_t size = (size_t)num_batches * num_experts * CAP * d_model * sizeof(half);

        CHECK_CUDA(cudaMalloc(&args.per_expert_wmma_inputs, size));
    }

    if (int8_only || int4_only) {
        // per_expert_wmma_inputs_int8: [num_batches, num_experts, CAP, d_model]
        size_t size = (size_t)num_batches * num_experts * CAP * d_model * sizeof(int8_t);
        CHECK_CUDA(cudaMalloc(&args.per_expert_wmma_inputs_int8, size));
    }

    if (int4_only) {
        // per_expert_wmma_inputs_int4: packed [num_batches, num_experts, CAP, d_model/2]
        size_t size = (size_t)num_batches * num_experts * CAP * d_model * sizeof(uint8_t) / 2;
        CHECK_CUDA(cudaMalloc(&args.per_expert_wmma_inputs_int4, size));
    }

    if (fp8_only) {
        // per_expert_wmma_inputs_fp8: [num_batches, num_experts, CAP, d_model]
        size_t size = (size_t)num_batches * num_experts * CAP * d_model * sizeof(__nv_fp8_e4m3);
        CHECK_CUDA(cudaMalloc(&args.per_expert_wmma_inputs_fp8, size));
    }


    {
        // hidden_mlp_layer_1_out: [num_batches, num_experts, CAP, 4 * d_model]
        size_t size = (size_t)num_batches * num_experts * CAP * (up_proj_dim * d_model) * sizeof(float);
        
        CHECK_CUDA(cudaMalloc(&args.hidden_mlp_layer_1_out, size));
    }

    {
        // hidden_mlp_gate_out: [num_batches, num_experts, CAP, 4 * d_model]
        size_t size = (size_t)num_batches * num_experts * CAP * (up_proj_dim * d_model) * sizeof(float);
        
        CHECK_CUDA(cudaMalloc(&args.hidden_mlp_gate_out, size));
    }

    if (!int8_only && !fp8_only && !int4_only) {
        // hidden_mlp_layer_1_out_fp16: [num_batches, num_experts, CAP, 4 * d_model]
        size_t size = (size_t)num_batches * num_experts * CAP * (up_proj_dim * d_model) * sizeof(half);

        CHECK_CUDA(cudaMalloc(&args.hidden_mlp_layer_1_out_fp16, size));
    }

    if (int8_only || int4_only) {
        // hidden_mlp_layer_1_out_int8: [num_batches, num_experts, CAP, 4 * d_model]
        size_t size = (size_t)num_batches * num_experts * CAP * (up_proj_dim * d_model) * sizeof(int8_t);
        CHECK_CUDA(cudaMalloc(&args.hidden_mlp_layer_1_out_int8, size));
    }

    if (int4_only) {
        // hidden_mlp_layer_1_out_int4: packed [num_batches, num_experts, CAP, (4 * d_model)/2]
        size_t size = (size_t)num_batches * num_experts * CAP * (up_proj_dim * d_model) * sizeof(uint8_t) / 2;
        CHECK_CUDA(cudaMalloc(&args.hidden_mlp_layer_1_out_int4, size));
    }

    if (fp8_only) {
        // hidden_mlp_layer_1_out_fp8: [num_batches, num_experts, CAP, 4 * d_model]
        size_t size = (size_t)num_batches * num_experts * CAP * (up_proj_dim * d_model) * sizeof(__nv_fp8_e4m3);
        CHECK_CUDA(cudaMalloc(&args.hidden_mlp_layer_1_out_fp8, size));
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

MoEArgs allocate_and_copy_to_device_int8(
    const std::vector<half>& h_input,
    const std::vector<half>& h_expert_up_proj_weights,
    const std::vector<half>& h_expert_gate_proj_weights,
    const std::vector<half>& h_expert_down_proj_weights,
    bool use_capacity
){
    return allocate_and_copy_to_device_impl(
        h_input,
        h_expert_up_proj_weights,
        h_expert_gate_proj_weights,
        h_expert_down_proj_weights,
        use_capacity,
        true,
        false,
        false,
        false
    );
}

MoEArgs allocate_and_copy_to_device_int4(
    const std::vector<half>& h_input,
    const std::vector<half>& h_expert_up_proj_weights,
    const std::vector<half>& h_expert_gate_proj_weights,
    const std::vector<half>& h_expert_down_proj_weights,
    bool use_capacity
){
    return allocate_and_copy_to_device_impl(
        h_input,
        h_expert_up_proj_weights,
        h_expert_gate_proj_weights,
        h_expert_down_proj_weights,
        use_capacity,
        false,
        false,
        true,
        false
    );
}

MoEArgs allocate_and_copy_to_device_int4_ptx(
    const std::vector<half>& h_input,
    const std::vector<half>& h_expert_up_proj_weights,
    const std::vector<half>& h_expert_gate_proj_weights,
    const std::vector<half>& h_expert_down_proj_weights,
    bool use_capacity
){
    return allocate_and_copy_to_device_impl(
        h_input,
        h_expert_up_proj_weights,
        h_expert_gate_proj_weights,
        h_expert_down_proj_weights,
        use_capacity,
        false,
        false,
        true,
        true
    );
}

MoEArgs allocate_and_copy_to_device_fp8(
    const std::vector<half>& h_input,
    const std::vector<half>& h_expert_up_proj_weights,
    const std::vector<half>& h_expert_gate_proj_weights,
    const std::vector<half>& h_expert_down_proj_weights,
    bool use_capacity
){
    return allocate_and_copy_to_device_impl(
        h_input,
        h_expert_up_proj_weights,
        h_expert_gate_proj_weights,
        h_expert_down_proj_weights,
        use_capacity,
        false,
        true,
        false,
        false
    );
}

void cleanup_device_data(MoEArgs& args) {
    // Free all device pointers
    if (args.input) CHECK_CUDA(cudaFree((void*)args.input));
    if (args.router_weights) CHECK_CUDA(cudaFree((void*)args.router_weights));
    if (args.expert_up_proj_weights) CHECK_CUDA(cudaFree((void*)args.expert_up_proj_weights));
    if (args.expert_gate_proj_weights) CHECK_CUDA(cudaFree((void*)args.expert_gate_proj_weights));
    if (args.expert_down_proj_weights) CHECK_CUDA(cudaFree((void*)args.expert_down_proj_weights));
    if (args.expert_up_proj_weights_int8) CHECK_CUDA(cudaFree((void*)args.expert_up_proj_weights_int8));
    if (args.expert_gate_proj_weights_int8) CHECK_CUDA(cudaFree((void*)args.expert_gate_proj_weights_int8));
    if (args.expert_down_proj_weights_int8) CHECK_CUDA(cudaFree((void*)args.expert_down_proj_weights_int8));
    if (args.expert_up_proj_weights_int4) CHECK_CUDA(cudaFree((void*)args.expert_up_proj_weights_int4));
    if (args.expert_gate_proj_weights_int4) CHECK_CUDA(cudaFree((void*)args.expert_gate_proj_weights_int4));
    if (args.expert_down_proj_weights_int4) CHECK_CUDA(cudaFree((void*)args.expert_down_proj_weights_int4));
    if (args.expert_up_proj_weights_fp8) CHECK_CUDA(cudaFree((void*)args.expert_up_proj_weights_fp8));
    if (args.expert_gate_proj_weights_fp8) CHECK_CUDA(cudaFree((void*)args.expert_gate_proj_weights_fp8));
    if (args.expert_down_proj_weights_fp8) CHECK_CUDA(cudaFree((void*)args.expert_down_proj_weights_fp8));
    if (args.expert_logits) CHECK_CUDA(cudaFree(args.expert_logits));
    if (args.selected_expert_indices) CHECK_CUDA(cudaFree(args.selected_expert_indices));
    if (args.selected_expert_weights) CHECK_CUDA(cudaFree(args.selected_expert_weights));
    if (args.expert_counts) CHECK_CUDA(cudaFree(args.expert_counts));
    if (args.expert_token_ids) CHECK_CUDA(cudaFree(args.expert_token_ids));
    if (args.expert_token_weights) CHECK_CUDA(cudaFree(args.expert_token_weights));
    if (args.per_expert_wmma_inputs) CHECK_CUDA(cudaFree(args.per_expert_wmma_inputs));
    if (args.per_expert_wmma_inputs_int8) CHECK_CUDA(cudaFree(args.per_expert_wmma_inputs_int8));
    if (args.per_expert_wmma_inputs_int4) CHECK_CUDA(cudaFree(args.per_expert_wmma_inputs_int4));
    if (args.per_expert_wmma_inputs_fp8) CHECK_CUDA(cudaFree(args.per_expert_wmma_inputs_fp8));
    if (args.hidden_mlp_layer_1_out) CHECK_CUDA(cudaFree(args.hidden_mlp_layer_1_out));
    if (args.hidden_mlp_gate_out) CHECK_CUDA(cudaFree(args.hidden_mlp_gate_out));
    if (args.hidden_mlp_layer_1_out_fp16) CHECK_CUDA(cudaFree(args.hidden_mlp_layer_1_out_fp16));
    if (args.hidden_mlp_layer_1_out_int8) CHECK_CUDA(cudaFree(args.hidden_mlp_layer_1_out_int8));
    if (args.hidden_mlp_layer_1_out_int4) CHECK_CUDA(cudaFree(args.hidden_mlp_layer_1_out_int4));
    if (args.hidden_mlp_layer_1_out_fp8) CHECK_CUDA(cudaFree(args.hidden_mlp_layer_1_out_fp8));
    if (args.hidden_mlp_layer_2_out) CHECK_CUDA(cudaFree(args.hidden_mlp_layer_2_out));
    if (args.final_output) CHECK_CUDA(cudaFree(args.final_output));
}
