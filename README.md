# Sparse Mixture-of-Experts (MoE)

Implementation of a sparse Mixture-of-Experts CUDA workflow. The repository contains multiple execution variants with detailed profiling analysis.

## Performance Highlights

All kernels are based on the double buffered wmma kernel with standard tile sizes `16x16x16`. The hardware I used (L4, Ada Lovelace, sm_89) does not support `fp4` hence it is excluded from the analysis.

**Tensor Cores** = **TC**; **CAP** = capacity factor for per-expert buffers; 

| Quant | CAP | Kernel | ms | Profile | Setup | Notes |
|----|----|----|-------|--------|-----------|---------|
| fp16 | No | [unfused](kernels/unfused.cu) | 2034 | [Run 1](prof/md/run1/ncu_details.md) | per-stage separate global kernels | overalloc of per-expert buffers |
| fp16 | No | [baseline](kernels/baseline.cu) | 54 | [Run 1](prof/md/run1/ncu_details.md) | full kernel fusion with per-token routing  | as above  |
| fp16 | Yes  | [capacity](kernels/capacity.cu) | 37.2 | [Run 1](prof/md/run1/ncu_details.md)  | capacity-aware routing |  |
| fp16 | Yes  | [capacity_ldmatrix](kernels/capacity_ldmatrix.cu) | 34.5 | [Run 2](prof/md/run2/ncu_details.md) | inject PTX  | ldmatrix, mma.sync |
| fp16 | Yes | [swizzle_xor](kernels/swizzle_xor.cu) | 70 | [Run 3](prof/md/run3/ncu_details.md), [Run 4](prof/md/run4/ncu_details.md)| Swizzle + Unswizzle |  |
| fp16 | Yes | [swizzle_ldmatrix](kernels/swizzle_ldmatrix.cu) | 45.5 | [Run 3](prof/md/run3/ncu_details.md), [Run 5](prof/md/run5/ncu_details.md) | Swizzle, no unswizzle, inject PTX | ldmatrix, mma.sync |
| fp16 | Yes | [swizzle_autotune](kernels/swizzle_autotune.cu) | 45.5 | [Run 6](prof/md/run6/ncu_details.md) | Swizzle: robust analysis | No perf improv possible |
| int8 | Yes | [capacity_int8](kernels/capacity_int8.cu) | 38.9 | [Run 7](prof/md/run7/ncu_details.md) | | |
| int8 | Yes | [capacity_int8_ptx](kernels/capacity_int8_ptx.cu) | 30.6 | [Run 7](prof/md/run7/ncu_details.md) | inject PTX | manual pack |
| fp8 | Yes | [capacity_fp8_ptx](kernels/capacity_fp8_ptx.cu) | 38.7 | [Run 7](prof/md/run7/ncu_details.md) | inject PTX | manual pack |
| int4 | Yes | [capacity_int4_ptx](kernels/capacity_int4_ptx.cu) | 14 | [Run 7](prof/md/run7/ncu_details.md) | inject PTX | manual pack |




## TL;DR

- Capacity-aware packing reduces DRAM footprint and enables grouped GEMMs.
- CAP = ceil(N * k / num_experts * capacity_factor), then rounded up to the nearest WMMA tile multiple.
- Overflow policy: drop excess routes beyond `CAP`.

## Architecture

Input shape: `[num_batches, N, d_model]` — tokens batched by sequence length `N` and model width `d_model`.

1. Router
   - Tiny linear projection per token: `[N, d_model] -> [N, num_experts]`.
2. Top-k selection
   - For each token select top-`k` experts and compute gated softmax weights: outputs are `indices [N, k]` and `weights [N, k]`.
3. Dispatch / Pack
   - Scatter selected tokens into contiguous per-expert buffers of shape `[num_experts, CAP, d_model]`.
   - `CAP` is computed per the formula below; empty slots are padded with zeros.
   - If an expert receives more than `CAP` assignments, extra routes are dropped.
4. Expert compute
   - Each expert processes its packed buffer: `[m, d_model] -> (up-proj, gate-proj) -> SwiGLU -> down-proj -> [m, d_model]` where `m` ≤ `CAP`.
   - Implementations use WMMA-friendly tiling; matrices are padded to WMMA tile sizes.
5. Combine
   - Multiply expert outputs by their gating weights and scatter-accumulate back into `[N, d_model]`.

## Capacity (CAP) details

Compute CAP (per expert):

```
CAP_raw = ceil(N * k / num_experts * capacity_factor)
CAP = ceil(CAP_raw / WMMA_M) * WMMA_M  // round up to WMMA_M multiple
```

Notes:
- Padding: slots between `m` (actual routed tokens) and `CAP` are zeroed before GEMMs so padded rows do not affect results.
- Overflow: any route that maps to a slot ≥ `CAP` is dropped. To avoid out-of-bounds reads, the implementation clamps `expert_counts[e] = min(expert_counts[e], CAP)` after packing.

## Kernels and important files

- Kernel implementations: `kernels/unfused.cu`, `kernels/baseline.cu`, `kernels/capacity.cu`.
- Host-side allocation and data plumbing: `inputs/data.cu`, `inputs/data.h`.
- Driver entry point: `drivers/main.cu`.
- Profiling notes and NCU outputs: `prof/txt/` and `prof/md/`.

## Profiling

Run the provided experiments (see `Makefile`) and collect Nsight Compute (`ncu`) reports for the `baseline` and `capacity` variants. The `prof/md/run1/notes.md` file contains a concise comparison of an example run that demonstrates the performance and memory throughput differences.

## Building & running

Quick start (Linux, CUDA enabled):

```bash
make clean && make KERNEL=baseline
# then run the produced binary in bin/ (project-specific runner)
```

Full NCU profile:
```bash
ncu --import-source yes --set full --export prof/ncu/capacity.ncu-rep ./bin/profile_capacity --kernel=capacity --warmup=5 --runs=10
```

If your GPU memory is limited, try the `capacity` variant which reduces per-expert allocations.

## Next steps / ideas

- Per-kernel Nsight Compute analysis to identify cache/DRAM bottlenecks.
- Profile both for prefill (large N) and decode (N=1)
- Different quantizations with accuracy measurement
- Grouped GEMM: concatenate expert buffers and run a batched/grouped GEMM, masking out padded rows afterward.
- Tune tile sizes and shared-memory usage for better occupancy.

---