# Sparse Mixture-of-Experts (MoE) — Overview

A lightweight reference implementation of a sparse Mixture-of-Experts block targeting CUDA/Wmma tensor-core GEMMs. The repository contains three execution variants used for performance and correctness experiments:

- `unfused`: per-stage kernels, `CAP = N` (naïve, memory-heavy)
- `baseline`: fused kernels with per-token routing but still overallocated per-expert buffers
- `capacity`: capacity-aware routing that packs routed tokens into per-expert buffers of size `CAP` to enable larger, more efficient GEMMs

This README summarizes the block architecture, capacity semantics, and where to find the kernel implementations and profiling notes.

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
   - Each expert processes its packed buffer: `[m, d_model] -> up-proj -> activation -> down-proj -> [m, d_model]` where `m` ≤ `CAP`.
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
- Grouped GEMM: concatenate expert buffers and run a batched/grouped GEMM, masking out padded rows afterward.
- Tune tile sizes and shared-memory usage for better occupancy.

---

If you want, I can also:
- add a short Usage / CLI section with exact run commands, or
- create a one-slide summary (PNG/SVG) of the profiling comparison.
