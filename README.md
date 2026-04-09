# Sparse Mixture-of-Experts (MoE)

A from-scratch CUDA implementation of a sparse Mixture-of-Experts inference kernel, progressively optimized through capacity-aware routing, PTX-level tensor-core intrinsics, shared-memory swizzling, and quantization (FP8/INT8/INT4). Each optimization step is profiled with Nsight Compute and documented with derived root-cause analysis — PTX intrinsics (`ldmatrix`, `mma.sync`, `cp.async`), roofline positioning, shared-memory bank conflict derivation, and quantization-aware operand packing.

**Hardware:** NVIDIA L4 (Ada Lovelace, SM 89). `fp4` is excluded — not supported on this GPU.

## Performance Highlights

All kernels use double-buffered WMMA tiles (`16×16×16`). **TC** = Tensor Cores; **CAP** = capacity-factor routing.

| Quant | CAP | Kernel | ms | Profile | Approach | Notes |
|----|----|----|------:|--------|-----------|---------|
| fp16 | No | [unfused](kernels/unfused.cu) | 2034 | [Run 1](prof/md/run1/ncu_details.md) | Per-stage global kernels | Over-alloc of per-expert buffers |
| fp16 | No | [baseline](kernels/baseline.cu) | 54 | [Run 1](prof/md/run1/ncu_details.md) | Fully fused with per-token routing | Over-alloc of per-expert buffers |
| fp16 | Yes | [capacity](kernels/capacity.cu) | 37.2 | [Run 1](prof/md/run1/ncu_details.md) | Capacity-aware routing | Reference baseline for all subsequent work |
| fp16 | Yes | [capacity_ldmatrix](kernels/capacity_ldmatrix.cu) | 34.5 | — | PTX `ldmatrix` + `mma.sync` | No explicit unswizzle required |
| fp16 | Yes | [swizzle_xor](kernels/swizzle_xor.cu) | 70 | [Run 2](prof/md/run2/ncu_details.md), [Run 3](prof/md/run3/ncu_details.md) | XOR swizzle + unswizzle | Instruction overhead exceeded bank-conflict savings |
| fp16 | Yes | [swizzle_ldmatrix](kernels/swizzle_ldmatrix.cu) | — | [Run 2](prof/md/run2/ncu_details.md) | Swizzle without unswizzle, PTX `ldmatrix` | Layout co-designed with `mma.sync` fragment |
| fp16 | Yes | [swizzle_autotune](kernels/swizzle_autotune.cu) | 45.5 | [Run 4](prof/md/run4/ncu_details.md) | Exhaustive row-mask search | Structurally impossible to resolve conflicts via swizzle |
| int8 | Yes | [capacity_int8](kernels/capacity_int8.cu) | 38.9 | — | INT8 via WMMA API | |
| int8 | Yes | [capacity_int8_ptx](kernels/capacity_int8_ptx.cu) | 30.6 | [Run 5](prof/md/run5/ncu_details.md) | PTX `mma.sync`, manual 4×int8→int32 pack | +19% vs FP16 |
| fp8 | Yes | [capacity_fp8_ptx](kernels/capacity_fp8_ptx.cu) | 38.7 | [Run 5](prof/md/run5/ncu_details.md) | PTX `mma.sync`, manual FP→INT pack | −5% vs FP16 (instruction explosion) |
| int4 | Yes | [capacity_int4_ptx](kernels/capacity_int4_ptx.cu) | 14 | [Run 5](prof/md/run5/ncu_details.md) | PTX `mma.sync`, nibble-packed B operands | **+2.6× vs FP16**; near compute-bound |

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

## Kernels and Important Files

**Kernel implementations** (`kernels/`):

| File | Description |
|---|---|
| `unfused.cu` | Per-stage global kernels — dispatch, expert GEMMs, combine as separate launches |
| `baseline.cu` | Fully fused single kernel with per-token routing |
| `capacity.cu` | Capacity-aware fused kernel — FP16 WMMA reference baseline |
| `capacity_ldmatrix.cu` | PTX `ldmatrix` + `mma.sync` without unswizzle overhead |
| `swizzle_xor.cu` | XOR-based shared-memory swizzle with `wmma` unswizzle pass |
| `swizzle_ldmatrix.cu` | Swizzle co-designed with `ldmatrix` consumer — no unswizzle required |
| `swizzle_autotune.cu` | Exhaustive row-mask search over all valid chunk-level swizzles |
| `capacity_int8.cu` | INT8 via WMMA C++ API |
| `capacity_int8_ptx.cu` | INT8 via PTX `mma.sync` with manual 4×int8→int32 operand packing |
| `capacity_fp8_ptx.cu` | FP8 via PTX `mma.sync` with manual FP→INT conversion |
| `capacity_int4_ptx.cu` | INT4 via PTX `mma.sync` with nibble-packed B-matrix operands |

**Supporting files:**
- Host-side allocation and data plumbing: `inputs/data.cu`, `inputs/data.h`
- Kernel argument structs: `include/moe_args.h`, `include/config.h`
- Driver entry point: `drivers/main.cu`
- Profiling notes and NCU outputs: `prof/txt/` and `prof/md/`

## Profiling

All profiling was done with **Nsight Compute** (`ncu`) on an NVIDIA L4 (Ada Lovelace, SM 89). Reports cover throughput, roofline, compute workload, memory workload, scheduler statistics, warp state statistics, and instruction statistics.

| Run | File | Kernels Covered |
|---|---|---|
| Run 1 | [prof/md/run1/ncu_details.md](prof/md/run1/ncu_details.md) | `unfused`, `baseline`, `capacity` |
| Run 2 | [prof/md/run2/ncu_details.md](prof/md/run2/ncu_details.md) | `swizzle_xor`, `swizzle_ldmatrix` (initial investigation) |
| Run 3 | [prof/md/run3/ncu_details.md](prof/md/run3/ncu_details.md) | `swizzle_xor` vs `capacity` (detailed analysis) |
| Run 4 | [prof/md/run4/ncu_details.md](prof/md/run4/ncu_details.md) | `swizzle_autotune` — robust swizzling framework |
| Run 5 | [prof/md/run5/ncu_details.md](prof/md/run5/ncu_details.md) | `capacity_fp8_ptx`, `capacity_int8_ptx`, `capacity_int4_ptx` — quantization study |

Full NCU capture command:
```bash
ncu --import-source yes --set full --export prof/ncu/capacity.ncu-rep ./bin/profile_capacity --kernel=capacity --warmup=5 --runs=10
```

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

## Potential Next Steps

- Quantization accuracy measurement: per-variant max absolute error / cosine similarity vs FP16 baseline.
- Grouped GEMM: concatenate expert buffers and run a batched/grouped GEMM, masking padded rows.
- Decode-path profiling (N=1, large `num_batches`) vs prefill (large N).
- Tile size and occupancy tuning for alternative hardware targets.

---