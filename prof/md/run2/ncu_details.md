 # Run 2 — Nsight Compute: XOR swizzle

 Overview
 --------
This report summarizes the first application of swizzling implemented in [capacity_v2](kernels/capacity_v2.cu).

The XOR swizzling was applied around the `cp.async` staging path inside the tensor-core GEMM loop. 

Here we compare both [capacity](kernels/capacity.cu) and [capacity_v2](kernels/capacity_v2.cu) via a detailed Nsight Compute Analysis.


### Implementation

Overview
--------
This experiment applies an XOR-based swizzle to the shared-memory staging path used by `cp.async` in the tensor-core GEMM loop. The goal is to reduce shared-memory bank conflicts and improve global memory access coalescing for the tiled loads.

Key changes
-----------
1. Staging layout:
	- `cp.async` was modified to write tile fragments into a permuted (swizzled) shared-memory layout instead of the original linear layout.
2. Swizzle mapping:
	- The column index used for each `cp.async` destination was transformed using an XOR-based mapping. The mapping is applied at 16-byte chunk granularity to match the `cp.async` transaction size.
3. Unsizzle step:
	- Prior to invoking `wmma::load_matrix_sync`, the swizzled tile in shared memory is restored (unswizzled) to the linear layout expected by the WMMA API.

Implementation notes
--------------------
- Granularity: swizzling at 16‑byte chunk granularity aligns with `cp.async` transfer granularity, reducing bookkeeping overhead compared with per-element permutes.
- Cost model: the transformation adds lightweight index arithmetic on the hot path and an explicit unswizzle copy step before the WMMA load; both increase instruction count and shared-memory traffic.
- Correctness: the unswizzle restores the original ordering required by `wmma::load_matrix_sync`, preserving semantic equivalence with the baseline GEMM.

The XOR swizzle implemented in this experiment operates at 16‑byte chunk granularity (groups of eight 2‑byte elements). In this pattern odd rows swap the two eight‑chunk halves; an illustrative mapping follows:

| Row | Chunks 0–7 | Chunks 8–15 |
|---:|:---|:---|
| 0 | 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 |
| 1 | 8 9 10 11 12 13 14 15 | 0 1 2 3 4 5 6 7 |
| 2 | 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 |
| 3 | 8 9 10 11 12 13 14 15 | 0 1 2 3 4 5 6 7 |

This compact representation highlights the per‑row XOR permutation used for the shared‑memory tile layout.


### What actually happened

Although the swizzle reduced some profiler-visible conflict symptoms, it required extra work:

- additional index arithmetic,
- extra shared-memory traffic for the unswizzle step,
- more instructions in the critical loop,
- and in some cases more register pressure.




