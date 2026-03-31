 # Run 2 — Nsight Compute: XOR swizzle

 Overview
 --------
This report summarizes the first application of swizzling implemented in [capacity_v2](kernels/capacity_v2.cu).

The XOR swizzling was applied around the `cp.async` staging path inside the tensor-core GEMM loop. 

Here we compare both [capacity](kernels/capacity.cu) and [capacity_v2](kernels/capacity_v2.cu) via a detailed Nsight Compute Analysis.

Key changes
-----------

1. Swizzle:
    - `cp.async` was modified to write tile fragments into a permuted (swizzled) shared-memory layout instead of the original linear layout.
	- The column index used for each `cp.async` destination was transformed using an XOR-based mapping. The mapping is applied at 16-byte chunk granularity to match the `cp.async` transaction size.

        for (int i = lane_id * 8; i < WMMA_M * WMMA_K; i += 32 * 8) {
            int row = i / WMMA_K;
            int col = i % WMMA_K;
            int col_chunk = col >> 3; // 8 half = 16B chunk
            int col_chunk_swz = col_chunk ^ (row & SWIZZLE_CHUNK_MASK);
            int col_swz = col_chunk_swz << 3;

            char* dst = (char*)&As[stage_buf][warp_id][row][col_swz];


2. Unswizzle step:
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

Example — XOR swizzle (16-byte chunks)
-------------------------------------
```cpp
// col: column index (element index within row)
// row: row index
// We operate on 16-byte chunks = groups of 8 elements (2 bytes each)
int chunk = col >> 3;                 // chunk index (col / 8)
int lane = col & 0x7;                 // position within chunk

// Simple per-row XOR swizzle: flip halves for odd rows
int chunk_swizzled = chunk ^ (row & 1);

// Reconstruct swizzled column index (in elements)
int col_swizzled = (chunk_swizzled << 3) | lane;

// --- inverse (unswizzle) ---
int chunk_unswizzled = chunk_swizzled ^ (row & 1);
int col_unswizzled = (chunk_unswizzled << 3) | lane;

// Use `col_swizzled` as the shared-memory destination index when writing
// and apply the inverse before calling WMMA load.
```


### What actually happened

Although the swizzle reduced some profiler-visible conflict symptoms, it required extra work:

- additional index arithmetic,
- extra shared-memory traffic for the unswizzle step,
- more instructions in the critical loop,
- and in some cases more register pressure.




