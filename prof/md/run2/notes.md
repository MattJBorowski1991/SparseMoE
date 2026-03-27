Swizzling - how to choose best way:

You choose swizzle by:

Access pattern driven

Example: matrix transpose tile in shared memory (tile[row][col] write, tile[col][row] read).
Best swizzle: column skew/XOR on column index, e.g. col' = col ^ (row & 0x7).
Why best: removes classic transpose bank conflicts with tiny arithmetic overhead.

Bank-layout driven

Example: 32-bank SRAM, warp reads contiguous 32-bit words but stride causes many lanes to hit same bank.
Best swizzle: permutation matched to bank bits, e.g. idx' = idx ^ (idx >> 5) (or row-group XOR) so bank-id bits are decorrelated.
Why best: directly targets bank mapping hardware, maximizing bank spread.

Objective driven (speed vs cost)

Example: latency-critical kernel with high register pressure.
Best swizzle: minimal swizzle (or only swizzle one tensor), e.g. swizzle only B, leave A linear.
Why best: captures most conflict reduction while avoiding extra instructions/register/shared overhead that can hurt occupancy.

Best method for our app in v2: try a small set (no swizzle, XOR with row&1/3/7, maybe only on B), then keep the one with lowest L1 Wavefronts Shared Excessive and best cudaEvent time.


## First swizzling attempt in `capacity_v2`

The first concrete swizzling experiment in `capacity_v2` was a shared-memory XOR swizzle applied around the `cp.async` staging path inside the tensor-core GEMM loop.

### What was changed

- `cp.async` no longer wrote tiles into a plain linear shared-memory layout.
- Instead, the destination column index was permuted with an XOR-based mapping.
- Because `cp.async` writes 16-byte chunks, the final version of this experiment swizzled at 16-byte chunk granularity rather than per-element granularity.
- Before `wmma::load_matrix_sync`, the staged tile was unswizzled back into the linear layout expected by the WMMA API.

### Why it seemed promising

The goal was to reduce shared-memory bank conflicts and replay pressure reported by Nsight Compute near the shared-memory load path. In principle, distributing accesses more evenly across banks can reduce `L1 Wavefronts Shared Excessive` and improve the operand feed into tensor-core instructions.

### Why it was supposed to help

The expected mechanism was:

1. `cp.async` writes land in a bank-friendlier permuted shared-memory layout.
2. Shared-memory bank conflicts decrease during staging-related access.
3. Tensor-core operand loading becomes cleaner, reducing replay overhead.

### What actually happened

Although the swizzle reduced some profiler-visible conflict symptoms, it required extra work:

- additional index arithmetic,
- extra shared-memory traffic for the unswizzle step,
- more instructions in the critical loop,
- and in some cases more register pressure.

In practice, the real `cudaEventRecord` runtime got worse even when some Nsight Compute metrics looked better.

### Takeaway

This first swizzling attempt was a valid experiment, but in this kernel the overhead of swizzle + unswizzle outweighed the benefit. That result motivated the later `ldmatrix + mma.sync` experiment, which tried to avoid the explicit unswizzle step entirely.



