Swizzling — choosing the right strategy

Summary
-------
Swizzling (index permutation) is a lightweight technique to reduce shared-memory bank conflicts and improve DRAM access coalescing. Choose the swizzle that best matches (1) your access pattern, (2) the underlying bank layout, and (3) your performance/occupancy trade-offs.

1) Access‑pattern driven
------------------------
- When to use: workloads that perform systematic reordering in shared memory (e.g., tiling + transpose).
- Example: shared-memory transpose where threads write `tile[row][col]` and read `tile[col][row]`.
- Recommended swizzle: column skew / XOR on the column index (example: `col' = col ^ (row & 0x7)`).
- Rationale: this simple transform removes classic transpose bank conflicts while incurring only a few cheap integer operations.

2) Bank‑layout driven
---------------------
- When to use: when the hardware bank mapping causes many threads to target the same bank (common for strided accesses).
- Example: 32-bank SRAM where a warp reads contiguous 32-bit words but a stride maps multiple lanes to the same bank.
- Recommended swizzle: a permutation that decorrelates bank-id bits (example: `idx' = idx ^ (idx >> 5)` or a row-group XOR). 
- Rationale: directly addresses the bank-id bit positions, spreading lane accesses across banks and reducing serialization.

3) Objective‑driven (speed vs. cost)
-----------------------------------
- When to prefer minimal swizzling: latency‑critical kernels with tight register/shared‑memory budgets.
- Example policy: swizzle only one tensor (e.g., `B`) and leave the other (`A`) linear.
- Rationale: a minimal swizzle retains most conflict-reduction benefits while avoiding extra instruction overhead, register use, or shared-memory indexing complexity that can reduce occupancy.

Practical recommendation for this codebase (v2)
---------------------------------------------
- Experiment set: {no swizzle, XOR with `row & 1`, `row & 3`, `row & 7`, swizzle only `B`}.
- Evaluation criteria: prefer the variant that minimizes TLB/L1/L2 shared-excess metrics (e.g., Nsight Compute `L1 Wavefronts Shared Excessive`) and yields the best end-to-end latency (cudaEvent time).
