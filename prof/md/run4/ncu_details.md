# Robust Swizzling Framework

Kernels profiled: [swizzle_autotune.cu](kernels/swizzle_autotune.cu).

## TL;DR

- The 4-way shared-memory bank conflict in `capacity.cu` is caused by a row-mod-8 wraparound: every 8 lanes alias to the same bank span, so lanes `{0,8,16,24}`, `{1,9,17,25}`, … always collide.
- `cp.async` forces 16-byte transaction granularity. With `WMMA_K = 16` each row is exactly 32 bytes (two 16-byte chunks), leaving only one non-identity permutation: swapping the left and right halves of a row.
- A within-row swap does not change which rows collide with each other. The inter-row conflict structure is therefore immutable under any valid chunk-level swizzle.
- **Conclusion: it is not possible to eliminate or reduce the shared-memory bank conflicts in this kernel through swizzling.** The tile geometry and `cp.async` granularity together make the conflict pattern structurally unavoidable.

The goal of this run is to develop a structured swizzling framework rather than rely on ad hoc experimentation.

We begin by analyzing exactly how bank conflicts arise in [capacity.cu](kernels/capacity.cu), instead of iterating experimentally as in [Run 2](prof/md/run2/ncu_details.md) and [Run 3](prof/md/run3/ncu_details.md).

As the implementation baseline, we use [swizzle_ldmatrix.cu](kernels/swizzle_ldmatrix.cu). This path avoids the explicit unswizzle step required by the `wmma::load_matrix_sync` approach, which increased instruction count and degraded performance in [Run 3](prof/md/run3/ncu_details.md).

## Current thread mapping - how the bank conflicts occur

This section derives the current shared-memory access pattern in the base kernel and shows why recurring bank-collision groups appear.

### 1. Lane to tile coordinates

In the staging loop, each lane starts from:

$i = 8 \cdot l$, where $l \in [0,31]$ is the lane id.

For a $16 \times 16$ tile:

$row = \left\lfloor \frac{i}{16} \right\rfloor = \left\lfloor \frac{l}{2} \right\rfloor$

$col = i \bmod 16 = (8l) \bmod 16 \in \{0,8\}$

Interpretation: each lane writes one 16-byte segment (8 half elements) into either columns 0 to 7 or columns 8 to 15 of row $\left\lfloor l/2 \right\rfloor$. Even lanes write the first half of the row and odd lanes write the second half.

### 2. Shared-memory bank index

The linear element index at position (row, col) in a WMMA_K=16 column tile is: $index = 16 \cdot row + col$

Hence with half precision (2 bytes per element), byte address inside the tile is:

$addr = 2 \cdot (16 \cdot row + col)$ ($col \in \{0,8\}$)

SRAM has 32 banks, each serving 32-bit word per cycle, hence the bank index is computed at 4-byte bank granularity:

$bank = \left\lfloor \frac{addr}{4} \right\rfloor \bmod 32$

So the starting bank is:

- even lanes: $bank = (8 \cdot row) \bmod 32$
- odd lanes: $bank = (8 \cdot row + 4) \bmod 32$

Because in [capacity.cu](kernels/capacity.cu) each lane writes 16 bytes and each bank is 4 bytes wide, one lane touches four consecutive banks starting from that bank index.

Exact examples for the current kernel:

- Lane 0: $(row, col) = (0, 0)$, $addr = 0$, start bank $= 0$, banks touched $= 0$ to $3$
- Lane 1: $(row, col) = (0, 8)$, $addr = 16$, start bank $= 4$, banks touched $= 4$ to $7$
- Lane 2: $(row, col) = (1, 0)$, $addr = 32$, start bank $= 8$, banks touched $= 8$ to $11$
- Lane 3: $(row, col) = (1, 8)$, $addr = 48$, start bank $= 12$, banks touched $= 12$ to $15$
- Lane 4: $(row, col) = (2, 0)$, $addr = 64$, start bank $= 16$, banks touched $= 16$ to $19$
- Lane 5: $(row, col) = (2, 8)$, $addr = 80$, start bank $= 20$, banks touched $= 20$ to $23$
- Lane 6: $(row, col) = (3, 0)$, $addr = 96$, start bank $= 24$, banks touched $= 24$ to $27$
- Lane 7: $(row, col) = (3, 8)$, $addr = 112$, start bank $= 28$, banks touched $= 28$ to $31$
- Lane 8: $(row, col) = (4, 0)$, $addr = 128$, start bank $= 0$, banks touched $= 0$ to $3$
- Lane 9: $(row, col) = (4, 8)$, $addr = 144$, start bank $= 4$, banks touched $= 4$ to $7$
- Lane 10: $(row, col) = (5, 0)$, $addr = 160$, start bank $= 8$, banks touched $= 8$ to $11$
- Lane 11: $(row, col) = (5, 8)$, $addr = 176$, start bank $= 12$, banks touched $= 12$ to $15$
- Lane 12: $(row, col) = (6, 0)$, $addr = 192$, start bank $= 16$, banks touched $= 16$ to $19$
- Lane 13: $(row, col) = (6, 8)$, $addr = 208$, start bank $= 20$, banks touched $= 20$ to $23$
- Lane 14: $(row, col) = (7, 0)$, $addr = 224$, start bank $= 24$, banks touched $= 24$ to $27$
- Lane 15: $(row, col) = (7, 8)$, $addr = 240$, start bank $= 28$, banks touched $= 28$ to $31$

This already shows the wraparound: lane 8 returns to the same bank span as lane 0.

### 3. Exact collisions

Exact collision groups in the current kernel:

- Lanes $\{0, 8, 16, 24\}$ all hit banks $0$ to $3$
- Lanes $\{1, 9, 17, 25\}$ all hit banks $4$ to $7$
- Lanes $\{2, 10, 18, 26\}$ all hit banks $8$ to $11$
- Lanes $\{3, 11, 19, 27\}$ all hit banks $12$ to $15$
- Lanes $\{4, 12, 20, 28\}$ all hit banks $16$ to $19$
- Lanes $\{5, 13, 21, 29\}$ all hit banks $20$ to $23$
- Lanes $\{6, 14, 22, 30\}$ all hit banks $24$ to $27$
- Lanes $\{7, 15, 23, 31\}$ all hit banks $28$ to $31$

So the current mapping is a 4-way bank-conflict pattern.

## Swizzling — Framework

For this kernel, swizzling must respect the `cp.async` transaction size: one `cp.async` moves 16 bytes, which here is exactly 8 half values. With `WMMA_K = 16`, each row contains 16 half values = 32 bytes total, so each row is split into exactly two 16-byte chunks: columns 0 to 7 and columns 8 to 15.

That means the swizzle unit is the chunk, not individual half elements. As a result, for this tile shape the only non-identity chunk-level permutation is to swap the left and right half of the row. More complex intra-row swizzles would require breaking a 16-byte transaction, which is not compatible with this `cp.async` staging pattern.

## Autotuning plan

1. Represent the swizzle as a 16-bit row mask, where each bit selects whether that row swaps its left and right 16-byte halves.
2. Apply the swap on the `ldmatrix` consumer side, while keeping `cp.async` producer stores linear and coalesced.
3. Tune the row mask, starting from simple structured patterns and ranking candidates by duration.
4. Validate each strong candidate against the non-swizzled reference to confirm correctness.

## Conclusion

The autotuning investigation confirms the theoretical prediction above. Despite an exhaustive search over all 65 536 possible row-mask patterns, no swizzle configuration reduced bank conflicts or improved kernel duration relative to the unswizzled baseline.

The root cause is structural:

- Bank conflicts arise because the staging loop maps lanes `{0, 8, 16, 24}` to the same bank span, `{1, 9, 17, 25}` to the next, and so on — a period-8 wraparound over the 32 SRAM banks.
- The only degree of freedom available under the 16-byte `cp.async` constraint is a per-row left/right half swap. This permutes elements *within* a row but leaves the bank assignment of each row unchanged.
- Because the conflict pattern is determined by *which rows* different lanes access, and not by the intra-row ordering, no row-mask swizzle can break the collision groups.

Swizzling is therefore not a viable optimisation path for this kernel. The bank conflict is a consequence of the tile geometry (`WMMA_K = 16`, 16-byte transaction size, 32-lane warp) and cannot be resolved without changing one of those constraints — for example, by using a wider tile, a different staging granularity, or padding the shared-memory row stride.

### Alternative: Smaller `cp.async` Transaction Sizes

`cp.async` also supports 8-byte and 4-byte transactions. Dropping to 8-byte would split each row into 4 chunks instead of 2, and 4-byte into 8 chunks, unlocking finer-grained intra-row permutations that could in principle change the bank assignment of individual chunks and break the conflict groups.

However, this is not a viable path. Moving the same data volume requires 2× (8-byte) or 4× (4-byte) as many `cp.async` instructions. Run 4 already demonstrated that a sharp increase in instruction count — even when bank conflicts are partially resolved — shifts the dominant bottleneck from DRAM/shared-memory pressure to instruction-throughput and long scoreboard stalls, and resulted in a +90% regression in kernel duration. Multiplying `cp.async` count to fix a 4-way bank conflict would produce the same trade-off or worse, since the instruction overhead is a guaranteed cost while the conflict saving is bounded. This path would only be worth exploring if NCU were to show bank-conflict stalls as the clear dominant bottleneck, which is not the case here.

