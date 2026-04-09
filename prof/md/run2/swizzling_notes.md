# Swizzling Strategy Notes

## Summary
Swizzling (index permutation) reduces shared-memory bank conflicts and can improve memory coalescing.

You can swizzle rows, columns, or both. In most cases, avoid designs that require unswizzling because they add instructions, shared-memory traffic, and synchronization that can erase the gain.

Swizzling is not limited to DRAM -> SRAM copies. It can be applied to any layout mapping, including:
- DRAM -> SRAM (common for tiled loads)
- SRAM -> SRAM (reorder or transpose in shared memory)
- SRAM -> registers (for `ldmatrix`-compatible layouts)
- DRAM offline packing formats

## 1. Access-Pattern-Driven Design
- Use when: the kernel performs structured reordering in shared memory, such as tiling plus transpose.
- Example: threads write `tile[row][col]` and later read `tile[col][row]`.
- Candidate swizzle: column XOR, for example `col' = col ^ (row & 0x7)`.
- Why it helps: this often removes transpose-style bank conflicts with low integer-overhead cost.

Here in capacity.cu it doesn't make sense to swizzle the rows, however for wider tiles (e.g. WMMA_K=32+, multiple warps sharing a buffer) you can XOR the row index with chunk-derived bits to spread rows across banks too. The rule is: never permute rows unless you have a concrete bank-conflict reason to, because row permutation costs index arithmetic and is only useful when the row stride itself causes conflicts.

## 2. Bank-Layout-Driven Design
- Use when: many lanes map to the same SRAM bank because of stride or addressing pattern.
- Example: 32-bank shared memory where a stride collapses multiple lanes onto the same bank.
- Candidate swizzle: permutation that decorrelates bank-index bits, for example `idx' = idx ^ (idx >> 5)` or row-group XOR.
- Why it helps: spreads lane accesses across banks and reduces serialization.

## 3. Objective-Driven Design (Speed vs Cost)
- Use when: the kernel is latency-sensitive and register or shared-memory budget is tight.
- Practical policy: swizzle only one operand, for example `B`, and keep the other, `A`, linear.
- Why it helps: captures most conflict-reduction benefit while minimizing indexing overhead and occupancy pressure.

## Practical Recommendation for This Codebase (v2)
- Candidate set: no swizzle, XOR with `row & 1`, XOR with `row & 3`, XOR with `row & 7`, swizzle-only-`B`.
- Selection rule: choose the variant with the best end-to-end kernel time, and use Nsight metrics as diagnostics, for example shared-memory excess wavefronts.
