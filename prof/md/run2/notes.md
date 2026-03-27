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