# Run 2 — Nsight Compute: Swizzling

Kernels profiled: [capacity.cu](kernels/capacity.cu), [capacity_v2.cu](kernels/capacity_v2.cu) and [capacity_v3.cu](kernels/capacity_v3.cu).

## Overview
This report summarizes the two applications of swizzling below and compares them to the base version [capacity.cu](kernels/capacity.cu):
- [capacity_v2](kernels/capacity_v2.cu) - XOR swizzle
- [capacity_v3](kernels/capacity_v3.cu) - swizzle via `ldmatrix.sync.aligned` and `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` 

Both swizzling approaches / thread mapping permutations were applied at 16-byte chunk granularity to match the `cp.async` transaction size and reduce bookkeeping overhead. The aim of this file is to document and explain why neither of the applications yielded performance improvement.

## Current thread mapping - how the bank conflicts occur

This section derives the current shared-memory access pattern in the base kernel and shows why recurring bank-collision groups appear.

### 1. Lane to tile coordinates

In the staging loop, each lane starts from:

$i = 8 \cdot l$, where $l \in [0,31]$ is the lane id.

For a $16 \times 16$ tile:

$row = \left\lfloor \frac{i}{16} \right\rfloor = \left\lfloor \frac{l}{2} \right\rfloor$

$col = i \bmod 16 = (8l) \bmod 16 \in \{0,8\}$

Interpretation: each lane writes one 16-byte segment (8 half elements) into either columns 0 to 7 or columns 8 to 15 of row $\left\lfloor l/2 \right\rfloor$. Even lanes write the first half of the row and odd lanes write the second half.

Exact mapping for the current kernel:

- Lane 0 $\rightarrow$ $(row, col) = (0, 0)$
- Lane 1 $\rightarrow$ $(row, col) = (0, 8)$
- Lane 2 $\rightarrow$ $(row, col) = (1, 0)$
- Lane 3 $\rightarrow$ $(row, col) = (1, 8)$
- Lane 4 $\rightarrow$ $(row, col) = (2, 0)$
- Lane 5 $\rightarrow$ $(row, col) = (2, 8)$
- Lane 6 $\rightarrow$ $(row, col) = (3, 0)$
- Lane 7 $\rightarrow$ $(row, col) = (3, 8)$
- Lane 8 $\rightarrow$ $(row, col) = (4, 0)$
- Lane 9 $\rightarrow$ $(row, col) = (4, 8)$
- Lane 10 $\rightarrow$ $(row, col) = (5, 0)$
- Lane 11 $\rightarrow$ $(row, col) = (5, 8)$
- Lane 12 $\rightarrow$ $(row, col) = (6, 0)$
- Lane 13 $\rightarrow$ $(row, col) = (6, 8)$
- Lane 14 $\rightarrow$ $(row, col) = (7, 0)$
- Lane 15 $\rightarrow$ $(row, col) = (7, 8)$

The same pattern continues for lanes 16 to 31:

- Lane 16 $\rightarrow$ $(8, 0)$, Lane 17 $\rightarrow$ $(8, 8)$
- Lane 18 $\rightarrow$ $(9, 0)$, Lane 19 $\rightarrow$ $(9, 8)$
- Lane 20 $\rightarrow$ $(10, 0)$, Lane 21 $\rightarrow$ $(10, 8)$
- Lane 22 $\rightarrow$ $(11, 0)$, Lane 23 $\rightarrow$ $(11, 8)$
- Lane 24 $\rightarrow$ $(12, 0)$, Lane 25 $\rightarrow$ $(12, 8)$
- Lane 26 $\rightarrow$ $(13, 0)$, Lane 27 $\rightarrow$ $(13, 8)$
- Lane 28 $\rightarrow$ $(14, 0)$, Lane 29 $\rightarrow$ $(14, 8)$
- Lane 30 $\rightarrow$ $(15, 0)$, Lane 31 $\rightarrow$ $(15, 8)$

### 2. Shared-memory bank index

With half precision (2 bytes per element), byte address inside the tile is:

$addr_{bytes} = 2 \cdot (16 \cdot row + col)$

Since $col \in \{0,8\}$, this becomes:

- even lanes: $addr_{bytes} = 32 \cdot row$
- odd lanes: $addr_{bytes} = 32 \cdot row + 16$

With 4-byte bank granularity and 32 banks:

$bank = \left\lfloor \frac{addr_{bytes}}{4} \right\rfloor \bmod 32$

So the starting bank is:

- even lanes: $bank = (8 \cdot row) \bmod 32$
- odd lanes: $bank = (8 \cdot row + 4) \bmod 32$

Because each lane writes 16 bytes and each bank is 4 bytes wide, one lane touches four consecutive banks starting from that bank index.

Exact examples for the current kernel:

- Lane 0: $(row, col) = (0, 0)$, $addr_{bytes} = 2 \cdot (16 \cdot 0 + 0) = 0$, start bank $= 0$, banks touched $= 0$ to $3$
- Lane 1: $(row, col) = (0, 8)$, $addr_{bytes} = 2 \cdot (16 \cdot 0 + 8) = 16$, start bank $= 4$, banks touched $= 4$ to $7$
- Lane 2: $(row, col) = (1, 0)$, $addr_{bytes} = 32$, start bank $= 8$, banks touched $= 8$ to $11$
- Lane 3: $(row, col) = (1, 8)$, $addr_{bytes} = 48$, start bank $= 12$, banks touched $= 12$ to $15$
- Lane 4: $(row, col) = (2, 0)$, $addr_{bytes} = 64$, start bank $= 16$, banks touched $= 16$ to $19$
- Lane 5: $(row, col) = (2, 8)$, $addr_{bytes} = 80$, start bank $= 20$, banks touched $= 20$ to $23$
- Lane 6: $(row, col) = (3, 0)$, $addr_{bytes} = 96$, start bank $= 24$, banks touched $= 24$ to $27$
- Lane 7: $(row, col) = (3, 8)$, $addr_{bytes} = 112$, start bank $= 28$, banks touched $= 28$ to $31$
- Lane 8: $(row, col) = (4, 0)$, $addr_{bytes} = 128$, start bank $= 0$, banks touched $= 0$ to $3$

This already shows the wraparound: lane 8 returns to the same bank span as lane 0.

### 3. Why collision groups repeat every 8 lanes

Compare lane $l$ and lane $l+8$:

- lane parity is unchanged, so both lanes target the same half-row
- $row$ increases by 4

So the bank shift is:

$(8 \cdot (row+4)) \bmod 32 = (8 \cdot row + 32) \bmod 32 = (8 \cdot row) \bmod 32$

and similarly for odd lanes:

$(8 \cdot (row+4) + 4) \bmod 32 = (8 \cdot row + 36) \bmod 32 = (8 \cdot row + 4) \bmod 32$

Therefore lanes in the set below map to the same starting bank and the same four-bank span:

$\{l, l+8, l+16, l+24\}$

When multiple lanes in one warp hit the same banks in the same instruction, those accesses are replayed/serialized instead of being served fully in parallel.

Exact collision groups in the current kernel:

- Lanes $\{0, 8, 16, 24\}$ all hit banks $0$ to $3$
- Lanes $\{1, 9, 17, 25\}$ all hit banks $4$ to $7$
- Lanes $\{2, 10, 18, 26\}$ all hit banks $8$ to $11$
- Lanes $\{3, 11, 19, 27\}$ all hit banks $12$ to $15$
- Lanes $\{4, 12, 20, 28\}$ all hit banks $16$ to $19$
- Lanes $\{5, 13, 21, 29\}$ all hit banks $20$ to $23$
- Lanes $\{6, 14, 22, 30\}$ all hit banks $24$ to $27$
- Lanes $\{7, 15, 23, 31\}$ all hit banks $28$ to $31$

So the current mapping creates eight recurring 4-lane collision groups across the warp.

### 4. Concrete examples

Example A:

- Lane 0: $row=0$, $col=0$, starts at bank $0$
- Lane 8: $row=4$, $col=0$, starts at bank $0$
- Lane 16: $row=8$, $col=0$, starts at bank $0$
- Lane 24: $row=12$, $col=0$, starts at bank $0$

Each of these lanes writes 16 bytes, so they all touch banks 0 to 3.

Example B:

- Lane 1: $row=0$, $col=8$, starts at bank $4$
- Lane 9: $row=4$, $col=8$, starts at bank $4$
- Lane 17: $row=8$, $col=8$, starts at bank $4$
- Lane 25: $row=12$, $col=8$, starts at bank $4$

Each of these lanes writes 16 bytes, so they all touch banks 4 to 7.

These repeated groups are the structural reason the same bank conflicts recur in the current layout.



## XOR swizzle

```cpp
        for (int i = lane_id * 8; i < WMMA_M * WMMA_K; i += 32 * 8) {
            int row = i / WMMA_K;
            int col = i % WMMA_K;
            int col_chunk = col >> 3; // 8 half = 16B chunk
            int col_chunk_swz = col_chunk ^ (row & SWIZZLE_CHUNK_MASK);
            int col_swz = col_chunk_swz << 3;

            char* dst = (char*)&As[stage_buf][warp_id][row][col_swz];
```

Subsequently, the unswizzle step is applied prior to invoking `wmma::load_matrix_sync`, the swizzled tile in shared memory is restored (unswizzled) to the linear layout expected by the WMMA API.

The transformations (swizzle + unswizzle) add index arithmetic and increase instruction count.


The XOR swizzle implemented in this experiment operates at 16‑byte chunk granularity (groups of eight 2‑byte elements). In this pattern odd rows swap the two eight‑chunk halves; an illustrative mapping follows:

| Row | Chunks 0–7 | Chunks 8–15 |
|---:|:---|:---|
| 0 | 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 |
| 1 | 8 9 10 11 12 13 14 15 | 0 1 2 3 4 5 6 7 |
| 2 | 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 |
| 3 | 8 9 10 11 12 13 14 15 | 0 1 2 3 4 5 6 7 |

This compact representation highlights the per‑row XOR permutation used for the shared‑memory tile layout.

### Example — XOR swizzle (16-byte chunks)
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

## ldmatrix swizzle

Replace `wmma::load_matrix_sync` and `wmma::mma_sync` with a swizzle using a PTX approach:

1. `cp.async` stages tiles into shared memory in a simple linear layout.
2. `ldmatrix` loads warp fragments directly from shared memory into registers.
3. `mma.sync` consumes those registers directly for tensor-core MMA operations.

My intention of this approach was to move closer to the layout strategy used in low-level libraries such as CUTLASS.

```cpp
static __device__ __forceinline__ void ldmatrix_a_m16n8k16(
    const half* tile,
    int lane_id,
    int ld,
    unsigned (&a)[4]
)
{
    // Matrix order expected by mma.sync: top-left, bottom-left, top-right, bottom-right.
    int group = lane_id >> 3;
    int row = (lane_id & 7) + ((group & 1) << 3);
    int col = (group >> 1) << 3;
    unsigned addr = smem_addr(tile + row * ld + col);
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "r"(addr));
}

static __device__ __forceinline__ void ldmatrix_b_m16n8k16(
    const half* tile,
    int lane_id,
    int ld,
    int col_block,
    unsigned (&b)[2]
)
{
    // For .x2, threads 16-31 can reuse the lower-thread addresses.
    int group = (lane_id >> 3) & 1;
    int row = (lane_id & 7) + (group << 3);
    unsigned addr = smem_addr(tile + row * ld + col_block);
    asm volatile(
        "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(b[0]), "=r"(b[1])
        : "r"(addr));
}

static __device__ __forceinline__ void mma_m16n8k16_f32(
    const unsigned (&a)[4],
    const unsigned (&b)[2],
    float (&c)[4]
)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
```

## NCU — High-level results

Summary: both swizzling attempts resulted in deterioration of perfomance due to the decrease in Memory & DRAM thoughput. 

### GPU Speed Of Light Throughput

> **Comment:**

| Metric Name | Metric Unit | capacity | capacity_v2 | capacity_v3 |
|---|---|---:|---:|---:|
| DRAM Frequency | GHz | 6.24 | 6.24 | 6.24 |
| SM Frequency | MHz | 804.67 | 822.95 | 803.42 |
| Elapsed Cycles | cycle | 30063030 | 57658205 | 37175334 |
| Memory Throughput | % | 86.24 | 67.62 | 79.42 |
| DRAM Throughput | % | 86.24 | 61.07 | 79.42 |
| Duration | ms | 36.97 | 69.37 | 45.81 |
| L1/TEX Cache Throughput | % | 84.05 | 70.02 | 52.05 |
| L2 Cache Throughput | % | 32.88 | 44.36 | 62.63 |
| SM Active Cycles | cycle | 29320941.95 | 55109655.78 | 35857617.72 |
| Compute (SM) Throughput | % | 33.63 | 67.62 | 29.40 |

Comments:
- capacity: INF This workload is utilizing greater than 80.0% of the available compute or memory performance of the device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing DRAM in the Memory Workload Analysis section.
- capacity_v2: INF Compute and Memory are well-balanced: To reduce runtime, both computation and memory traffic must be reduced. Check both the Compute Workload Analysis and Memory Workload Analysis sections.
- capacity_v3: OPT Memory is more heavily utilized than Compute: Look at the Memory Workload Analysis section to identify the DRAM bottleneck. Check memory replay (coalescing) metrics to make sure you're efficiently utilizing the bytes transferred. Also consider whether it is possible to do more work per memory access (kernel fusion) or whether there are values you can (re)compute.

### Launch Statistics

> **Comment:**

| Metric Name | Metric Unit | capacity | capacity_v2 | capacity_v3 |
|---|---|---:|---:|---:|
| Block Size |  | 256 | 256 | 256 |
| Function Cache Configuration |  | CachePreferNone | CachePreferNone | CachePreferNone |
| Grid Size |  | 2048 | 2048 | 2048 |
| Registers Per Thread | register/thread | 72 | 80 | 72 |
| Shared Memory Configuration Size | KiB | 102.40 | 102.40 | 102.40 |
| Driver Shared Memory Per Block | KiB/block | 1.02 | 1.02 | 1.02 |
| Dynamic Shared Memory Per Block | byte/block | 0 | 0 | 0 |
| Static Shared Memory Per Block | KiB/block | 33.02 | 32.77 | 32.77 |
| # SMs | SM | 58 | 58 | 58 |
| Stack Size |  | 1024 | 1024 | 1024 |
| Threads | thread | 524288 | 524288 | 524288 |
| # TPCs |  | 29 | 29 | 29 |
| Enabled TPC IDs |  | all | all | all |
| Uses Green Context |  | 0 | 0 | 0 |
| Waves Per SM |  | 11.77 | 11.77 | 11.77 |

## Occupancy

| Metric Name | Metric Unit | capacity | capacity_v2 | capacity_v3 |
|---|---|---:|---:|---:|
| Block Limit SM | block | 24 | 24 | 24 |
| Block Limit Registers | block | 3 | 3 | 3 |
| Block Limit Shared Mem | block | 3 | 3 | 3 |
| Block Limit Warps | block | 6 | 6 | 6 |
| Theoretical Active Warps per SM | warp | 24 | 24 | 24 |
| Theoretical Occupancy | % | 50 | 50 | 50 |
| Achieved Occupancy | % | 49.19 | 48.99 | 48.93 |
| Achieved Active Warps Per SM | warp | 23.61 | 23.51 | 23.49 |

Comments (same for all):
- OPT Est. Local Speedup: 50%. The 6.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (50.0%) is limited by the number of required registers, and the required amount of shared memory.

## GPU and Memory Workload Distribution

| Metric Name | Metric Unit | capacity % | capacity | capacity_v2 % | capacity_v2 | capacity_v3 % | capacity_v3 |
|---|---|---:|---:|---:|---:|---:|---:|
| Average DRAM Active Cycles | cycle | 28.9% | 199,112,960 | 38.3% | 264,573,200 | 32.9% | 227,192,048 |
| Average L1 Active Cycles | cycle | 24.4% | 29,320,942 | 45.9% | 55,109,656 | 29.9% | 35,857,618 |
| Average L2 Active Cycles | cycle | 24.5% | 30,315,918 | 45.4% | 56,303,511 | 30.3% | 37,590,135 |
| Average SM Active Cycles | cycle | 24.4% | 29,320,942 | 45.9% | 55,109,656 | 29.9% | 35,857,618 |
| Average SMSP Active Cycles | cycle | 24.4% | 29,318,293 | 45.9% | 55,079,822 | 29.9% | 35,844,260 |
| Total DRAM Elapsed Cycles | cycle | 24.3% | 1,385,345,024 | 45.6% | 2,599,341,056 | 30.2% | 1,716,303,872 |
| Total L1 Elapsed Cycles | cycle | 24.1% | 1,723,078,788 | 46.2% | 3,309,925,018 | 29.9% | 2,141,142,518 |
| Total L2 Elapsed Cycles | cycle | 24.3% | 732,076,992 | 45.7% | 1,378,815,744 | 30.1% | 906,970,200 |
| Total SM Elapsed Cycles | cycle | 24.1% | 1,723,078,788 | 46.2% | 3,309,925,018 | 29.9% | 2,141,142,518 |
| Total SMSP Elapsed Cycles | cycle | 24.1% | 6,892,315,152 | 46.2% | 13,239,700,072 | 29.9% | 8,564,570,072 |
| **Sum of Total Elapsed** | **cycle** | **24.1%** | **12,455,894,744** | **46.1%** | **23,837,706,908** | **29.9%** | **15,470,129,180** |

Note:
- The ratio between `Total ... Elapsed Cycles` and `Average ... Active Cycles` is **not expected to be identical** across DRAM, L1, L2, SM, and SMSP.
- For each subsystem, the `Average ... Active Cycles` value is averaged over the profiled kernel passes/instances collected in that Nsight run.
- The percentage columns on `Average ... Active Cycles` are relative split across kernels for that row (informational), while `Total ... Elapsed Cycles` percentages represent contribution to total elapsed cycles for that subsystem.
- These counters are collected at different hardware scopes and with different aggregation semantics, so cross-subsystem ratios should not be compared as if they shared one common denominator.