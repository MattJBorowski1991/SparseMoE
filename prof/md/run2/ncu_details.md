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

Define a chunk selector:

$col_{chunk} = l \bmod 2$

Then:

$col = 8 \cdot col_{chunk}$

Interpretation: each lane writes one 16-byte chunk (8 half elements) at row $\left\lfloor l/2 \right\rfloor$ and chunk $l \bmod 2$.

### 2. Shared-memory bank index

With half precision (2 bytes per element), byte address inside the tile is:

$addr_{bytes} = 2 \cdot (16 \cdot row + col)$

Substitute $col = 8 \cdot col_{chunk}$:

$addr_{bytes} = 32 \cdot row + 16 \cdot col_{chunk}$

With 4-byte bank granularity and 32 banks:

$bank = \left\lfloor \frac{addr_{bytes}}{4} \right\rfloor \bmod 32 = (8 \cdot row + 4 \cdot col_{chunk}) \bmod 32$

This is the bank-start formula for the 16-byte lane chunk.

### 3. Why collision groups repeat every 8 lanes

Compare lane $l$ and lane $l+8$:

- $col_{chunk}$ is unchanged (same parity)
- $row$ increases by 4

So the bank shift is:

$8 \cdot (row+4) + 4 \cdot col_{chunk} = (8 \cdot row + 4 \cdot col_{chunk}) + 32$

Modulo 32, that is identical. Therefore lanes in the set below map to the same bank pattern:

$\{l, l+8, l+16, l+24\}$

### 4. Concrete examples

Example A:

- Lane 0: $row=0$, $col_{chunk}=0$, $bank=(8\cdot0+4\cdot0)\bmod32=0$
- Lane 8: $row=4$, $col_{chunk}=0$, $bank=(8\cdot4+0)\bmod32=0$
- Lane 16: $row=8$, $col_{chunk}=0$, $bank=64\bmod32=0$
- Lane 24: $row=12$, $col_{chunk}=0$, $bank=96\bmod32=0$

Example B:

- Lane 1: $row=0$, $col_{chunk}=1$, $bank=(0+4)\bmod32=4$
- Lane 9: $row=4$, $col_{chunk}=1$, $bank=(32+4)\bmod32=4$
- Lane 17: $row=8$, $col_{chunk}=1$, $bank=(64+4)\bmod32=4$
- Lane 25: $row=12$, $col_{chunk}=1$, $bank=(96+4)\bmod32=4$

These repeated groups are the structural reason conflicts recur unless layout mapping is changed.

### 5. Practical swizzle implication

The swizzle should be designed from this exact lane to bank mapping, and validated at the consumer side as well (for example, shared-memory loads used by matrix instructions). A swizzle that improves producer mapping but adds extra shared-memory traffic or an unswizzle pass can still lose overall runtime.



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