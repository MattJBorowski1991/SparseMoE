 # Run 1 — Nsight Compute: Summary

 **Date:** 2026-03-30

 Overview
 --------
This report summarizes high level profiling outputs for: [unfused.cu](kernels/unfused.cu), [baseline.cu](kernels/baseline.cu) and [capacity.cu](kernels/capacity.cu). The [unfused.cu](kernels/unfused.cu) variant produces separate kernel traces and is not included in the aggregated tables below. The timings reported for the three workflows ([unfused.cu](kernels/unfused.cu), [baseline.cu](kernels/baseline.cu), and [capacity.cu](kernels/capacity.cu)) use a small configuration — larger configurations cause the device to run out of memory for the [unfused.cu](kernels/unfused.cu) variant.

 Configuration
 -------------
 ```cpp
 constexpr int N = 512;
 constexpr int d_model = 4096;
 constexpr int num_batches = 4;
 constexpr int num_experts = 32;
 constexpr int up_proj_dim = 4;
 ```

 Executive Summary
 -----------------
 - **Unfused (aggregated):** 2780 ms - WMMA `up_proj` / `down_proj` kernels dominate `99%` of runtime (per-kernel traces).
 - **Baseline:** 85.6 ms — elevated DRAM utilization and replay indicate potential locality and coalescing issues.
 - **Capacity:** 61 ms — demonstrates improved runtime due to more efficient per-expert buffering and a better compute/memory balance.

Observations
------------
- **Kernel Fusion:** Delivers substantial benefits in this workload; prioritize reducing redundant memory traffic and improving data locality. Measured improvement for the fused implementation is approximately **32×** over the unfused workflow.
- **Per‑Expert Allocation:** Capacity-aware buffer sizing materially improves performance compared with naive over-allocation — the `capacity` variant shows roughly **+40%** speedup versus `baseline` under this configuration.
- **Primary Hotspots:** WMMA `up_proj` and `down_proj` kernels dominate the unfused runtime and should be the first optimization targets.
- **Memory vs Compute:** The `baseline` variant exhibits DRAM-bound behavior; the `capacity` variant shifts the workload toward better compute utilization.

## GPU Speed Of Light Throughput

> <u>Author's note:</u> Crucial metrics all increased: Memory Throughput, DRAM Throughput, Compute Throughput, and both cache throughputs.

| Metric Name | Metric Unit | baseline | capacity |
|---|---:|---:|---:|
| DRAM Frequency | Ghz | 6.24 | 6.24 |
| SM Frequency | Mhz | 795.00 | 795.00 |
| Elapsed Cycles | cycle | 68,099,844 | 48,492,117 |
| Memory Throughput | % | 76.38 | 84.91 |
| DRAM Throughput | % | 76.38 | 78.78 |
| Duration | ms | 85.66 | 61.00 |
| L1/TEX Cache Throughput | % | 62.98 | 85.81 |
| L2 Cache Throughput | % | 29.56 | 33.27 |
| SM Active Cycles | cycle | 67,272,033.71 | 47,972,319.97 |
| Compute (SM) Throughput | % | 27.95 | 34.58 |

**Comments from NCU:**

- `capacity` (INF): "This workload is utilizing greater than 80.0% of the available compute or memory performance of the device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section." (from `ncu_capacity.txt`)
- `baseline` (OPT): "Memory is more heavily utilized than Compute: Look at the Memory Workload Analysis section to identify the DRAM bottleneck. Check memory replay (coalescing) metrics to make sure you're efficiently utilizing the bytes transferred. Also consider whether it is possible to do more work per memory access (kernel fusion) or whether there are values you can (re)compute." (from `ncu_baseline.txt`)


## Launch Statistics

> <u>Author's note:</u> No change in Launch Statisticts other than slight increase in Register pressure.

| Metric Name | Metric Unit | baseline | capacity |
|---|---:|---:|---:|
| Block Size |  | 256 | 256 |
| Function Cache Configuration |  | CachePreferNone | CachePreferNone |
| Grid Size |  | 4096 | 4096 |
| Registers Per Thread | register/thread | 70 | 72 |
| Shared Memory Configuration Size | Kbyte | 102.40 | 102.40 |
| Driver Shared Memory Per Block | Kbyte/block | 1.02 | 1.02 |
| Dynamic Shared Memory Per Block | byte/block | 0 | 0 |
| Static Shared Memory Per Block | Kbyte/block | 33.02 | 33.02 |
| # SMs | SM | 58 | 58 |
| Stack Size |  | 1024 | 1024 |
| Threads | thread | 1,048,576 | 1,048,576 |
| # TPCs |  | 29 | 29 |
| Enabled TPC IDs |  | all | all |
| Uses Green Context |  | 0 | 0 |
| Waves Per SM |  | 23.54 | 23.54 |

## Occupancy

> <u>Author's note:</u> Restrictive limits on block count in both kernels due to SRAM and Register pressure..

| Metric Name | Metric Unit | baseline | capacity |
|---|---:|---:|---:|
| Block Limit SM | block | 24 | 24 |
| Block Limit Registers | block | 3 | 3 |
| Block Limit Shared Mem | block | 3 | 3 |
| Block Limit Warps | block | 6 | 6 |
| Theoretical Active Warps per SM | warp | 24 | 24 |
| Theoretical Occupancy | % | 50 | 50 |
| Achieved Occupancy | % | 49.58 | 49.69 |
| Achieved Active Warps Per SM | warp | 23.80 | 23.85 |

**Comments:**

- `capacity` (OPT): "Est. Local Speedup: 50% ... The 6.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (50.0%) is limited by the number of required registers, and the required amount of shared memory." (from `ncu_capacity.txt`)
- `baseline` (OPT): same note present in `ncu_baseline.txt`.


## GPU and Memory Workload Distribution

| Metric Name | Metric Unit | baseline | capacity | % change |
|---|---:|---:|---:|---:|
| Average DRAM Active Cycles | cycle | 408,562,912 | 300,068,704 | -26.6% |
| Total DRAM Elapsed Cycles | cycle | 3,209,560,064 | 2,285,443,072 | -28.8% |
| Average L1 Active Cycles | cycle | 67,272,033.71 | 47,972,319.97 | -28.7% |
| Total L1 Elapsed Cycles | cycle | 3,947,222,188 | 2,812,027,652 | -28.8% |
| Average L2 Active Cycles | cycle | 70,409,927 | 50,055,529.21 | -28.9% |
| Total L2 Elapsed Cycles | cycle | 1,696,071,432 | 1,207,756,176 | -28.8% |
| Average SM Active Cycles | cycle | 67,272,033.71 | 47,972,319.97 | -28.7% |
| Total SM Elapsed Cycles | cycle | 3,947,222,188 | 2,812,027,652 | -28.8% |
| Average SMSP Active Cycles | cycle | 67,269,141.98 | 47,969,813.99 | -28.7% |
| Total SMSP Elapsed Cycles | cycle | 15,788,888,752 | 11,248,110,608 | -28.8% |


---

## Selective Nsight Compute Analysis — `capacity.cu`

This section presents a focused analysis of Nsight Compute results for [capacity.cu](kernels/capacity.cu). It highlights the primary bottlenecks, the source-level causes identified by the profiler, and concise recommendations for targeted fixes. For broader comparisons and additional detail see Run 2 and Run 3.

![Capacity - Bottlenecks](../../images/run1/capacity_bottlenecks.png)

Root cause summary
------------------
- The dominant contributors to the observed bottlenecks are the explicit DRAM→shared-memory copy operations for matrices `A` and `B`, implemented with PTX `cp.async.ca.shared.global` instructions. These copies appear as the primary sources of uncoalesced accesses.

![Capacity - Source Code - Uncoalesced 1](../../images/run1/capacity_source_code_uncoalesced_shared_access_1.png)

- Additional uncoalesced/shared-access events are attributed to the WMMA load path (`wmma::load_matrix_sync`). Nsight Compute reports these indirectly (via `mma.hpp`/NVidia headers), so the profiler maps them to the helper headers rather than the user source file.

![Capacity - Source Code - Uncoalesced 2](../../images/run1/capacity_source_code_uncoalesced_shared_access_2.png)

Synchronization issue
---------------------
- The `__syncthreads()` barrier present after the `cp.async` sequence contributes heavily to warp stalls (profile reports ~58% of stalls). It appears unnecessary because each warp writes into its own warp-indexed shared buffer slot and `cp.async.wait_group 0` already establishes the required ordering for that warp's async copies. Removing this redundant barrier will reduce warp stalls without changing correctness when the per-warp buffer convention is preserved.

![Capacity - Source Code - syncthreads after wait_group 0](../../images/run1/capacity_source_code_syncthreads_after_commit_group_wait_group.png)


