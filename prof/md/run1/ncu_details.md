 # Run 1 — Nsight Compute: Professional Summary

 **Date:** 2026-03-30

 Overview
 --------
This report summarizes Nsight Compute outputs for Run 1 and compares the [capacity](kernels/capacity.cu) and [baseline](kernels/baseline.cu) implementations. The [unfused](kernels/unfused.cu) variant produces separate kernel traces and is not included in the aggregated tables below.

 Configuration
 -------------
 ```cpp
 constexpr int N = 512;
 constexpr int d_model = 4096;
 constexpr int num_batches = 4;
 constexpr int num_experts = 32;
 constexpr int up_proj_dim = 4;
 ```

Results
-------
- [unfused](kernels/unfused.cu): 2780 ms (WMMA kernels for `up_proj` and `down_proj` account for `~99%` of latency)
- [baseline](kernels/baseline.cu): 85.6 ms (high DRAM usage due to naive per-expert over-allocation)
- [capacity](kernels/capacity.cu): 61 ms

This showcases the strength/necessity of kernel fusion - 32x speedup on raw fusion! Subsequently this showcases how important it is not to waste VRAM - extra +40% speedup of [capacity](kernels/capacity.cu) over [baseline](kernels/baseline.cu) based on sensible per-expert buffer allocation via `capacity_factor`.

 Executive Summary
 -----------------
 - **Unfused (per-kernel aggregate):** WMMA `up_proj` / `down_proj` kernels dominate runtime (per-kernel traces).
 - **Baseline (aggregated):** 85.6 ms — elevated DRAM utilization and replay indicate potential locality and coalescing issues.
 - **Capacity:** 61 ms — demonstrates improved runtime due to more efficient per-expert buffering and a better compute/memory balance.

 Observations
 ------------
 - Kernel fusion delivers significant benefits in this workload; reducing redundant memory traffic and improving data locality should be primary optimization targets.
 - Capacity-aware per-expert buffer allocation materially improves performance compared with naive over-allocation.

## GPU Speed Of Light Throughput

| Metric Name | Metric Unit | capacity | baseline |
|---|---:|---:|---:|
| DRAM Frequency | Ghz | 6.24 | 6.24 |
| SM Frequency | Mhz | 795.00 | 795.00 |
| Elapsed Cycles | cycle | 48,492,117 | 68,099,844 |
| Memory Throughput | % | 84.91 | 76.38 |
| DRAM Throughput | % | 78.78 | 76.38 |
| Duration | ms | 61.00 | 85.66 |
| L1/TEX Cache Throughput | % | 85.81 | 62.98 |
| L2 Cache Throughput | % | 33.27 | 29.56 |
| SM Active Cycles | cycle | 47,972,319.97 | 67,272,033.71 |
| Compute (SM) Throughput | % | 34.58 | 27.95 |

**Comments:**

- `capacity` (INF): "This workload is utilizing greater than 80.0% of the available compute or memory performance of the device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section." (from `ncu_capacity.txt`)
- `baseline` (OPT): "Memory is more heavily utilized than Compute: Look at the Memory Workload Analysis section to identify the DRAM bottleneck. Check memory replay (coalescing) metrics to make sure you're efficiently utilizing the bytes transferred. Also consider whether it is possible to do more work per memory access (kernel fusion) or whether there are values you can (re)compute." (from `ncu_baseline.txt`)

---

## Launch Statistics

| Metric Name | Metric Unit | capacity | baseline |
|---|---:|---:|---:|
| Block Size |  | 256 | 256 |
| Function Cache Configuration |  | CachePreferNone | CachePreferNone |
| Grid Size |  | 4096 | 4096 |
| Registers Per Thread | register/thread | 72 | 70 |
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

**Notes:** All listed fields are present in both profiles; values differ only for `Registers Per Thread` (capacity=72, baseline=70).

---

## Occupancy

| Metric Name | Metric Unit | capacity | baseline |
|---|---:|---:|---:|
| Block Limit SM | block | 24 | 24 |
| Block Limit Registers | block | 3 | 3 |
| Block Limit Shared Mem | block | 3 | 3 |
| Block Limit Warps | block | 6 | 6 |
| Theoretical Active Warps per SM | warp | 24 | 24 |
| Theoretical Occupancy | % | 50 | 50 |
| Achieved Occupancy | % | 49.69 | 49.58 |
| Achieved Active Warps Per SM | warp | 23.85 | 23.80 |

**Comments:**

- `capacity` (OPT): "Est. Local Speedup: 50% ... The 6.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (50.0%) is limited by the number of required registers, and the required amount of shared memory." (from `ncu_capacity.txt`)
- `baseline` (OPT): same note present in `ncu_baseline.txt`.

---

## GPU and Memory Workload Distribution

| Metric Name | Metric Unit | capacity | baseline |
|---|---:|---:|---:|
| Average DRAM Active Cycles | cycle | 300,068,704 | 408,562,912 |
| Total DRAM Elapsed Cycles | cycle | 2,285,443,072 | 3,209,560,064 |
| Average L1 Active Cycles | cycle | 47,972,319.97 | 67,272,033.71 |
| Total L1 Elapsed Cycles | cycle | 2,812,027,652 | 3,947,222,188 |
| Average L2 Active Cycles | cycle | 50,055,529.21 | 70,409,927 |
| Total L2 Elapsed Cycles | cycle | 1,207,756,176 | 1,696,071,432 |
| Average SM Active Cycles | cycle | 47,972,319.97 | 67,272,033.71 |
| Total SM Elapsed Cycles | cycle | 2,812,027,652 | 3,947,222,188 |
| Average SMSP Active Cycles | cycle | 47,969,813.99 | 67,269,141.98 |
| Total SMSP Elapsed Cycles | cycle | 11,248,110,608 | 15,788,888,752 |

---

Perform a detailed Nsight Compute analysis of `capacity` for further variants. The `unfused` variant produces separate kernel profiles and is not included in the aggregated tables below.