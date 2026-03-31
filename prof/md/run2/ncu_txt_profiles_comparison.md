# NCU TXT Profiles Comparison (Run 2)

Source files:
- capacity_ncu.txt
- capacity_v2_ncu.txt
- capacity_v3_ncu.txt

Kernel columns:
- capacity
- capacity_v2
- capacity_v3

## GPU Speed Of Light Throughput

| Metric Name | Metric Unit | Metric Value | capacity | capacity_v2 | capacity_v3 |
|---|---|---:|---:|---:|---:|
| DRAM Frequency | Ghz | Metric Value | 6.24 | 6.24 | 6.24 |
| SM Frequency | Mhz | Metric Value | 804.67 | 822.95 | 803.42 |
| Elapsed Cycles | cycle | Metric Value | 30063030 | 57658205 | 37175334 |
| Memory Throughput | % | Metric Value | 86.24 | 67.62 | 79.42 |
| DRAM Throughput | % | Metric Value | 86.24 | 61.07 | 79.42 |
| Duration | ms | Metric Value | 36.97 | 69.37 | 45.81 |
| L1/TEX Cache Throughput | % | Metric Value | 84.05 | 70.02 | 52.05 |
| L2 Cache Throughput | % | Metric Value | 32.88 | 44.36 | 62.63 |
| SM Active Cycles | cycle | Metric Value | 29320941.95 | 55109655.78 | 35857617.72 |
| Compute (SM) Throughput | % | Metric Value | 33.63 | 67.62 | 29.40 |

Comments:
- capacity: INF This workload is utilizing greater than 80.0% of the available compute or memory performance of the device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing DRAM in the Memory Workload Analysis section.
- capacity_v2: INF Compute and Memory are well-balanced: To reduce runtime, both computation and memory traffic must be reduced. Check both the Compute Workload Analysis and Memory Workload Analysis sections.
- capacity_v3: OPT Memory is more heavily utilized than Compute: Look at the Memory Workload Analysis section to identify the DRAM bottleneck. Check memory replay (coalescing) metrics to make sure you're efficiently utilizing the bytes transferred. Also consider whether it is possible to do more work per memory access (kernel fusion) or whether there are values you can (re)compute.

Note:
- This table section is present in all three kernels.

## Launch Statistics

| Metric Name | Metric Unit | Metric Value | capacity | capacity_v2 | capacity_v3 |
|---|---|---:|---:|---:|---:|
| Block Size |  | Metric Value | 256 | 256 | 256 |
| Function Cache Configuration |  | Metric Value | CachePreferNone | CachePreferNone | CachePreferNone |
| Grid Size |  | Metric Value | 2048 | 2048 | 2048 |
| Registers Per Thread | register/thread | Metric Value | 72 | 80 | 72 |
| Shared Memory Configuration Size | Kbyte | Metric Value | 102.40 | 102.40 | 102.40 |
| Driver Shared Memory Per Block | Kbyte/block | Metric Value | 1.02 | 1.02 | 1.02 |
| Dynamic Shared Memory Per Block | byte/block | Metric Value | 0 | 0 | 0 |
| Static Shared Memory Per Block | Kbyte/block | Metric Value | 33.02 | 32.77 | 32.77 |
| # SMs | SM | Metric Value | 58 | 58 | 58 |
| Stack Size |  | Metric Value | 1024 | 1024 | 1024 |
| Threads | thread | Metric Value | 524288 | 524288 | 524288 |
| # TPCs |  | Metric Value | 29 | 29 | 29 |
| Enabled TPC IDs |  | Metric Value | all | all | all |
| Uses Green Context |  | Metric Value | 0 | 0 | 0 |
| Waves Per SM |  | Metric Value | 11.77 | 11.77 | 11.77 |

Comments:
- capacity: No comment block in this section.
- capacity_v2: No comment block in this section.
- capacity_v3: No comment block in this section.

Note:
- This table section is present in all three kernels.

## Occupancy

| Metric Name | Metric Unit | Metric Value | capacity | capacity_v2 | capacity_v3 |
|---|---|---:|---:|---:|---:|
| Block Limit SM | block | Metric Value | 24 | 24 | 24 |
| Block Limit Registers | block | Metric Value | 3 | 3 | 3 |
| Block Limit Shared Mem | block | Metric Value | 3 | 3 | 3 |
| Block Limit Warps | block | Metric Value | 6 | 6 | 6 |
| Theoretical Active Warps per SM | warp | Metric Value | 24 | 24 | 24 |
| Theoretical Occupancy | % | Metric Value | 50 | 50 | 50 |
| Achieved Occupancy | % | Metric Value | 49.19 | 48.99 | 48.93 |
| Achieved Active Warps Per SM | warp | Metric Value | 23.61 | 23.51 | 23.49 |

Comments:
- capacity: OPT Est. Local Speedup: 50%. The 6.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (50.0%) is limited by the number of required registers, and the required amount of shared memory.
- capacity_v2: OPT Est. Local Speedup: 50%. The 6.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (50.0%) is limited by the number of required registers, and the required amount of shared memory.
- capacity_v3: OPT Est. Local Speedup: 50%. The 6.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (50.0%) is limited by the number of required registers, and the required amount of shared memory.

Note:
- This table section is present in all three kernels.

## GPU and Memory Workload Distribution

| Metric Name | Metric Unit | Metric Value | capacity | capacity_v2 | capacity_v3 |
|---|---|---:|---:|---:|---:|
| Average DRAM Active Cycles | cycle | Metric Value | 199112960 | 264573200 | 227192048 |
| Total DRAM Elapsed Cycles | cycle | Metric Value | 1385345024 | 2599341056 | 1716303872 |
| Average L1 Active Cycles | cycle | Metric Value | 29320941.95 | 55109655.78 | 35857617.72 |
| Total L1 Elapsed Cycles | cycle | Metric Value | 1723078788 | 3309925018 | 2141142518 |
| Average L2 Active Cycles | cycle | Metric Value | 30315917.54 | 56303510.96 | 37590134.88 |
| Total L2 Elapsed Cycles | cycle | Metric Value | 732076992 | 1378815744 | 906970200 |
| Average SM Active Cycles | cycle | Metric Value | 29320941.95 | 55109655.78 | 35857617.72 |
| Total SM Elapsed Cycles | cycle | Metric Value | 1723078788 | 3309925018 | 2141142518 |
| Average SMSP Active Cycles | cycle | Metric Value | 29318292.22 | 55079821.02 | 35844259.28 |
| Total SMSP Elapsed Cycles | cycle | Metric Value | 6892315152 | 13239700072 | 8564570072 |

Comments:
- capacity: No comment block in this section.
- capacity_v2: No comment block in this section.
- capacity_v3: No comment block in this section.

Note:
- This table section is present in all three kernels.
