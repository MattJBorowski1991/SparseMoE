# NCU Profiles Comparison

This file consolidates the NCU high-level reports from `prof/txt/run_quant/all_ncu_combined.txt` for the five kernels:

- `capacity`
- `capacity_int8`
- `capacity_int8_ptx`
- `capacity_fp8_ptx`
- `capacity_int4_ptx`

The tables below preserve exact metric names, metric units, and metric values. Each comparison table groups the metrics by the same sections and order found in the source file.

**GPU Speed Of Light Throughput**

| Metric Name | Metric Unit | capacity | capacity_int8 | capacity_int8_ptx | capacity_fp8_ptx | capacity_int4_ptx |
|---|---:|---:|---:|---:|---:|---:|
| DRAM Frequency | Ghz | 6.24 | 6.24 | 6.24 | 6.24 | 6.24 |
| SM Frequency | Mhz | 795.00 | 795.00 | 795.00 | 795.00 | 795.01 |
| Elapsed Cycles | cycle | 29588600 | 30892554 | 24357729 | 30767616 | 11145744 |
| Memory Throughput | % | 85.75 | 82.08 | 79.91 | 62.95 | 60.91 |
| DRAM Throughput | % | 85.75 | 55.62 | 68.38 | 56.56 | 58.26 |
| Duration | ms | 37.22 | 38.86 | 30.64 | 38.70 | 14.02 |
| L1/TEX Cache Throughput | % | 84.81 | 83.42 | 81.38 | 64.01 | 62.90 |
| L2 Cache Throughput | % | 32.62 | 22.67 | 27.92 | 21.78 | 22.36 |
| SM Active Cycles | cycle | 29070664 | 30385127.66 | 23855711.90 | 30300570 | 10825457.64 |
| Compute (SM) Throughput | % | 33.71 | 52.01 | 53.86 | 57.78 | 55.69 |

Comments (copied verbatim from the source, associated with kernel):

- capacity: INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of the device.   
  To further improve performance, work will likely need to be shifted from the most utilized to another unit.   
  Start by analyzing DRAM in the Memory Workload Analysis section.

- capacity_int8: INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of the device.   
  To further improve performance, work will likely need to be shifted from the most utilized to another unit.   
  Start by analyzing L1 in the Memory Workload Analysis section.

- capacity_int8_ptx: OPT   Memory is more heavily utilized than Compute: Look at the Memory Workload Analysis section to identify the L1 
  bottleneck. Check memory replay (coalescing) metrics to make sure you're efficiently utilizing the bytes      
  transferred. Also consider whether it is possible to do more work per memory access (kernel fusion) or        
  whether there are values you can (re)compute.

- capacity_fp8_ptx: INF   Compute and Memory are well-balanced: To reduce runtime, both computation and memory traffic must be reduced. 
  Check both the Compute Workload Analysis and Memory Workload Analysis sections.

- capacity_int4_ptx: INF   Compute and Memory are well-balanced: To reduce runtime, both computation and memory traffic must be reduced. 
  Check both the Compute Workload Analysis and Memory Workload Analysis sections.


**Launch Statistics**

| Metric Name | Metric Unit | capacity | capacity_int8 | capacity_int8_ptx | capacity_fp8_ptx | capacity_int4_ptx |
|---|---|---:|---:|---:|---:|---:|
| Block Size |  | 256 | 256 | 256 | 256 | 256 |
| Function Cache Configuration |  | CachePreferNone | CachePreferNone | CachePreferNone | CachePreferNone | CachePreferNone |
| Grid Size |  | 2048 | 2048 | 2048 | 2048 | 2048 |
| Registers Per Thread | register/thread | 72 | 64 | 75 | 72 | 72 |
| Shared Memory Configuration Size | Kbyte | 102.40 | 102.40 | 102.40 | 102.40 | 102.40 |
| Driver Shared Memory Per Block | Kbyte/block | 1.02 | 1.02 | 1.02 | 1.02 | 1.02 |
| Dynamic Shared Memory Per Block | byte/block | 0 | 0 | 0 | 0 | 0 |
| Static Shared Memory Per Block | Kbyte/block | 33.02 | 24.83 | 33.02 | 33.02 | 28.93 |
| # SMs | SM | 58 | 58 | 58 | 58 | 58 |
| Stack Size |  | 1024 | 1024 | 1024 | 1024 | 1024 |
| Threads | thread | 524288 | 524288 | 524288 | 524288 | 524288 |
| # TPCs |  | 29 | 29 | 29 | 29 | 29 |
| Enabled TPC IDs |  | all | all | all | all | all |
| Uses Green Context |  | 0 | 0 | 0 | 0 | 0 |
| Waves Per SM |  | 11.77 | 11.77 | 11.77 | 11.77 | 11.77 |

Notes: The `Function Cache Configuration` values appear in the source under the "Metric Unit" column and the source's "Metric Value" cell is empty for that row; the table above preserves that content as shown in the source.


**Occupancy**

| Metric Name | Metric Unit | capacity | capacity_int8 | capacity_int8_ptx | capacity_fp8_ptx | capacity_int4_ptx |
|---|---:|---:|---:|---:|---:|---:|
| Block Limit SM | block | 24 | 24 | 24 | 24 | 24 |
| Block Limit Registers | block | 3 | 4 | 3 | 3 | 3 |
| Block Limit Shared Mem | block | 3 | 3 | 3 | 3 | 3 |
| Block Limit Warps | block | 6 | 6 | 6 | 6 | 6 |
| Theoretical Active Warps per SM | warp | 24 | 24 | 24 | 24 | 24 |
| Theoretical Occupancy | % | 50 | 50 | 50 | 50 | 50 |
| Achieved Occupancy | % | 49.28 | 48.95 | 48.91 | 48.84 | 48.79 |
| Achieved Active Warps Per SM | warp | 23.65 | 23.49 | 23.48 | 23.44 | 23.42 |

Comments (Occupancy):

- capacity: OPT   Est. Local Speedup: 50%                                                                                        
  The 6.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the       
  hardware maximum of 12. This kernel's theoretical occupancy (50.0%) is limited by the number of required      
  registers, and the required amount of shared memory.

- capacity_int8: OPT   Est. Local Speedup: 50%                                                                                        
  The 6.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the       
  hardware maximum of 12. This kernel's theoretical occupancy (50.0%) is limited by the required amount of      
  shared memory.

- capacity_int8_ptx: OPT   Est. Local Speedup: 50%                                                                                        
  The 6.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the       
  hardware maximum of 12. This kernel's theoretical occupancy (50.0%) is limited by the number of required      
  registers, and the required amount of shared memory.

- capacity_fp8_ptx: OPT   Est. Local Speedup: 50%                                                                                        
  The 6.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the       
  hardware maximum of 12. This kernel's theoretical occupancy (50.0%) is limited by the number of required      
  registers, and the required amount of shared memory.

- capacity_int4_ptx: OPT   Est. Local Speedup: 50%                                                                                        
  The 6.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the       
  hardware maximum of 12. This kernel's theoretical occupancy (50.0%) is limited by the number of required      
  registers, and the required amount of shared memory.


**GPU and Memory Workload Distribution**

| Metric Name | Metric Unit | capacity | capacity_int8 | capacity_int8_ptx | capacity_fp8_ptx | capacity_int4_ptx |
|---|---:|---:|---:|---:|---:|---:|
| Average DRAM Active Cycles | cycle | 199295122.67 | 134964736 | 130838658.67 | 136696389.33 | 51000768 |
| Total DRAM Elapsed Cycles | cycle | 1394502656 | 1455967232 | 1147978752 | 1450075136 | 525268992 |
| Average L1 Active Cycles | cycle | 29070664 | 30385127.66 | 23855711.90 | 30300570 | 10825457.64 |
| Total L1 Elapsed Cycles | cycle | 1719022632 | 1791227406 | 1409009068 | 1787114872 | 648335918 |
| Average L2 Active Cycles | cycle | 30476312.25 | 32008860.88 | 25133239.04 | 31888388.92 | 11384673.83 |
| Total L2 Elapsed Cycles | cycle | 736916520 | 769396848 | 606642744 | 766282416 | 277575216 |
| Average SM Active Cycles | cycle | 29070664 | 30385127.66 | 23855711.90 | 30300570 | 10825457.64 |
| Total SM Elapsed Cycles | cycle | 1719022632 | 1791227406 | 1409009068 | 1787114872 | 648335918 |
| Average SMSP Active Cycles | cycle | 29067214.59 | 30384896.37 | 23853800.54 | 30299048.93 | 10820931.03 |
| Total SMSP Elapsed Cycles | cycle | 6876090528 | 7164909624 | 5636036272 | 7148459488 | 2593343672 |

Notes: All four section types ("GPU Speed Of Light Throughput", "Launch Statistics", "Occupancy", "GPU and Memory Workload Distribution") were present for all five kernels in the source file; no section types were missing for any kernel.
