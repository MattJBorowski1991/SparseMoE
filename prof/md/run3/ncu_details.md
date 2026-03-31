# Nsight Compute Analysis — capacity_v2 vs capacity

This document summarizes a focused analysis of Nsight Compute results for [capacity_v2.cu](kernels/capacity_v2.cu) compared to [capacity.cu](kernels/capacity.cu).

## TL;DR

- DRAM traffic and shared-store conflicts decreased (~37%).
- Swizzling increased on‑chip work and memory‑dependency latency (extra copies/unswizzle; degraded L1 → more L2 hits).
- Compute IPC roughly doubled, but instruction counts and long scoreboard stalls rose sharply.
- Net effect: ~+90% longer duration — external DRAM pressure was traded for increased on‑chip serialization/latency.

## Key Findings

- Shared-memory uncoalesced accesses reduced (positive).
- Swizzling reduced shared-store bank conflicts (~37%), but shared-load bank conflicts did not improve.
- Long scoreboard stalls increased substantially, becoming the dominant bottleneck.
- Compute throughput rose significantly, but memory throughput and DRAM throughput fell, resulting in a longer overall duration.
- The kernel remains memory-bound on the roofline.

## Detailed Analysis

### Bottlenecks

The estimated speedups for main bottlenecks changed compared to `capacity.cu`:

- Uncoalesced Shared Accesses: reduced from ~66% to ~38% (improvement).
- Shared Load Bank Conflicts: reduced from ~42% to ~18% (improvement for stores; loads still show conflicts).
- Long Scoreboard Stalls: increased from ~13% to ~32% (new major issue).
- Theoretical occupancy: unchanged.

![Capacity_v2 - Bottlenecks](../../images/run3/capacity_v2_bottlenecks.jpg)

### Throughput

Compute throughput increased by 103%, but kernel duration still worsened (+90%) because:

- Overall memory throughput dropped to ~68% of peak.
- DRAM throughput dropped to ~61% of peak.
- Aggregate memory bandwidth decreased ~30% to 182.5 GB/s.

![Capacity_v2 - Throughput](../../images/run3/capacity_v2_throughput.jpg)

![Capacity_v2 - Throughput %](../../images/run3/capacity_v2_throughput_p.jpg)

### Roofline

Both kernels remain strongly memory bound according to the roofline analysis.

![Capacity_v2 - Roofline](../../images/run3/capacity_v2_roofline.jpg)

### Compute Workload

Observed compute-workload metrics show ~doubling of Instruction Per Clock as well as SM core instruction throughput ("SM busy [%]").

![Capacity_v2 - Compute Workload](../../images/run3/capacity_v2_compute_workload.jpg)

### Memory Workload

- Memory throughput decreased ~30% to 182.5 GB/s.
- 'Mem Busy' decreased ~31% and max bandwidth decreased ~21%.
- L1 hit rate decreased ~57%; L2 hit rate increased ~93% (more L2 servicing).
- Swizzling reduced shared-store bank conflicts by ~37%, but shared-load bank conflicts did not improve.

![Capacity_v2 - Memory Workload 1](../../images/run3/capacity_v2_memory_workload_1.jpg)

![Capacity_v2 - Memory Workload 2](../../images/run3/capacity_v2_memory_workload_2.jpg)

Close-up on the DRAM throughput reduction:

![Capacity_v2 - Memory Workload 3](../../images/run3/capacity_v2_memory_workload_3.jpg)

### Scheduler Statistics

Scheduler eligibility and issue rates improved (eligible ↑170% toward 1.0, issued ↑100%), but these improvements could not overcome the new on-chip latency sources.

![Capacity_v2 - Scheduler Statistics](../../images/run3/capacity_v2_scheduler_stats.jpg)

### Warp State Statistics

Warp cycle activity improved (warp cycles reduced ~50%), indicating better warp-level work distribution.

![Capacity_v2 - Warp State Statistics](../../images/run3/capacity_v2_warp_state_stats.jpg)

### Instruction Statistics

Instruction counts increased substantially (executed and issued ↑~270%), contributing to higher on-chip pressure and longer execution.

![Capacity_v2 - Instruction Statistics](../../images/run3/capacity_v2_warp_state_stats.jpg)

### Launch Statistics

No major changes in launch parameters other than a modest increase in register pressure.

![Capacity_v2 - Launch Statistics](../../images/run3/capacity_v2_launch_stats.jpg)

### GPU and Memory Workload

Average active cycles increased for DRAM by ~33%; for SM, SMSP, L1 and L2 the average active cycles rose by ~90%.

![Capacity_v2 - GPU and Memory Workload](../../images/run3/capacity_v2_gpu_and_memory_workload.jpg)


