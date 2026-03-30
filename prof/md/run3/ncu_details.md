## Run 3 — `ldmatrix` experiment in `capacity_v3`

### What was changed

The tensor-core inner loop in `capacity_v3` was rewritten from the higher-level WMMA path

- `wmma::load_matrix_sync`
- `wmma::mma_sync`

to a lower-level tensor-core path based on

- `ldmatrix.sync.aligned`
- `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`

This replaced the previous shared-memory swizzle / unswizzle approach used in `capacity_v2`.

### Why this was done

The earlier swizzling attempt reduced some profiler-reported shared-memory wavefront issues, but it introduced additional shared-memory traffic and extra instructions for the unswizzle step. In practice, that made the real runtime worse.

The `ldmatrix` rewrite was intended to address that more directly:

- load tensor-core fragments from shared memory in the format expected by the hardware,
- avoid the extra unswizzle copy,
- reduce shared-memory bank-conflict/replay pressure near the tensor-core load path,
- move closer to the layout strategy used in low-level libraries such as CUTLASS.

### How it was supposed to help

The expected benefit was improved tensor-core operand loading efficiency:

1. `cp.async` stages tiles into shared memory in a simple linear layout.
2. `ldmatrix` loads warp fragments directly from shared memory into registers.
3. `mma.sync` consumes those registers directly for tensor-core MMA operations.

Compared to the WMMA path, this was supposed to:

- reduce shared-memory replay overhead,
- reduce the amount of shared-memory shuffling needed before MMA,
- give the compiler less opaque fragment-handling work,
- improve the critical path around tensor-core loads.

### Observed result

Profiler-side kernel duration improved, but the end-to-end `cudaEventRecord` duration did not improve correspondingly.

Interpretation:

- the change likely improved the instrumented tensor-core load path,
- but it did not improve the true application bottleneck enough to reduce real runtime,
- or the gains were offset by new costs such as extra register pressure, additional address-generation overhead, or reduced scheduling flexibility.

### Takeaway

The `ldmatrix` rewrite was a valid low-level optimization experiment aimed at improving tensor-core data movement and reducing shared-memory replay effects. However, lower Nsight Compute kernel duration alone is not sufficient evidence of a real speedup.

For final decisions, `cudaEventRecord` remains the primary metric.
