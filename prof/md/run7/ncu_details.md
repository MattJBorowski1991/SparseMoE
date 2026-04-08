**Overview**

This document summarizes an NCU-based comparison between the following kernels:

- `capacity` — reference WMMA-backed kernel (FP16 baseline). Use as the baseline for compute-bound behavior and occupancy.
- `capacity_int8` — INT8 variant using higher-op throughput on Tensor Cores; expect improved throughput if memory layout and packing are optimal.
- `capacity_int8_ptx` — INT8 variant implemented with explicit PTX `mma.sync` (hand-packed registers); micro-optimizations may improve tensor-core utilization and reduce overhead compared to higher-level APIs.
- `capacity_fp8_ptx` — FP8 variant implemented via PTX `mma.sync` (needed because WMMA API lacks FP8 types); watch for quantization-induced accuracy vs performance trade-offs and for how many tensor lanes are active per instruction on Ada.
- `capacity_int4_ptx` — INT4 variant (packed nibbles) implemented with PTX; correctness of packing and B-layout is critical for correct mma operands and for maximizing throughput.

**Ada Lovelace Precision Support**

Target hardware: Ada Lovelace (SM 89, e.g. L4/RTX 40-class). Known precision support relevant to these kernels:

- FP8 (e8m0 / e4m3 / e5m2): native Tensor Core support on Ada Lovelace; the CUDA WMMA C++ API does not provide FP8 helpers, so FP8 use is typically via direct PTX `mma.sync` variants.
- INT8: native Tensor Core support; supported through WMMA and PTX.
- INT4: supported on Tensor Cores; experimental WMMA variants (e.g. `wmma::experimental::precision::s4`) exist but have known caveats — PTX `mma.sync` can be used for hand-written, explicit variants.
- FP4: not supported on Ada (FP4 appears in newer architectures such as Blackwell); ignore for Ada.

Notes: where the WMMA API lacks a convenience type (FP8), kernel implementations typically use PTX `mma.sync` intrinsics or hand-packed register formats.


**NCU Metrics & Comparison Notes**


**Observations & Recommendations**


**Appendix / Next Steps**

