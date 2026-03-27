# Sparse Mixture-of-Experts (MoE) Block — Overview

Input to a sparse MoE block is a matrix of token attention-based embeddings.

## Architecture

### 1. Input

Input shape: `[num_batches, N, d_model]`, where `N` is the sequence length.

### 2. Router (tiny linear layer)

The router projects each token to per-expert logits:

`[N, d_model] -> [N, num_experts]`

### 3. Top-k + gating weights

Per-token: select top-k experts (e.g. `k=4`) and compute softmax gating weights.

`[N, num_experts] -> routing indices [N, k] and weights [N, k]`.

### 4. Dispatch

Scatter/sort tokens into expert-contiguous buffers of shape `[num_experts, CAP, d_model]`. where we `CAP`= capacity per-expert
For example (k=2, 5 tokens):

- Expert0 buffer: [ slot0=T0, slot1=T3 ]
- Expert1 buffer: [ slot0=T2, slot1=T5 ]
- Expert2 buffer: [ slot0=T1, slot1=T4 ]

### 5. Expert compute (per expert)

Each expert runs its MLP on its buffer. Not all slots may be occupied, so the number of active tokens `m` can be < capacity.

For `m` tokens: `[m, d_model] -> up-proj -> [m, d_ff] -> activation -> down-proj -> [m, d_model]`.

### 6. Combine

Multiply each expert output by its gating weight, sum the `k` contributions per token to produce `[N, d_model]`, and restore the original token order.

## Kernels

### Unfused

All global kernels launched seperately in sequence with `CAP = N`.

### Baseline

All kernels from Unfused fused into one global kernel. 

## Capacity


Compute per-expert capacity `CAP` dependent on `capacity_factor` and and assign slots up to that capacity; apply an overflow policy for excess assignments.

Capacity (per expert) computed as:

```
CAP = ceil(N * k / num_experts * capacity_factor)
```

and then roundup to the nearest wmma tile size.

The goal is to pack each expert's routed token vectors into fixed-size, contiguous per-expert tensors so we can execute larger, more efficient GEMMs (per-expert or grouped across experts) instead of many small GEMMs.

Overflow policy: drop.

## Next steps

### Profile for both Prefill (large N) and Decode (N = 1)

### Grouped GEMM (alternative)

Concatenate expert buffers (pad empty slots to `CAP`) and perform grouped GEMM. Mask out padded outputs afterward.




