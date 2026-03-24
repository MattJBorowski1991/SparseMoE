# Sparse Mixture-of-Experts (MoE) Block — Overview

Input to a sparse MoE block is a matrix of token representations (attention-based vectors), not a single large vector.

## 1. Input

Input shape: `[num_tokens, hidden_dim] = [num_batches * seq_len (N), d_model]`.
Example: `num_batches=32`, `seq_len=1024`, `d_model=4096` → `[32 * 1024, 4096] = [32768, 4096]`.

## 2. Router (tiny linear layer)

The router projects each token to per-expert logits:

`[32768, d_model] -> [32768, num_experts]` (e.g. `[32768, 64]`).

## 3. Top-k + gating weights

Per-token: select top-k experts (e.g. `k=4`) and compute softmax gating weights.

`[32768, 64] -> routing indices [32768, 4] and weights [32768, 4]`.

## 4. Capacity & slot assignment

Compute per-expert capacity and assign slots up to that capacity; apply an overflow policy for excess assignments.

Capacity (per expert) can be computed as:

```
CAP = ceil(N * k / num_experts * capacity_factor)
```

The goal is to pack each expert's routed token vectors into fixed-size, contiguous per-expert tensors so we can
execute larger, more efficient GEMMs (per-expert or grouped across experts) instead of many small GEMMs.

Overflow policy examples:
- Reroute to the next-best expert
- Drop the assignment (if latency is prioritized over accuracy)

## 5. Dispatch (local, no AllToAll)

Scatter/sort tokens into expert-contiguous buffers of shape approximately `[num_experts, capacity, d_model]`.
For example (k=2, 5 tokens):

- Expert0 buffer: [ slot0=T0, slot1=T3 ]
- Expert1 buffer: [ slot0=T2, slot1=T5 ]
- Expert2 buffer: [ slot0=T1, slot1=T4 ]

## 6a. Expert compute (per expert)

Each expert runs its MLP on its buffer. Not all slots may be occupied, so the number of active tokens `m` can be < capacity.

For `m` tokens: `[m, d_model] -> up-proj -> [m, d_ff] -> activation -> down-proj -> [m, d_model]`.

## 6b. Grouped GEMM (alternative)

Concatenate expert buffers (pad empty slots to `capacity`) and perform grouped GEMM. Mask out padded outputs afterward.

## 7. Combine / gather

Multiply each expert output by its gating weight, sum the `k` contributions per token to produce `[N, d_model]`, and restore the original token order.


