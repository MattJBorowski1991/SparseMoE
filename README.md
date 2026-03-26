# Sparse Mixture-of-Experts (MoE) Block — Overview

Input to a sparse MoE block is a matrix of token attention-based embeddings.

## Baseline

### 1. Input

Input shape: `[num_tokens, hidden_dim] = [num_batches * seq_len (N), d_model]`.
Example: `num_batches=1`, `seq_len=1024`, `d_model=4096` → `[1 * 1024, 4096] = [1024, 4096]`.

### 2. Router (tiny linear layer)

The router projects each token to per-expert logits:

`[1024, d_model] -> [1024, num_experts]` (e.g. `[1024, 64]`).

### 3. Top-k + gating weights

Per-token: select top-k experts (e.g. `k=4`) and compute softmax gating weights.

`[1024, 64] -> routing indices [1024, 4] and weights [1024, 4]`.

### 4. Expert compute (per expert)

Each expert runs its own mini-MLP. 
m_e = number of routed tokens to this expert e. `[m_e, d_model]` is obtained by gathering all tokens whose routing indices (from point 3) include expert e. Note that m_e can be very small (single digit) and hence we will need to pad m_e to nearest WMMA_M to be able to perform wmma.

`[m_e, d_model] -> up-proj -> [m_e, 4 x d_model] -> activation -> down-proj -> [m_e, d_model]`.

### 5. Combine

Multiply each expert output by its gating weight, sum the `k` contributions per token to produce `[N, d_model]`, and restore the original token order.


## Optimized

The optimized path expands baseline Steps 4-5 into the following routing and compute pipeline:

### 4. Capacity & slot assignment

Compute per-expert capacity and assign slots up to that capacity; apply an overflow policy for excess assignments.

Capacity (per expert) can be computed as:

```
CAP = ceil(N * k / num_experts * capacity_factor)
```

The goal is to pack each expert's routed token vectors into fixed-size, contiguous per-expert tensors so we can execute larger, more efficient GEMMs (per-expert or grouped across experts) instead of many small GEMMs.

Overflow policy: drop.

### 5. Dispatch (local, no AllToAll)

Scatter/sort tokens into expert-contiguous buffers of shape `[num_experts, capacity, d_model]`.
For example (k=2, 5 tokens):

- Expert0 buffer: [ slot0=T0, slot1=T3 ]
- Expert1 buffer: [ slot0=T2, slot1=T5 ]
- Expert2 buffer: [ slot0=T1, slot1=T4 ]

### 6a. Expert compute (per expert)

Each expert runs its MLP on its buffer. Not all slots may be occupied, so the number of active tokens `m` can be < capacity.

For `m` tokens: `[m, d_model] -> up-proj -> [m, d_ff] -> activation -> down-proj -> [m, d_model]`.

### 6b. Grouped GEMM (alternative)

Concatenate expert buffers (pad empty slots to `capacity`) and perform grouped GEMM. Mask out padded outputs afterward.

### 7. Combine / gather

Multiply each expert output by its gating weight, sum the `k` contributions per token to produce `[N, d_model]`, and restore the original token order.


