#include "include/config.h"
#include "include/moe_args.h"
#include <mma.h>
using namespace nvcuda;
#include <stdio.h>
#include <assert.h>
#include <cstdint>

#define MOE_KERNEL capacity
#define MOE_USES_CAPACITY 1

constexpr int PTX_MMA_K_INT4 = 64;

// Host-side requirement: transpose and pack weight matrices to [N/2][K] layout before uploading.
// B packs along the N dimension (2 adjacent N-cols per byte, one byte per K-position),
// so each transposed row is K bytes wide (not K/2).

// Packed = 2 int4 values stored in 1 byte (one per nibble=4bits (low nibble = bits 3:0, high nibble = bits 7:4)). 
// Unpacked = 1 int4 value stored in 1 byte (wasting the upper 4 bits).


// ---------------------------------------------------------------------------
// ptx_mma_m16n8k64_s4s4
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void ptx_mma_m16n8k64_s4s4(
    const uint32_t a[4],
    const uint32_t b[2],
    int32_t        c[4]
){
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),
           "r"(b[0]),  "r"(b[1])
    );
}

// ---------------------------------------------------------------------------
// load_a_m16k64_s4:
//   A smem tile: [WMMA_M][PTX_MMA_K_INT4/2 + PAD] stored as PACKED int4
//   (2 nibbles per byte) — so physical width = 32 bytes per row
//   Each thread reads two uint32 (8 bytes) covering its 2 rows × 8 nibbles.
//   PTX m16n8k64 s4 A layout (row-major):
//     a[0]: row (lane/4)*2+0, k-cols  0..31  → 8 nibbles = 4 bytes = 1 uint32
//     a[1]: row (lane/4)*2+0, k-cols 32..63  → 1 uint32
//     a[2]: row (lane/4)*2+1, k-cols  0..31  → 1 uint32
//     a[3]: row (lane/4)*2+1, k-cols 32..63  → 1 uint32
//   col offset = (lane % 4) * 4 bytes (each lane covers 8 consecutive nibbles = 4 bytes)
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void load_a_m16k64_s4(
    const uint8_t* tile,  // [WMMA_M][PTX_MMA_K_INT4/2 + PAD] packed int4
    int            ldA,   // = PTX_MMA_K_INT4/2 + PAD  (bytes per row)
    uint32_t       a[4]
){
    int lane = threadIdx.x % 32;
    int row0 = (lane / 4) * 2;
    int row1 = row0 + 1;
    int col  = (lane % 4) * 4;  // byte offset: 4 bytes = 8 nibbles

    // first k-half  (nibble cols  0..31 = bytes 0..15)
    a[0] = *reinterpret_cast<const uint32_t*>(&tile[row0 * ldA + col]);
    a[2] = *reinterpret_cast<const uint32_t*>(&tile[row1 * ldA + col]);
    // second k-half (nibble cols 32..63 = bytes 16..31)
    a[1] = *reinterpret_cast<const uint32_t*>(&tile[row0 * ldA + col + 16]);
    a[3] = *reinterpret_cast<const uint32_t*>(&tile[row1 * ldA + col + 16]);
}

// ---------------------------------------------------------------------------
// load_b_m8k64_s4:
//   B smem tile: [PTX_MMA_K_INT4][WMMA_N/2 + PAD] packed int4
//   PTX m16n8k64 s4 B layout (col-major):
//     b[0]: k-rows  0.. 7, col this thread owns → 8 nibbles = 4 bytes = 1 uint32
//     b[1]: k-rows 32..39, col this thread owns → 1 uint32
//   Each thread owns 1 logical col; 8 k-rows of packed int4 = 4 bytes.
//   Stored row-major in smem so: address = row * ldB_bytes + byte_col
//   We read 4 consecutive bytes across 8 consecutive k-rows:
//   pack by reading one byte per k-row and assembling into uint32.
//   BUT: smem is [K][N/2], so cols are packed horizontally.
//   For 8 consecutive k-rows, col is fixed → 8 separate byte reads.
//   HOWEVER: since k-stride is ldB_bytes (= N/2+PAD = 8+PAD bytes),
//   we can read a uint32 by transposing: use ldA-style read along k if
//   we store B transposed. Instead, keep B as [K][N/2] and do 4 byte reads
//   with __byte_perm — still scalar but only 4 ops not 8.
//   Best: store B as [N/2][K] (transposed) so k is contiguous.
//   B packs along N: each byte holds 1 K-position × 2 N-cols (low nibble=even, high nibble=odd).
//   We transpose B on host. Then ldB = K bytes per row (one byte per K-position, NOT K/2).
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void load_b_m8k64_s4_transposed(
    const uint8_t* tile,       // [WMMA_N/2][PTX_MMA_K_INT4] B transposed, packed int4
    int            ldB,        // = PTX_MMA_K_INT4 (bytes per row = 64)
    int            n_col_base, // 0 or 8 (logical nibble col base)
    uint32_t       b[2]
){
    int lane = threadIdx.x % 32;
    // logical col in [0..7] within this n8 block
    int logical_col  = n_col_base + (lane / 4) % 2;
    // packed: 2 nibbles per byte → byte row = logical_col/2, nibble = logical_col%2
    int byte_row     = logical_col / 2;
    int nibble_shift = (logical_col % 2) * 4;  // 0 or 4

    // k-half 0: nibble k-cols 0..31 → bytes 0..15 in this row
    // each uint32 holds 8 nibbles = 4 bytes, lane%4 selects which 4-byte chunk
    int k_byte_off = (lane % 4) * 4;  // NOT used for b — b is col-indexed

    // For B col-major: we need 8 consecutive k-rows for one n-col.
    // With B stored as [N/2][K] transposed (one byte per K-position):
    //   row = byte_row (selects the n-col pair)
    //   within that row, k-half 0 = bytes [0..31], k-half 1 = bytes [32..63]
    // Lane%4 selects which 4-byte chunk within each k-half (4 lanes × 4 bytes = 16 bytes per half).
    // raw0+raw1 cover k-half 0; raw2+raw3 cover k-half 1.
    const uint8_t* row_ptr = &tile[byte_row * ldB];

    uint32_t raw0 = *reinterpret_cast<const uint32_t*>(&row_ptr[k_byte_off]);
    uint32_t raw1 = *reinterpret_cast<const uint32_t*>(&row_ptr[k_byte_off + 16]);

    // if nibble_shift==0: low nibbles; if nibble_shift==4: high nibbles
    // extract every other nibble from raw into packed output
    if (nibble_shift == 0) {
        // extract low nibbles: bits [3:0], [11:8], [19:16], [27:24], ...
        // i.e. nibbles 0,2,4,6 of each byte → pack into 4-byte uint32
        b[0] = ((raw0 >>  0) & 0xF)        |
               (((raw0 >>  8) & 0xF) <<  4) |
               (((raw0 >> 16) & 0xF) <<  8) |
               (((raw0 >> 24) & 0xF) << 12) |
               (((raw1 >>  0) & 0xF) << 16) |
               (((raw1 >>  8) & 0xF) << 20) |
               (((raw1 >> 16) & 0xF) << 24) |
               (((raw1 >> 24) & 0xF) << 28);
        uint32_t raw2 = *reinterpret_cast<const uint32_t*>(&row_ptr[k_byte_off + 32]);
        uint32_t raw3 = *reinterpret_cast<const uint32_t*>(&row_ptr[k_byte_off + 48]);
        b[1] = ((raw2 >>  0) & 0xF)        |
               (((raw2 >>  8) & 0xF) <<  4) |
               (((raw2 >> 16) & 0xF) <<  8) |
               (((raw2 >> 24) & 0xF) << 12) |
               (((raw3 >>  0) & 0xF) << 16) |
               (((raw3 >>  8) & 0xF) << 20) |
               (((raw3 >> 16) & 0xF) << 24) |
               (((raw3 >> 24) & 0xF) << 28);
    } else {
        b[0] = (((raw0 >>  4) & 0xF))       |
               (((raw0 >> 12) & 0xF) <<  4) |
               (((raw0 >> 20) & 0xF) <<  8) |
               (((raw0 >> 28) & 0xF) << 12) |
               (((raw1 >>  4) & 0xF) << 16) |
               (((raw1 >> 12) & 0xF) << 20) |
               (((raw1 >> 20) & 0xF) << 24) |
               (((raw1 >> 28) & 0xF) << 28);
        uint32_t raw2 = *reinterpret_cast<const uint32_t*>(&row_ptr[k_byte_off + 32]);
        uint32_t raw3 = *reinterpret_cast<const uint32_t*>(&row_ptr[k_byte_off + 48]);
        b[1] = (((raw2 >>  4) & 0xF))       |
               (((raw2 >> 12) & 0xF) <<  4) |
               (((raw2 >> 20) & 0xF) <<  8) |
               (((raw2 >> 28) & 0xF) << 12) |
               (((raw3 >>  4) & 0xF) << 16) |
               (((raw3 >> 12) & 0xF) << 20) |
               (((raw3 >> 20) & 0xF) << 24) |
               (((raw3 >> 28) & 0xF) << 28);
    }
}

// ---------------------------------------------------------------------------
// wmma_db_int4: fully packed int4 x int4, double-buffered
//   A smem: [WMMA_M][PTX_MMA_K_INT4/2 + PAD] packed int4 (32 bytes/row)
//   B smem: [WMMA_N/2][PTX_MMA_K_INT4/2 + PAD] B TRANSPOSED packed int4
//   cp.async loads 16 bytes at a time — no scalar loops anywhere
// ---------------------------------------------------------------------------
template<bool calculatePerExpert>
static __device__ __forceinline__ void wmma_db_int4(
    const uint8_t* A,          // packed int4 activations [M][K/2]
    const uint8_t* B,          // packed int4 weights, B-transposed [N/2][K] per expert
    int32_t*       C_global,
    int M, int N, int K,
    int tile_row_in,
    int tile_col_in
){
    assert((K % PTX_MMA_K_INT4) == 0);

    int batch   = blockIdx.z;
    int tid     = threadIdx.x;
    int warp_id = tid / THREADS_PER_WARP;
    int lane_id = tid % THREADS_PER_WARP;

    const int K2 = K / 2;   // packed byte width of A rows
    const int N2 = N / 2;   // number of B-transposed rows (N-col pairs)

    const uint8_t* A_batch = A + batch * M * K2;
    const uint8_t* B_batch = B;

    const uint8_t* A_e;
    const uint8_t* B_e;
    int tile_row_local = tile_row_in;

    if constexpr (calculatePerExpert) {
        const int rows_per_expert = M / num_experts;
        const int expert_id       = tile_row_in / rows_per_expert;
        tile_row_local            = tile_row_in % rows_per_expert;
        A_e = A_batch + expert_id * rows_per_expert * K2;
        // B transposed: [N/2][K] per expert — K bytes per row (one byte per K-position)
        B_e = B_batch + expert_id * N2 * K;
    } else {
        A_e = A_batch;
        B_e = B_batch;
    }

    // A smem: [WMMA_M][K_INT4/2 + PAD] packed int4 — 16*32 = 512 bytes per warp  (A packs along K: 2 int4/byte)
    // B smem: [WMMA_N/2][K_INT4 + PAD] transposed packed int4 — 8*64 = 512 bytes per warp  (B packs along N: 1 byte/K-pos)
    constexpr int A_ROW_BYTES = PTX_MMA_K_INT4 / 2 + PAD;  // 32
    constexpr int B_ROW_BYTES = PTX_MMA_K_INT4 + PAD;       // 64

    __shared__ __align__(16) uint8_t As[2][WARPS_PER_BLOCK][WMMA_M][A_ROW_BYTES];
    __shared__ __align__(16) uint8_t Bs[2][WARPS_PER_BLOCK][WMMA_N / 2][B_ROW_BYTES];

    int32_t D0[4] = {0, 0, 0, 0};
    int32_t D1[4] = {0, 0, 0, 0};

    int buf = 0;

    // --- initial load: 16 bytes per cp.async, no scalar loops ---
    // A tile: WMMA_M * A_ROW_BYTES = 16 * 32 = 512 bytes
    // 32 threads × 16 bytes = 512 bytes exactly — one cp.async per thread
    {
        int byte_idx = lane_id * 16;
        int row = byte_idx / A_ROW_BYTES, col = byte_idx % A_ROW_BYTES;
        As[buf][warp_id][row][col] = 0;  // clear first (use direct assign below)
    }
    for (int i = lane_id * 16; i < WMMA_M * A_ROW_BYTES; i += 32 * 16) {
        int row = i / A_ROW_BYTES, col = i % A_ROW_BYTES;
        char*       dst = (char*)&As[buf][warp_id][row][col];
        const char* src = (const char*)&A_e[(tile_row_local + row) * K2 + col];
        unsigned smem_ptr = __cvta_generic_to_shared(dst);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(smem_ptr), "l"(src));
    }
    // B tile (transposed): (WMMA_N/2) * B_ROW_BYTES = 8 * 64 = 512 bytes
    // load tile_col_in/2 .. tile_col_in/2 + WMMA_N/2 rows of B-transposed
    for (int i = lane_id * 16; i < (WMMA_N / 2) * B_ROW_BYTES; i += 32 * 16) {
        int row = i / B_ROW_BYTES, col = i % B_ROW_BYTES;
        char*       dst = (char*)&Bs[buf][warp_id][row][col];
        const char* src = (const char*)&B_e[(tile_col_in / 2 + row) * K + col];
        unsigned smem_ptr = __cvta_generic_to_shared(dst);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(smem_ptr), "l"(src));
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");

    uint32_t a[4], b0[2], b1[2];
    load_a_m16k64_s4        (&As[buf][warp_id][0][0],  A_ROW_BYTES, a);
    load_b_m8k64_s4_transposed(Bs[buf][warp_id][0],    B_ROW_BYTES, 0, b0);
    load_b_m8k64_s4_transposed(Bs[buf][warp_id][0],    B_ROW_BYTES, 8, b1);

    // --- main loop ---
    for (int k_off = PTX_MMA_K_INT4; k_off < K; k_off += PTX_MMA_K_INT4) {
        int next    = 1 - buf;
        int k_off2  = k_off / 2;  // byte offset into packed K

        for (int i = lane_id * 16; i < WMMA_M * A_ROW_BYTES; i += 32 * 16) {
            int row = i / A_ROW_BYTES, col = i % A_ROW_BYTES;
            char*       dst = (char*)&As[next][warp_id][row][col];
            const char* src = (const char*)&A_e[(tile_row_local + row) * K2 + k_off2 + col];
            unsigned smem_ptr = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(smem_ptr), "l"(src));
        }
        for (int i = lane_id * 16; i < (WMMA_N / 2) * B_ROW_BYTES; i += 32 * 16) {
            int row = i / B_ROW_BYTES, col = i % B_ROW_BYTES;
            char*       dst = (char*)&Bs[next][warp_id][row][col];
            const char* src = (const char*)&B_e[(tile_col_in / 2 + row) * K + k_off + col];
            unsigned smem_ptr = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(smem_ptr), "l"(src));
        }
        asm volatile("cp.async.commit_group;");
        __syncthreads();

        ptx_mma_m16n8k64_s4s4(a, b0, D0);
        ptx_mma_m16n8k64_s4s4(a, b1, D1);
        asm volatile("cp.async.wait_group 0;");

        buf = next;
        load_a_m16k64_s4        (&As[buf][warp_id][0][0], A_ROW_BYTES, a);
        load_b_m8k64_s4_transposed(Bs[buf][warp_id][0],   B_ROW_BYTES, 0, b0);
        load_b_m8k64_s4_transposed(Bs[buf][warp_id][0],   B_ROW_BYTES, 8, b1);
    }

    ptx_mma_m16n8k64_s4s4(a, b0, D0);
    ptx_mma_m16n8k64_s4s4(a, b1, D1);

    {
        int row0      = (lane_id / 4) * 2;
        int row1      = row0 + 1;
        int col_left  = lane_id % 4;
        int col_right = lane_id % 4;

        int32_t* base = C_global + (blockIdx.z * M + tile_row_in) * N + tile_col_in;

        base[(row0) * N + col_left]          = D0[0];
        base[(row0) * N + col_left  + 4]     = D0[1];
        base[(row1) * N + col_left]          = D0[2];
        base[(row1) * N + col_left  + 4]     = D0[3];

        base[(row0) * N + 8 + col_right]     = D1[0];
        base[(row0) * N + 8 + col_right + 4] = D1[1];
        base[(row1) * N + 8 + col_right]     = D1[2];
        base[(row1) * N + 8 + col_right + 4] = D1[3];
    }
}

// ---------------------------------------------------------------------------
// wmma_db_fp16 — unchanged
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void wmma_db_fp16(
    float alpha,
    const half* A,
    const half* B,
    float* C,
    int M, int N, int K
){
    assert( (M % WMMA_M == 0) && (N % WMMA_N == 0) && (K % WMMA_K_FP16 == 0) );
    int batch = blockIdx.z;

    const half* A_batch = A + batch * M * K;
    const half* B_batch = B;
    float*      C_batch = C + batch * M * N;

    int tid     = threadIdx.x;
    int warp_id = tid / THREADS_PER_WARP;
    int lane_id = tid % THREADS_PER_WARP;

    int warp_tile_row = warp_id / WARP_TILES_X;
    int warp_tile_col = warp_id % WARP_TILES_X;
    const int tile_row = blockIdx.y * (WMMA_M * WARP_TILES_Y) + warp_tile_row * WMMA_M;
    const int tile_col = blockIdx.x * (WMMA_N * WARP_TILES_X) + warp_tile_col * WMMA_N;
    if (tile_row >= M || tile_col >= N) return;

    __shared__ __align__(16) half As[2][WARPS_PER_BLOCK][WMMA_M][WMMA_K_FP16 + PAD];
    __shared__ __align__(16) half Bs[2][WARPS_PER_BLOCK][WMMA_K_FP16][WMMA_N + PAD];

    wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K_FP16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K_FP16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K_FP16, float>                 c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int buf = 0;

    for (int i = lane_id; i < WMMA_M * WMMA_K_FP16; i += 32) {
        int row = i / WMMA_K_FP16, col = i % WMMA_K_FP16;
        As[buf][warp_id][row][col] = A_batch[(tile_row + row) * K + col];
    }
    for (int i = lane_id; i < WMMA_K_FP16 * WMMA_N; i += 32) {
        int row = i / WMMA_N, col = i % WMMA_N;
        Bs[buf][warp_id][row][col] = B_batch[row * N + (tile_col + col)];
    }
    __syncthreads();

    wmma::load_matrix_sync(a_frag, &As[buf][warp_id][0][0], WMMA_K_FP16 + PAD);
    wmma::load_matrix_sync(b_frag, &Bs[buf][warp_id][0][0], WMMA_N + PAD);

    for (int k_off = WMMA_K_FP16; k_off < K; k_off += WMMA_K_FP16) {
        int next = 1 - buf;
        for (int i = lane_id * 8; i < WMMA_M * WMMA_K_FP16; i += 32 * 8) {
            int row = i / WMMA_K_FP16, col = i % WMMA_K_FP16;
            char*       dst      = (char*)&As[next][warp_id][row][col];
            const char* src      = (const char*)&A_batch[(tile_row + row) * K + (k_off + col)];
            unsigned    smem_ptr = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(smem_ptr), "l"(src));
        }
        for (int i = lane_id * 8; i < WMMA_K_FP16 * WMMA_N; i += THREADS_PER_WARP * 8) {
            int row = i / WMMA_N, col = i % WMMA_N;
            char*       dst      = (char*)&Bs[next][warp_id][row][col];
            const char* src      = (const char*)&B_batch[(k_off + row) * N + (tile_col + col)];
            unsigned    smem_ptr = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(smem_ptr), "l"(src));
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        buf = next;
        wmma::load_matrix_sync(a_frag, &As[buf][warp_id][0][0], WMMA_K_FP16 + PAD);
        wmma::load_matrix_sync(b_frag, &Bs[buf][warp_id][0][0], WMMA_N + PAD);
    }
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    float* c_dst = C_batch + tile_row * N + tile_col;
    for (int i = 0; i < c_frag.num_elements; ++i) c_frag.x[i] = alpha * c_frag.x[i];
    wmma::store_matrix_sync(c_dst, c_frag, N, wmma::mem_row_major);
}

// ---------------------------------------------------------------------------
// quantize_and_assign_per_expert_inputs: fp16 → packed int4 activations
//   Two elements per byte, clamped to [-7, 7]
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void quantize_and_assign_per_expert_inputs(
    const half*   __restrict__ input,
    const int*    __restrict__ expert_counts,
    const int*    __restrict__ expert_token_ids,
    uint8_t*      __restrict__ per_expert_wmma_inputs,  // packed int4
    float                      scale_input_act,
    int                        CAP
){
    const int batch              = blockIdx.z;
    const half* input_b          = input              + batch * N * d_model;
    const int*  expert_counts_b  = expert_counts      + batch * num_experts;
    const int*  expert_token_ids_b = expert_token_ids + batch * num_experts * CAP;
    // packed: d_model/2 bytes per row
    uint8_t*    per_expert_b     = per_expert_wmma_inputs + batch * num_experts * CAP * (d_model / 2);

    const int tid     = threadIdx.x;
    const int warp_id = tid / THREADS_PER_WARP;
    const int lane_id = tid % THREADS_PER_WARP;
    const int row_id  = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row_id >= num_experts * CAP) return;

    const int expert_id = row_id / CAP;
    const int slot      = row_id % CAP;
    const int row_base  = row_id * (d_model / 2);  // byte offset

    if (slot < expert_counts_b[expert_id]) {
        const int token_id = expert_token_ids_b[expert_id * CAP + slot];
        if (token_id >= 0 && token_id < N) {
            const int in_base = token_id * d_model;
            // process pairs of elements → pack into one byte
            for (int col = lane_id * 2; col < d_model; col += THREADS_PER_WARP * 2) {
                float v0 = __half2float(input_b[in_base + col]);
                float v1 = __half2float(input_b[in_base + col + 1]);
                int8_t q0 = (int8_t)__float2int_rn(fminf(fmaxf(v0 / scale_input_act, -7.f), 7.f));
                int8_t q1 = (int8_t)__float2int_rn(fminf(fmaxf(v1 / scale_input_act, -7.f), 7.f));
                per_expert_b[row_base + col / 2] = ((uint8_t)(q0 & 0xF)) | (((uint8_t)(q1 & 0xF)) << 4);
            }
        } else {
            for (int col = lane_id; col < d_model / 2; col += THREADS_PER_WARP)
                per_expert_b[row_base + col] = 0;
        }
    } else {
        for (int col = lane_id; col < d_model / 2; col += THREADS_PER_WARP)
            per_expert_b[row_base + col] = 0;
    }
}

// ---------------------------------------------------------------------------
// silu_and_requant_int4 — unchanged from previous version
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void silu_and_requant_int4(
    const int32_t* __restrict__ up_int32,
    const int32_t* __restrict__ gate_int32,
    uint8_t*       __restrict__ out_int4_packed,
    float scale_input_act,
    float scale_up_w,
    float scale_gate_w,
    float scale_mid_act,
    int   total_elements
){
    const int batch         = blockIdx.z;
    const int32_t* up_b     = up_int32        + batch * total_elements;
    const int32_t* gate_b   = gate_int32      + batch * total_elements;
    uint8_t*       out_b    = out_int4_packed + batch * (total_elements / 2);

    const int block_linear  = blockIdx.y * gridDim.x + blockIdx.x;
    const int global_tid    = block_linear * blockDim.x + threadIdx.x;
    const int global_stride = gridDim.x * gridDim.y * blockDim.x;

    const float dequant_up    = scale_input_act * scale_up_w;
    const float dequant_gate  = scale_input_act * scale_gate_w;
    const float inv_scale_mid = 1.0f / scale_mid_act;
    const int total_pairs = total_elements / 2;

    for (int pair_idx = global_tid; pair_idx < total_pairs; pair_idx += global_stride) {
        int idx0 = pair_idx * 2;
        int idx1 = idx0 + 1;

        auto quantize_one = [&](int idx) -> int8_t {
            const float up_f   = (float)up_b[idx]   * dequant_up;
            const float gate_f = (float)gate_b[idx] * dequant_gate;
            const float silu   = gate_f / (1.0f + __expf(-gate_f));
            const float fused  = up_f * silu * inv_scale_mid;
            return (int8_t)__float2int_rn(fminf(fmaxf(fused, -7.f), 7.f));
        };

        int8_t v0 = quantize_one(idx0);
        int8_t v1 = quantize_one(idx1);
        out_b[pair_idx] = ((uint8_t)(v0 & 0xF)) | (((uint8_t)(v1 & 0xF)) << 4);
    }
}

// ---------------------------------------------------------------------------
// top_k_gating, build_per_expert_buffers, clamp_expert_counts,
// combine, zero_final_output_and_expert_counts — all unchanged
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void top_k_gating(
    const float* logits, int* selected_expert_indices, float* selected_expert_weights,
    float* max_vals, int* max_idxs
){
    int batch = blockIdx.z;
    const float* logits_batch          = logits + batch * N * num_experts;
    int*   sel_idx_b                   = selected_expert_indices + batch * N * k;
    float* sel_w_b                     = selected_expert_weights + batch * N * k;
    const int tid = threadIdx.x, warp_id = tid/THREADS_PER_WARP, lane_id = tid%THREADS_PER_WARP;
    const int token_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (token_id >= N) return;
    float* warp_max_vals = max_vals + warp_id * k;
    int*   warp_max_idxs = max_idxs + warp_id * k;
    const float* logits_row = logits_batch + token_id * num_experts;
    if (lane_id == 0) {
        for (int i = 0; i < k; ++i) { warp_max_vals[i] = -1e20f; warp_max_idxs[i] = -1; }
        for (int e = 0; e < num_experts; ++e) {
            float val = logits_row[e];
            if (val > warp_max_vals[k-1]) {
                warp_max_vals[k-1] = val; warp_max_idxs[k-1] = e;
                for (int i = k-1; i > 0 && warp_max_vals[i] > warp_max_vals[i-1]; --i) {
                    float tv = warp_max_vals[i-1]; warp_max_vals[i-1] = warp_max_vals[i]; warp_max_vals[i] = tv;
                    int   ti = warp_max_idxs[i-1]; warp_max_idxs[i-1] = warp_max_idxs[i]; warp_max_idxs[i] = ti;
                }
            }
        }
        float mv = warp_max_vals[0], se = 0.f;
        for (int l = 0; l < k; ++l) se += expf(warp_max_vals[l] - mv);
        for (int l = 0; l < k; ++l) {
            sel_idx_b[token_id*k+l] = warp_max_idxs[l];
            sel_w_b  [token_id*k+l] = expf(warp_max_vals[l]-mv)/(se+1e-10f);
        }
    }
}

static __device__ __forceinline__ void build_per_expert_buffers(
    const int* __restrict__ sel_idx, const float* __restrict__ sel_w,
    int* __restrict__ counts, int* __restrict__ tok_ids, float* __restrict__ tok_w, int CAP
){
    const int batch = blockIdx.z;
    const int* si = sel_idx + batch*N*k; const float* sw = sel_w + batch*N*k;
    int* c = counts + batch*num_experts; int* ti = tok_ids + batch*num_experts*CAP;
    float* tw = tok_w + batch*num_experts*CAP;
    const int tid=threadIdx.x, wid=tid/THREADS_PER_WARP, lid=tid%THREADS_PER_WARP;
    const int wl=blockIdx.x*WARPS_PER_BLOCK+wid, ws=gridDim.x*WARPS_PER_BLOCK;
    for (int r=wl; r<N*k; r+=ws) if (lid==0) {
        int tok=r/k, exp=si[r];
        if (exp>=0 && exp<num_experts) {
            int slot=atomicAdd(&c[exp],1);
            if (slot<CAP) { ti[exp*CAP+slot]=tok; tw[exp*CAP+slot]=sw[r]; }
        }
    }
}

static __device__ __forceinline__ void clamp_expert_counts(int* __restrict__ counts, int CAP){
    const int batch=blockIdx.z; int* c=counts+batch*num_experts;
    const int gt=blockIdx.x*blockDim.x+threadIdx.x, gs=gridDim.x*blockDim.x;
    for (int i=gt; i<num_experts; i+=gs) if (c[i]>CAP) c[i]=CAP;
}

static __device__ __forceinline__ void combine(
    const int32_t* __restrict__ input, const int* __restrict__ tok_ids,
    const float* __restrict__ tok_w, const int* __restrict__ counts,
    float* final_output, float scale_mid_act, float scale_down_w, int CAP
){
    const int batch=blockIdx.z; const float dq=scale_mid_act*scale_down_w;
    const int32_t* ib=input+batch*num_experts*CAP*d_model;
    const int* ti=tok_ids+batch*num_experts*CAP; const float* tw=tok_w+batch*num_experts*CAP;
    const int* cb=counts+batch*num_experts; float* fb=final_output+batch*N*d_model;
    const int tid=threadIdx.x,wid=tid/THREADS_PER_WARP,lid=tid%THREADS_PER_WARP;
    const int row_id=blockIdx.x*WARPS_PER_BLOCK+wid;
    if (row_id>=num_experts*CAP) return;
    const int eid=row_id/CAP, slot=row_id%CAP;
    if (slot>=cb[eid]) return;
    const int token_id=ti[eid*CAP+slot];
    if (token_id<0||token_id>=N) return;
    const float rw=tw[eid*CAP+slot];
    for (int col=lid; col<d_model; col+=THREADS_PER_WARP)
        atomicAdd(&fb[token_id*d_model+col], rw*(float)ib[row_id*d_model+col]*dq);
}

static __device__ __forceinline__ void zero_final_output_and_expert_counts(
    float* __restrict__ fo, int* __restrict__ ec
){
    const int batch=blockIdx.z; if (blockIdx.y!=0) return;
    float* fb=fo+batch*N*d_model; int* cb=ec+batch*num_experts;
    const int gt=blockIdx.x*blockDim.x+threadIdx.x, gs=gridDim.x*blockDim.x;
    for (int i=gt; i<num_experts; i+=gs) cb[i]=0;
    for (int i=gt; i<N*d_model;   i+=gs) fb[i]=0.f;
}

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------
__global__ void capacity(MoEArgs args){

    const half*    input                    = args.input;
    const half*    router_weights           = args.router_weights;
    // weights: packed int4, B-transposed layout [N/2][K/2] per expert
    const uint8_t* expert_up_proj_weights   = args.expert_up_proj_weights_int4;
    const uint8_t* expert_gate_proj_weights = args.expert_gate_proj_weights_int4;
    const uint8_t* expert_down_proj_weights = args.expert_down_proj_weights_int4;

    const float scale_up_w      = args.scale_up_w;
    const float scale_gate_w    = args.scale_gate_w;
    const float scale_down_w    = args.scale_down_w;
    const float scale_input_act = args.scale_input_act;
    const float scale_mid_act   = args.scale_mid_act;

    float*     expert_logits           = args.expert_logits;
    int*       selected_expert_indices = args.selected_expert_indices;
    float*     selected_expert_weights = args.selected_expert_weights;
    int*       expert_counts           = args.expert_counts;
    int*       expert_token_ids        = args.expert_token_ids;
    float*     expert_token_weights    = args.expert_token_weights;
    uint8_t*   per_expert_wmma_inputs  = args.per_expert_wmma_inputs_int4;  // packed int4
    uint8_t*   hidden_mlp_int4         = args.hidden_mlp_layer_1_out_int4;
    float*     final_output            = args.final_output;

    int32_t* up_int32   = reinterpret_cast<int32_t*>(args.hidden_mlp_layer_1_out);
    int32_t* gate_int32 = reinterpret_cast<int32_t*>(args.hidden_mlp_gate_out);
    int32_t* down_int32 = reinterpret_cast<int32_t*>(args.hidden_mlp_layer_2_out);

    int CAP_raw = (int)ceilf((float)N*(float)k/(float)num_experts*capacity_factor);
    int CAP     = ((CAP_raw + WMMA_M - 1) / WMMA_M) * WMMA_M;

    zero_final_output_and_expert_counts(final_output, expert_counts);
    wmma_db_fp16(1.0f, input, router_weights, expert_logits, N, num_experts, d_model);

    __shared__ float max_vals[WARPS_PER_BLOCK * k];
    __shared__ int   max_indices[WARPS_PER_BLOCK * k];
    top_k_gating(expert_logits, selected_expert_indices, selected_expert_weights, max_vals, max_indices);
    __syncthreads();

    build_per_expert_buffers(selected_expert_indices, selected_expert_weights,
                             expert_counts, expert_token_ids, expert_token_weights, CAP);
    __syncthreads();
    clamp_expert_counts(expert_counts, CAP);
    __syncthreads();

    // Stage 4: fp16 → packed int4 activations
    quantize_and_assign_per_expert_inputs(input, expert_counts, expert_token_ids,
                                          per_expert_wmma_inputs, scale_input_act, CAP);
    __syncthreads();

    // Stage 5: up_proj + gate_proj int4×int4 GEMMs
    {
        int tid=threadIdx.x, wid=tid/THREADS_PER_WARP;
        int wtr=wid/WARP_TILES_X, wtc=wid%WARP_TILES_X;
        const int M=num_experts*CAP, Nw=up_proj_dim*d_model, K=d_model;
        const int tile_row=blockIdx.y*(WMMA_M*WARP_TILES_Y)+wtr*WMMA_M;
        const int tile_col=blockIdx.x*(WMMA_N*WARP_TILES_X)+wtc*WMMA_N;
        if (tile_row<M && tile_col<Nw) {
            wmma_db_int4<true>(per_expert_wmma_inputs, expert_up_proj_weights,
                               up_int32,   M, Nw, K, tile_row, tile_col);
            wmma_db_int4<true>(per_expert_wmma_inputs, expert_gate_proj_weights,
                               gate_int32, M, Nw, K, tile_row, tile_col);
        }
    }
    __syncthreads();

    // Stage 6: dequant + SiLU + requant → packed int4
    silu_and_requant_int4(up_int32, gate_int32, hidden_mlp_int4,
                          scale_input_act, scale_up_w, scale_gate_w, scale_mid_act,
                          num_experts * CAP * up_proj_dim * d_model);
    __syncthreads();

    // Stage 7: down_proj int4×int4 GEMM
    {
        int tid=threadIdx.x, wid=tid/THREADS_PER_WARP;
        int wtr=wid/WARP_TILES_X, wtc=wid%WARP_TILES_X;
        const int M=num_experts*CAP, Nd=d_model, K=up_proj_dim*d_model;
        const int tile_row=blockIdx.y*(WMMA_M*WARP_TILES_Y)+wtr*WMMA_M;
        const int tile_col=blockIdx.x*(WMMA_N*WARP_TILES_X)+wtc*WMMA_N;
        if (tile_row<M && tile_col<Nd) {
            wmma_db_int4<true>(hidden_mlp_int4, expert_down_proj_weights,
                               down_int32, M, Nd, K, tile_row, tile_col);
        }
    }
    __syncthreads();

    combine(down_int32, expert_token_ids, expert_token_weights, expert_counts,
            final_output, scale_mid_act, scale_down_w, CAP);
}

#include "launcher.h"