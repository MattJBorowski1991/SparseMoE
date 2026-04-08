#include "include/config.h"
#include "include/moe_args.h"
#include <mma.h>
using namespace nvcuda;
#include <stdio.h>
#include <assert.h>
#include <cstdint>

#define MOE_KERNEL capacity
#define MOE_USES_CAPACITY 1

constexpr int PTX_MMA_K = 32;

// ******** one warp produces a 16x16 C tile in two chunks: left n8 (16x8) and right n8 (16x8) ********


// ---------------------------------------------------------------------------
// PTX mma.sync constants for int8 on sm_89 (Ada):
//   shape:      m16n8k32
//   A:          16x32 int8, row-major  → 4x uint32_t per thread (packed)
//   B:          32x8  int8, col-major  → 4x uint32_t per thread (packed)
//   C/D:        16x8  int32           → 4x int32_t  per thread
//   Two ops per warp tile to cover m16n16k32 (tile two n8 side by side)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// ptx_store_int32x4: inline PTX store of 4 int32 registers to smem/global
// ---------------------------------------------------------------------------
// static __device__ __forceinline__ void ptx_store_int32x4(
//     int32_t* dst,
//     int32_t d0, int32_t d1, int32_t d2, int32_t d3
// ){
//     asm volatile(
//         "st.global.v4.s32 [%0], {%1, %2, %3, %4};"
//         :: "l"(dst), "r"(d0), "r"(d1), "r"(d2), "r"(d3)
//     );
// }

// ---------------------------------------------------------------------------
// ptx_mma_m16n8k32_s8: one m16n8k32 int8->int32 mma.sync
//   a[4]:  4x uint32 = 4x (4x int8) = 16 elems of A = packed A = owned by one lane
//   b[2]:  2x uint32 = 2x (4x int8) = 8 elems of B = packed B
//   c[4]:  4x int32  accumulator in, updated in place
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void ptx_mma_m16n8k32_s8(
    const uint32_t a[4],
    const uint32_t b[2],
    int32_t        c[4]
){
    asm volatile(   // issue one wmma via mma.sync & tie compiler registers to ptx operands so results accumulates into c[]
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " 
        // row.col == A is row-major, B is col-major. this affects how packed bytes map into operand registers
        // s32.s8.s8.s32 = types for DABC, where D = A@B + C; D=destination, C=accumulator. s32 = signed 32-bit.
        "{%0,%1,%2,%3}, "   // D registers = 4 x 32-bit regs
        "{%4,%5,%6,%7}, "   // A registers = 4 x 32-bit regs
        "{%8,%9}, "         // B registers = 2 x 32-bit regs
        "{%0,%1,%2,%3};"    // C registers (same as for D!)
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])    // "+r" = read and write to a general-purpose register
        :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),
           "r"(b[0]),  "r"(b[1])
    );
}

// *** Load A tile the way mma.sync expects ***
// ---------------------------------------------------------------------------
// load_a_m16k32: load A tile (16x32 int8, row-major) into 4 uint32 registers (per lane)
// all 32 lanes cover the full tile
//   smem ptr: points to [WMMA_M][WMMA_K] int8 tile for this warp
//   Each thread in the warp owns a specific row/col set per PTX layout rules:
//     thread groups of 4, each group covers 2 rows x 16 cols
//   Layout (m16k32 row-major, per PTX spec):
//     a[0]: row (lane/4)*2+0, cols  0-3   packed
//     a[1]: row (lane/4)*2+0, cols 16-19  packed  (k>=16 half)
//     a[2]: row (lane/4)*2+1, cols  0-3   packed
//     a[3]: row (lane/4)*2+1, cols 16-19  packed
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void load_a_m16k32(
    const int8_t* tile,   // [WMMA_M][WMMA_K+PAD] in smem
    int           ldA,    // = WMMA_K + PAD
    uint32_t      a[4]
){
    int lane = threadIdx.x % 32;
    int row0 = (lane / 4) * 2;
    int row1 = row0 + 1;
    int col  = (lane % 4) * 4;  // 4 consecutive int8 → 1 uint32

    // vectorized (uint32_t) load
    // load the 4x4 = 16 int8s into packs of 4 = 4 x uint32s
    
    // left half of A (cols 0..15)
    a[0] = *reinterpret_cast<const uint32_t*>(&tile[row0 * ldA + col]);
    a[2] = *reinterpret_cast<const uint32_t*>(&tile[row1 * ldA + col]);
    // right half of A (cols 16..31)
    a[1] = *reinterpret_cast<const uint32_t*>(&tile[row0 * ldA + col + 16]);
    a[3] = *reinterpret_cast<const uint32_t*>(&tile[row1 * ldA + col + 16]);

    //e.g.
    // // I. for lane=0:
    // a[0] holds elems (0,0), .. (0,3)
    // a[2] holds elems (1,0),.. (1,3)
    // a[1] holds elems (0,16), .. (0, 19)
    // a[3] holds elems (1,16), .. (1, 19)

    // // II. for lane=1:
    // a[0] holds elems (0,4), .. (0,7)
    // a[2] holds elems (1,4),.. (1,7)
    // a[1] holds elems (0,20), .. (0, 23)
    // a[3] holds elems (1,20), .. (1, 23)


}


// *** Load B tile the way mma.sync expects ***

// ---------------------------------------------------------------------------
// load_b_m8k32: load B tile (32x8 int8, col-major) into 4 uint32 registers
//   For m16n8k32, B is [K][N] row-major = [32][8] but PTX expects col-major
//   packing. B stored in smem as [WMMA_K][WMMA_N+PAD] row-major (as before),
//   so we read col-major on the fly.
//   PTX layout for B (k32n8, col-major):
//     b[0]: rows (lane%4)*4 .. (lane%4)*4+3,    col n_col_base+(lane/4)  packed
//     b[1]: rows (lane%4)*4+16 .. (lane%4)*4+19, col n_col_base+(lane/4)  packed
//   n_col_base: 0 for first n8, 8 for second n8
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void load_b_m8k32(
    const int8_t* tile,       // [WMMA_K][WMMA_N+PAD] in smem
    int           ldB,        // = WMMA_N + PAD
    int           n_col_base, // 0 or 8 for two n8 ops
    uint32_t      b[2]
){
    int lane = threadIdx.x % 32;
    int col  = n_col_base + (lane / 4);  // n-col this thread owns in [0,7]

    // pack 4 k-rows into one uint32 (int8x4)
    // k-row ordering per PTX m16n8k32 B col-major spec
    // col below is fixed and rows vary as we need to perform col-major packing for B (as PTX requires)
    // "read col-major on the fly"
    auto pack4 = [&](int k0) -> uint32_t {
        uint32_t v = 0;
        // GOAL: pack 4 separate bytes (single int8) into on 32-bit register (thats what mma.sync consumes/requires)
        // (the below 4 bytes are ldB-bytes apart. if they were contiguous we could just use single: v = *reinterpret_cast<const uint32_t*>(&tile[k0 * ldB + col]);)
        // Visually:
        // uint32:  [byte3][byte2][byte1][byte0]   ← bits 31..0
        //           <<24   <<16   <<8    <<0
        v |= ((uint32_t)(uint8_t)tile[(k0+0)*ldB + col]); // "a|=b" == "a = (a|b)" == places each byte in its slot == bitwise OR == compare bits of a and b and return 1 if at least one of the bits is one = return 0 only if both are 0
        v |= ((uint32_t)(uint8_t)tile[(k0+1)*ldB + col]) << 8;
        v |= ((uint32_t)(uint8_t)tile[(k0+2)*ldB + col]) << 16;
        v |= ((uint32_t)(uint8_t)tile[(k0+3)*ldB + col]) << 24;
        return v;
    };

    b[0] = pack4((lane % 4) * 4);
    b[1] = pack4((lane % 4) * 4 + 16);
    // lane%4 selects 4-row group: 0->rows 0-3, 1->rows 4-7, 2->rows 8-11, 3->rows 12-15 (+ 16 for b[1])
    // lane/4 selects n-col: 0->col0, 1->col1, ..., 7->col7
    // hence:
    // lane0 packs in col0: rows 0-3 and 16-19 (into b[0] and b[1] as two int32 4-packs of int8s)
    // lane1 packs in col0: rows 4-7 and 20-23
    // lane2 packs in col0: rows 8-11 and 24-27
    // lane3 packs in col0: rows 12-15 and 28-31
    // lane4 packs in col1: rows 0-3 and 16-19
    // etc
}

// ---------------------------------------------------------------------------
// wmma_db_int8 — PTX version, double-buffered, two m16n8k32 ops per warp tile
//   covers m16n16k32 by issuing op0 (n_col_base=0) + op1 (n_col_base=8)
//   D0[4] covers left  n8 of the m16n16 tile
//   D1[4] covers right n8 of the m16n16 tile
//   Both stored to global int32 scratch via ptx_store_int32x4
// ---------------------------------------------------------------------------
template<bool calculatePerExpert>
static __device__ __forceinline__ void wmma_db_int8(
    const int8_t* A,
    const int8_t* B,
    int32_t*      C_global,   // output: [M, N] int32 global scratch
    int M, int N, int K,
    int tile_row_in,
    int tile_col_in
){
    assert((K % PTX_MMA_K) == 0);

    int batch   = blockIdx.z;
    int tid     = threadIdx.x;
    int warp_id = tid / THREADS_PER_WARP;
    int lane_id = tid % THREADS_PER_WARP;

    const int8_t* A_batch = A + batch * M * K;
    const int8_t* B_batch = B;

    const int8_t* A_e;
    const int8_t* B_e;
    int tile_row_local = tile_row_in;

    if constexpr (calculatePerExpert) {
        const int rows_per_expert = M / num_experts;
        const int expert_id       = tile_row_in / rows_per_expert;
        tile_row_local            = tile_row_in % rows_per_expert;
        A_e = A_batch + expert_id * rows_per_expert * K;
        B_e = B_batch + expert_id * K * N;
    } else {
        A_e = A_batch;
        B_e = B_batch;
    }

    __shared__ __align__(16) int8_t As8[2][WARPS_PER_BLOCK][WMMA_M][PTX_MMA_K + PAD];
    __shared__ __align__(16) int8_t Bs8[2][WARPS_PER_BLOCK][PTX_MMA_K][WMMA_N + PAD];

    // accumulators: two m16n8k32 ops → D0 (left n8), D1 (right n8)
    int32_t D0[4] = {0, 0, 0, 0};
    int32_t D1[4] = {0, 0, 0, 0};

    int buf = 0;

    // --- initial load ---
    for (int i = lane_id; i < WMMA_M * PTX_MMA_K; i += 32) {
        int row = i / PTX_MMA_K, col = i % PTX_MMA_K;
        As8[buf][warp_id][row][col] = A_e[(tile_row_local + row) * K + col];
    }
    for (int i = lane_id; i < PTX_MMA_K * WMMA_N; i += 32) {
        int row = i / WMMA_N, col = i % WMMA_N;
        Bs8[buf][warp_id][row][col] = B_e[row * N + (tile_col_in + col)];
    }
    __syncthreads();

    // load registers from initial smem tile
    uint32_t a[4], b0[2], b1[2];
    load_a_m16k32 (&As8[buf][warp_id][0][0], PTX_MMA_K + PAD, a);
    load_b_m8k32  (&Bs8[buf][warp_id][0][0], WMMA_N + PAD, 0, b0);
    load_b_m8k32  (&Bs8[buf][warp_id][0][0], WMMA_N + PAD, 8, b1);

    // --- main loop ---
    for (int k = PTX_MMA_K; k < K; k += PTX_MMA_K) {
        int next = 1 - buf;

        // async load next tile
        for (int i = lane_id * 16; i < WMMA_M * PTX_MMA_K; i += THREADS_PER_WARP * 16) {
            int row = i / PTX_MMA_K, col = i % PTX_MMA_K;
            char*       dst = (char*)&As8[next][warp_id][row][col];
            const char* src = (const char*)&A_e[(tile_row_local + row) * K + (k + col)];
            unsigned smem_ptr = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(smem_ptr), "l"(src));
        }
        for (int i = lane_id * 16; i < PTX_MMA_K * WMMA_N; i += THREADS_PER_WARP * 16) {
            int row = i / WMMA_N, col = i % WMMA_N;
            char*       dst = (char*)&Bs8[next][warp_id][row][col];
            const char* src = (const char*)&B_e[(k + row) * N + (tile_col_in + col)];
            unsigned smem_ptr = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(smem_ptr), "l"(src));
        }
        asm volatile("cp.async.commit_group;");

        // compute on current tile (two m16n8k32 ops)
        ptx_mma_m16n8k32_s8(a, b0, D0);
        ptx_mma_m16n8k32_s8(a, b1, D1);
        asm volatile("cp.async.wait_group 0;");

        // load next tile into registers
        buf = next;
        load_a_m16k32(&As8[buf][warp_id][0][0], PTX_MMA_K + PAD, a);
        load_b_m8k32 (&Bs8[buf][warp_id][0][0], WMMA_N + PAD, 0, b0);
        load_b_m8k32 (&Bs8[buf][warp_id][0][0], WMMA_N + PAD, 8, b1);
    }

    // compute last tile
    ptx_mma_m16n8k32_s8(a, b0, D0);
    ptx_mma_m16n8k32_s8(a, b1, D1);

    // --- store D0 (left n8) and D1 (right n8) to global int32 scratch ---
    // hand-coded according to how it is specified in PTX ISA: 
    // hardware delivers each D reg to a specific (lane, reg)->(row, col) position:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-s8
    // PTX m16n8 output layout: each thread owns 2 rows x 1 col (4 elements total)
    // row0 = (lane/4)*2, row1 = (lane/4)*2+1, col = lane%4
    {
        int row0     = (lane_id / 4) * 2;
        int row1     = row0 + 1;
        int col_left = lane_id % 4;           // col within left  n8
        int col_right= lane_id % 4;           // col within right n8

        // D[0],D[1] → row0: left col, right col
        // D[2],D[3] → row1: left col, right col
        int32_t* base = C_global + (blockIdx.z * M + tile_row_in) * N + tile_col_in;

        // left n8 block
        base[(row0) * N + col_left]     = D0[0];
        base[(row0) * N + col_left + 4] = D0[1];  // second group of 4 cols
        base[(row1) * N + col_left]     = D0[2];
        base[(row1) * N + col_left + 4] = D0[3];

        // right n8 block (offset +8 in N)
        base[(row0) * N + 8 + col_right]     = D1[0];
        base[(row0) * N + 8 + col_right + 4] = D1[1];
        base[(row1) * N + 8 + col_right]     = D1[2];
        base[(row1) * N + 8 + col_right + 4] = D1[3];
    }
}


// ---------------------------------------------------------------------------
// wmma_db_fp16: router matmul only — stays fp16 (router is not bandwidth bottleneck)
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void wmma_db_fp16(
    float alpha,
    const half* A,
    const half* B,
    float* C,
    int M, int N, int K
){
    assert( (M % WMMA_M == 0) && (N % WMMA_N == 0) && (K % WMMA_K == 0) );
    int batch = blockIdx.z;

    const half* A_batch = A + batch * M * K;
    const half* B_batch = B;
    float*      C_batch = C + batch * M * N;

    int tid        = threadIdx.x;
    int warp_id    = tid / THREADS_PER_WARP;
    int lane_id    = tid % THREADS_PER_WARP;

    int warp_tile_row = warp_id / WARP_TILES_X;
    int warp_tile_col = warp_id % WARP_TILES_X;
    const int tile_row = blockIdx.y * (WMMA_M * WARP_TILES_Y) + warp_tile_row * WMMA_M;
    const int tile_col = blockIdx.x * (WMMA_N * WARP_TILES_X) + warp_tile_col * WMMA_N;
    if (tile_row >= M || tile_col >= N) return;

    __shared__ __align__(16) half As[2][WARPS_PER_BLOCK][WMMA_M][WMMA_K + PAD];
    __shared__ __align__(16) half Bs[2][WARPS_PER_BLOCK][WMMA_K][WMMA_N + PAD];

    wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>                 c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int buf = 0;

    for (int i = lane_id; i < WMMA_M * WMMA_K; i += 32) {
        int row = i / WMMA_K, col = i % WMMA_K;
        As[buf][warp_id][row][col] = A_batch[(tile_row + row) * K + col];
    }
    for (int i = lane_id; i < WMMA_K * WMMA_N; i += 32) {
        int row = i / WMMA_N, col = i % WMMA_N;
        Bs[buf][warp_id][row][col] = B_batch[row * N + (tile_col + col)];
    }
    __syncthreads();

    wmma::load_matrix_sync(a_frag, &As[buf][warp_id][0][0], WMMA_K + PAD);
    wmma::load_matrix_sync(b_frag, &Bs[buf][warp_id][0][0], WMMA_N + PAD);

    for (int k = WMMA_K; k < K; k += WMMA_K) {
        int next = 1 - buf;

        for (int i = lane_id * 8; i < WMMA_M * WMMA_K; i += 32 * 8) {
            int row = i / WMMA_K, col = i % WMMA_K;
            char*       dst      = (char*)&As[next][warp_id][row][col];
            const char* src      = (const char*)&A_batch[(tile_row + row) * K + (k + col)];
            unsigned    smem_ptr = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(smem_ptr), "l"(src));
        }
        for (int i = lane_id * 8; i < WMMA_K * WMMA_N; i += THREADS_PER_WARP * 8) {
            int row = i / WMMA_N, col = i % WMMA_N;
            char*       dst      = (char*)&Bs[next][warp_id][row][col];
            const char* src      = (const char*)&B_batch[(k + row) * N + (tile_col + col)];
            unsigned    smem_ptr = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(smem_ptr), "l"(src));
        }
        asm volatile("cp.async.commit_group;");
        __syncthreads();

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        asm volatile("cp.async.wait_group 0;");

        buf = next;
        wmma::load_matrix_sync(a_frag, &As[buf][warp_id][0][0], WMMA_K + PAD);
        wmma::load_matrix_sync(b_frag, &Bs[buf][warp_id][0][0], WMMA_N + PAD);
    }
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    float* c_dst = C_batch + tile_row * N + tile_col;
    for (int i = 0; i < c_frag.num_elements; ++i) c_frag.x[i] = alpha * c_frag.x[i];
    wmma::store_matrix_sync(c_dst, c_frag, N, wmma::mem_row_major);
}


// ---------------------------------------------------------------------------
// quantize_input_to_int8:
//   Converts fp16 input activations → int8 per_expert_wmma_inputs in one pass.
//   Replaces assign_per_expert_wmma_inputs + does quantization.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void quantize_and_assign_per_expert_inputs(
    const half* __restrict__  input,
    const int*  __restrict__  expert_counts,
    const int*  __restrict__  expert_token_ids,
    int8_t*     __restrict__  per_expert_wmma_inputs,
    float                     scale_input_act,
    int                       CAP
){
    const int batch = blockIdx.z;
    const half* input_b              = input + batch * N * d_model;
    const int*  expert_counts_b      = expert_counts + batch * num_experts;
    const int*  expert_token_ids_b   = expert_token_ids + batch * num_experts * CAP;
    int8_t*     per_expert_wmma_inputs_b         = per_expert_wmma_inputs + batch * num_experts * CAP * d_model;

    const int tid     = threadIdx.x;
    const int warp_id = tid / THREADS_PER_WARP;
    const int lane_id = tid % THREADS_PER_WARP;
    const int row_id  = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row_id >= num_experts * CAP) return;

    const int expert_id = row_id / CAP;
    const int slot      = row_id % CAP;
    const int row_base  = row_id * d_model;

    if (slot < expert_counts_b[expert_id]) {
        const int token_id = expert_token_ids_b[expert_id * CAP + slot];
        if (token_id >= 0 && token_id < N) {
            const int in_base = token_id * d_model;
            for (int col = lane_id; col < d_model; col += THREADS_PER_WARP) {
                float val = __half2float(input_b[in_base + col]);
                per_expert_wmma_inputs_b[row_base + col] = (int8_t)__float2int_rn(
                    fminf(fmaxf(val / scale_input_act, -128.f), 127.f));
            }
        } else {
            for (int col = lane_id; col < d_model; col += THREADS_PER_WARP)
                per_expert_wmma_inputs_b[row_base + col] = (int8_t)0;
        }
    } else {
        for (int col = lane_id; col < d_model; col += THREADS_PER_WARP)
            per_expert_wmma_inputs_b[row_base + col] = (int8_t)0;
    }
}

// ---------------------------------------------------------------------------
// silu_and_requant:
//   Fused: dequant int32 up+gate results → SiLU(up)*gate → requant to int8.
//   Replaces fp32_to_fp16. Writes directly to hidden_mlp_layer_1_out_int8.
//   Called after up_proj and gate_proj wmma_db_int8 results are stored as int32
//   into two temporary smem tiles — but since we can't hold two full int32 tiles
//   across the whole [num_experts, CAP, 4*d_model] space in smem, we write
//   the int32 results to global as int32 (reusing the now-freed fp32 buffer
//   pointers, cast), then fuse here.
//   Strategy: pass int32 global scratch for up and gate, fuse in this kernel.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void silu_and_requant(
    const int32_t* __restrict__ up_int32,       // [num_experts, CAP, 4*d_model]
    const int32_t* __restrict__ gate_int32,     // [num_experts, CAP, 4*d_model]
    int8_t*        __restrict__ out_int8,        // [num_experts, CAP, 4*d_model]
    float scale_input_act,
    float scale_up_w,
    float scale_gate_w,
    float scale_mid_act,
    int   total_size
){
    const int batch          = blockIdx.z;
    const int up_size_batch  = total_size; // per batch
    const int32_t* up_b      = up_int32   + batch * up_size_batch;
    const int32_t* gate_b    = gate_int32 + batch * up_size_batch;
    int8_t*        out_b     = out_int8   + batch * up_size_batch;

    const int block_linear  = blockIdx.y * gridDim.x + blockIdx.x;
    const int global_tid    = block_linear * blockDim.x + threadIdx.x;
    const int global_stride = gridDim.x * gridDim.y * blockDim.x;

    const float dequant_up   = scale_input_act * scale_up_w;
    const float dequant_gate = scale_input_act * scale_gate_w;
    const float inv_scale_mid = 1.0f / scale_mid_act;

    for (int idx = global_tid; idx < total_size; idx += global_stride) {
        const float up_f   = (float)up_b[idx]   * dequant_up;
        const float gate_f = (float)gate_b[idx] * dequant_gate;
        const float silu   = gate_f / (1.0f + __expf(-gate_f));  // SiLU on gate
        const float fused  = up_f * silu;
        out_b[idx] = (int8_t)__float2int_rn(fminf(fmaxf(fused * inv_scale_mid, -128.f), 127.f));
    }
}

// ---------------------------------------------------------------------------
// top_k_gating — unchanged
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void top_k_gating(
    const float* logits,
    int*   selected_expert_indices,
    float* selected_expert_weights,
    float* max_vals,
    int*   max_idxs
){
    int batch = blockIdx.z;
    const float* logits_batch                 = logits + batch * N * num_experts;
    int*         selected_expert_indices_b    = selected_expert_indices + batch * N * k;
    float*       selected_expert_weights_b    = selected_expert_weights + batch * N * k;

    const int tid     = threadIdx.x;
    const int warp_id = tid / THREADS_PER_WARP;
    const int lane_id = tid % THREADS_PER_WARP;
    const int token_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (token_id >= N) return;

    float* warp_max_vals = max_vals  + warp_id * k;
    int*   warp_max_idxs = max_idxs  + warp_id * k;
    const float* logits_row = logits_batch + token_id * num_experts;

    if (lane_id == 0) {
        for (int i = 0; i < k; ++i) { warp_max_vals[i] = -1e20f; warp_max_idxs[i] = -1; }
        for (int logit_id = 0; logit_id < num_experts; ++logit_id) {
            float val = logits_row[logit_id];
            if (val > warp_max_vals[k - 1]) {
                warp_max_vals[k - 1] = val; warp_max_idxs[k - 1] = logit_id;
                for (int i = k - 1; i > 0 && warp_max_vals[i] > warp_max_vals[i - 1]; --i) {
                    float tv = warp_max_vals[i-1]; warp_max_vals[i-1] = warp_max_vals[i]; warp_max_vals[i] = tv;
                    int   ti = warp_max_idxs[i-1]; warp_max_idxs[i-1] = warp_max_idxs[i]; warp_max_idxs[i] = ti;
                }
            }
        }
        float max_val = warp_max_vals[0], sum_of_exps = 0.0f;
        for (int l = 0; l < k; ++l) sum_of_exps += expf(warp_max_vals[l] - max_val);
        for (int l = 0; l < k; ++l) {
            selected_expert_indices_b[token_id * k + l]  = warp_max_idxs[l];
            selected_expert_weights_b[token_id * k + l]  = expf(warp_max_vals[l] - max_val) / (sum_of_exps + 1e-10f);
        }
    }
}

// ---------------------------------------------------------------------------
// build_per_expert_buffers — unchanged
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void build_per_expert_buffers(
    const int*   __restrict__ selected_expert_indices,
    const float* __restrict__ selected_expert_weights,
    int*         __restrict__ expert_counts,
    int*         __restrict__ expert_token_ids,
    float*       __restrict__ expert_token_weights,
    int CAP
){
    const int batch = blockIdx.z;
    const int* sel_idx_b  = selected_expert_indices  + batch * N * k;
    const float* sel_w_b  = selected_expert_weights  + batch * N * k;
    int*   counts_b       = expert_counts            + batch * num_experts;
    int*   tok_ids_b      = expert_token_ids         + batch * num_experts * CAP;
    float* tok_w_b        = expert_token_weights     + batch * num_experts * CAP;

    const int tid = threadIdx.x, warp_id = tid / THREADS_PER_WARP, lane_id = tid % THREADS_PER_WARP;
    const int warp_linear = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int warp_stride = gridDim.x  * WARPS_PER_BLOCK;

    for (int route_id = warp_linear; route_id < N * k; route_id += warp_stride) {
        if (lane_id == 0) {
            const int token_id  = route_id / k;
            const int expert_id = sel_idx_b[route_id];
            if (expert_id >= 0 && expert_id < num_experts) {
                const int slot = atomicAdd(&counts_b[expert_id], 1);
                if (slot < CAP) {
                    tok_ids_b[expert_id * CAP + slot] = token_id;
                    tok_w_b  [expert_id * CAP + slot] = sel_w_b[route_id];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// clamp_expert_counts — unchanged
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void clamp_expert_counts(int* __restrict__ expert_counts, int CAP){
    const int batch  = blockIdx.z;
    int* counts_b    = expert_counts + batch * num_experts;
    const int global_tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_stride = gridDim.x  * blockDim.x;
    for (int idx = global_tid; idx < num_experts; idx += global_stride) {
        if (counts_b[idx] > CAP) counts_b[idx] = CAP;
    }
}

// ---------------------------------------------------------------------------
// combine: dequantize int32 down_proj result and accumulate into final output.
//   scale applied here — no requant needed (final output is fp32).
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void combine(
    const int32_t* __restrict__ input,              // int32 down_proj output [num_experts, CAP, d_model]
    const int*     __restrict__ expert_token_ids,
    const float*   __restrict__ expert_token_weights,
    const int*     __restrict__ expert_counts,
    float*                      final_output,
    float scale_mid_act,
    float scale_down_w,
    int   CAP
){
    const int batch = blockIdx.z;
    const float dequant = scale_mid_act * scale_down_w;
    const int rows_per_expert = CAP;

    const int32_t* input_b       = input              + batch * num_experts * rows_per_expert * d_model;
    const int*     tok_ids_b     = expert_token_ids   + batch * num_experts * rows_per_expert;
    const float*   tok_w_b       = expert_token_weights + batch * num_experts * rows_per_expert;
    const int*     counts_b      = expert_counts       + batch * num_experts;
    float*         final_b       = final_output        + batch * N * d_model;

    const int tid     = threadIdx.x;
    const int warp_id = tid / THREADS_PER_WARP;
    const int lane_id = tid % THREADS_PER_WARP;
    const int row_id  = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row_id >= num_experts * rows_per_expert) return;

    const int expert_id = row_id / rows_per_expert;
    const int slot      = row_id % rows_per_expert;
    if (slot >= counts_b[expert_id]) return;

    const int token_id    = tok_ids_b[expert_id * rows_per_expert + slot];
    if (token_id < 0 || token_id >= N) return;

    const float route_weight    = tok_w_b[expert_id * rows_per_expert + slot];
    const int   expert_row_base = row_id  * d_model;
    const int   token_row_base  = token_id * d_model;

    for (int col = lane_id; col < d_model; col += THREADS_PER_WARP) {
        float val = (float)input_b[expert_row_base + col] * dequant;
        atomicAdd(&final_b[token_row_base + col], route_weight * val);
    }
}

// ---------------------------------------------------------------------------
// zero_final_output_and_expert_counts — unchanged
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void zero_final_output_and_expert_counts(
    float* __restrict__ final_output,
    int*   __restrict__ expert_counts
){
    const int batch = blockIdx.z;
    if (blockIdx.y != 0) return;

    float* final_b  = final_output  + batch * N * d_model;
    int*   counts_b = expert_counts + batch * num_experts;

    const int global_tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_stride = gridDim.x  * blockDim.x;

    for (int idx = global_tid; idx < num_experts; idx += global_stride)
        counts_b[idx] = 0;
    for (int idx = global_tid; idx < N * d_model; idx += global_stride)
        final_b[idx] = 0.0f;
}

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------
__global__ void capacity(MoEArgs args){

    const half*    input                   = args.input;
    const half*    router_weights          = args.router_weights;
    const int8_t*  expert_up_proj_weights  = args.expert_up_proj_weights_int8;
    const int8_t*  expert_gate_proj_weights= args.expert_gate_proj_weights_int8;
    const int8_t*  expert_down_proj_weights= args.expert_down_proj_weights_int8;

    const float scale_up_w        = args.scale_up_w;
    const float scale_gate_w      = args.scale_gate_w;
    const float scale_down_w      = args.scale_down_w;
    const float scale_input_act   = args.scale_input_act;
    const float scale_mid_act     = args.scale_mid_act;

    float*     expert_logits          = args.expert_logits;
    int*       selected_expert_indices= args.selected_expert_indices;
    float*     selected_expert_weights= args.selected_expert_weights;
    int*       expert_counts          = args.expert_counts;
    int*       expert_token_ids       = args.expert_token_ids;
    float*     expert_token_weights   = args.expert_token_weights;
    int8_t*    per_expert_wmma_inputs = args.per_expert_wmma_inputs_int8;
    int8_t*    hidden_mlp_int8        = args.hidden_mlp_layer_1_out_int8; // post SiLU, int8
    float*     final_output           = args.final_output;

    // Scratch int32 buffers: reuse host-allocated storage.
    // Host must allocate two int32 buffers of size [num_batches, num_experts, CAP, 4*d_model]
    // and one of size [num_batches, num_experts, CAP, d_model] and pass via args (see note).
    // For clarity cast from the host-side int32* fields (add to MoEArgs if needed):
    int32_t* up_int32   = reinterpret_cast<int32_t*>(args.hidden_mlp_layer_1_out);  // repurposed
    int32_t* gate_int32 = reinterpret_cast<int32_t*>(args.hidden_mlp_gate_out);     // repurposed
    int32_t* down_int32 = reinterpret_cast<int32_t*>(args.hidden_mlp_layer_2_out);  // repurposed

    int CAP_raw = (int)ceilf((float)N * (float)k / (float)num_experts * capacity_factor);
    int CAP     = ((CAP_raw + WMMA_M - 1) / WMMA_M) * WMMA_M;

    // --- Stage 0: zero buffers ---
    zero_final_output_and_expert_counts(final_output, expert_counts);

    // --- Stage 1: router GEMM (fp16, small matrix, not bandwidth bottleneck) ---
    wmma_db_fp16(1.0f, input, router_weights, expert_logits, N, num_experts, d_model);

    // --- Stage 2: top-k gating ---
    __shared__ float max_vals[WARPS_PER_BLOCK * k];
    __shared__ int   max_indices[WARPS_PER_BLOCK * k];
    top_k_gating(expert_logits, selected_expert_indices, selected_expert_weights, max_vals, max_indices);
    __syncthreads();

    // --- Stage 3: build per-expert token lists ---
    build_per_expert_buffers(selected_expert_indices, selected_expert_weights,
                              expert_counts, expert_token_ids, expert_token_weights, CAP);
    __syncthreads();

    clamp_expert_counts(expert_counts, CAP);
    __syncthreads();

    // --- Stage 4: quantize fp16 input activations → int8 per-expert tiles ---
    quantize_and_assign_per_expert_inputs(input, expert_counts, expert_token_ids,
                                          per_expert_wmma_inputs, scale_input_act, CAP);
    __syncthreads();

    // --- Stage 5: up_proj and gate_proj int8 GEMMs → int32 scratch ---
    {
        int tid      = threadIdx.x;
        int warp_id  = tid / THREADS_PER_WARP;
        int warp_tile_row = warp_id / WARP_TILES_X;
        int warp_tile_col = warp_id % WARP_TILES_X;
        const int M  = num_experts * CAP;
        const int Nw = up_proj_dim * d_model;
        const int K  = d_model;
        const int tile_row = blockIdx.y * (WMMA_M * WARP_TILES_Y) + warp_tile_row * WMMA_M;
        const int tile_col = blockIdx.x * (WMMA_N * WARP_TILES_X) + warp_tile_col * WMMA_N;

        if (tile_row < M && tile_col < Nw) {
            wmma_db_int8<true>(per_expert_wmma_inputs, expert_up_proj_weights,
                               up_int32, M, Nw, K, tile_row, tile_col);
            wmma_db_int8<true>(per_expert_wmma_inputs, expert_gate_proj_weights,
                               gate_int32, M, Nw, K, tile_row, tile_col);
        }
    }
    __syncthreads();

    // --- Stage 6: fused dequant + SiLU + requant → int8 ---
    silu_and_requant(up_int32, gate_int32, hidden_mlp_int8,
                     scale_input_act, scale_up_w, scale_gate_w, scale_mid_act,
                     num_experts * CAP * up_proj_dim * d_model);
    __syncthreads();

    // --- Stage 7: down_proj int8 GEMM → int32 scratch ---
    {
        int tid      = threadIdx.x;
        int warp_id  = tid / THREADS_PER_WARP;
        int warp_tile_row = warp_id / WARP_TILES_X;
        int warp_tile_col = warp_id % WARP_TILES_X;
        const int M  = num_experts * CAP;
        const int Nd = d_model;
        const int K  = up_proj_dim * d_model;
        const int tile_row = blockIdx.y * (WMMA_M * WARP_TILES_Y) + warp_tile_row * WMMA_M;
        const int tile_col = blockIdx.x * (WMMA_N * WARP_TILES_X) + warp_tile_col * WMMA_N;

        if (tile_row < M && tile_col < Nd) {
            wmma_db_int8<true>(hidden_mlp_int8, expert_down_proj_weights,
                               down_int32, M, Nd, K, tile_row, tile_col);
        }
    }
    __syncthreads();

    // --- Stage 8: dequant + weighted combine → fp32 final output ---
    combine(down_int32, expert_token_ids, expert_token_weights, expert_counts,
            final_output, scale_mid_act, scale_down_w, CAP);
}

#include "launcher.h"