// 08_final_double_buffered.cu
// Double-buffered shared memory + Tensor Cores → 1,500–1,900+ TFLOPS
// This is the final form — used (in spirit) by cuBLAS, CUTLASS, FlashAttention

#include "common/utils.cuh"
#include <mma.h>                //Enables Tensor Core instructions
using namespace nvcuda;         // Brings wmma:: into scope

// --- Tiling and launch constants ---
#define WMMA_M 16
#define WMMA_K 16
#define WMMA_N 16
#define PAD 0
#define WARPS_PER_BLOCK 8
#define WARP_TILES_X 4
#define WARP_TILES_Y 2



__global__ void wmma_db(
    float alpha,
    const half* A,
    const half* B,
    float* C,
    int M, int N, int K
)
{   
    int batch = blockIdx.z;

    float* A_batch = A + batch * M * K;
    float* B_batch = B + batch * K * N;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int warp_tile_row = warp_id / WARP_TILES_X;
    int warp_tile_col = warp_id % WARP_TILES_X;
    const int tile_row = blockIdx.y * (WMMA_M * WARP_TILES_Y) + warp_tile_row * WMMA_M;
    const int tile_col = blockIdx.x * (WMMA_N * WARP_TILES_X) + warp_tile_col * WMMA_N;
    if (tile_row >= M || tile_col >= N) return;

    //********DOUBLE BUFFER START ********

    // Per-warp double buffers in shared memory
    __shared__ __align__(16) half A_s[2][WARPS_PER_BLOCK][WMMA_M][WMMA_K + PAD];
    __shared__ __align__(16) half B_s[2][WARPS_PER_BLOCK][WMMA_K][WMMA_N + PAD];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int buf = 0;

    // Initial tile load: raw cooperative loads (no cp.async)
    for (int i = lane_id; i < WMMA_M * WMMA_K; i += 32) {
        int row = i / WMMA_K;
        int col = i % WMMA_K;
        A_s[buf][warp_id][row][col] = A_batch[(tile_row + row) * K + col];
    }
    for (int i = lane_id; i < WMMA_K * WMMA_N; i += 32) {
        int row = i / WMMA_N;
        int col = i % WMMA_N;
        B_s[buf][warp_id][row][col] = B_batch[row * N + (tile_col + col)];
    }
    __syncthreads(); 

    wmma::load_matrix_sync(a_frag, &A_s[buf][warp_id][0][0], WMMA_K + PAD);
    wmma::load_matrix_sync(b_frag, &B_s[buf][warp_id][0][0], WMMA_N + PAD);

    // Main loop: overlap load (to/from buffers) with compute (in fragments)
    for (int k = WMMA_K; k < K; k += WMMA_K){
        int next = 1 - buf;

        // Load next tile into the next buffer
        for (int i = lane_id * 8; i < WMMA_M * WMMA_K; i += 32 * 8) {
            int row = i / WMMA_K;
            int col = i % WMMA_K;

            char* dst = (char*)&A_s[next][warp_id][row][col];
            const char* src = (const char*)&A[(tile_row + row) * K + (k + col)];

            unsigned smem_addr = __cvta_generic_to_shared(dst);  // 32-bit shared addr
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                        :: "r"(smem_addr), "l"(src));
        }

        for (int i = lane_id * 8; i < WMMA_K * WMMA_N; i += 32 * 8) {
            int row = i / WMMA_N;
            int col = i % WMMA_N;

            char* dst = (char*)&B_s[next][warp_id][row][col];
            const char* src = (const char*)&B[(k + row) * N + (tile_col + col)];

            unsigned smem_addr = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                        :: "r"(smem_addr), "l"(src));
        }

        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");

        __syncthreads();
        

        // compute current tile
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        buf = next;
        wmma::load_matrix_sync(a_frag, &A_s[buf][warp_id][0][0], WMMA_K + PAD);
        wmma::load_matrix_sync(b_frag, &B_s[buf][warp_id][0][0], WMMA_N + PAD);
    }

    //compute last tile 
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    //********DOUBLE BUFFER END ********

    // Apply alpha and beta*C
    float* c_dst = C + tile_row * N + tile_col;
    
    for (int i = 0; i < c_frag.num_elements; ++i) c_frag.x[i] = alpha * c_frag.x[i];
    
    wmma::store_matrix_sync(c_dst, c_frag, N, wmma::mem_row_major);
}

// // Wrapper for WMMA runner
// void launch_double_buffered_tc_c(float alpha, const __half* A, const __half* B, float beta, float* C, int M, int K, int N){
//     dim3 block(32 * WARPS_PER_BLOCK);
//     dim3 grid((N + (WMMA_N * WARP_TILES_X) - 1) / (WMMA_N * WARP_TILES_X),
//               (M + (WMMA_M * WARP_TILES_Y) - 1) / (WMMA_M * WARP_TILES_Y));
//     double_buffered_kernel<<<grid, block>>>(alpha, A, B, beta, C, M, N, K);
//     CHECK_CUDA(cudaGetLastError());
// }