#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuda_fp16.h>

#include <vector>
#include <stdio.h>
#include <cstring>
#include <string>
#include <cstdlib>
#include <sys/stat.h>
#include <errno.h>
#ifdef _WIN32
#include <direct.h>
#endif

#include "../utils/check_cuda.h"
#include "../include/config.h"
#include "../include/moe_args.h"
#include "../inputs/data.h"

int THREADS = 1024;         //runtime block size

// forward declaration of launcher
extern "C" void solve(MoEArgs args);


std::string get_cache_filename(int N){
    char buffer[256];
    snprintf(buffer, sizeof(buffer), ".cache/ref_N%d.bin", N);
    return std::string(buffer);
}

bool ensure_cache_dir_exists(){
    const char* dir = ".cache";
    struct stat st = {};
    if (stat(dir, &st) == 0) {
#ifdef _WIN32
        return (st.st_mode & _S_IFDIR) != 0;
#else
        return S_ISDIR(st.st_mode);
#endif
    }

#ifdef _WIN32
    if (_mkdir(dir) == 0) return true;
#else
    if (mkdir(dir, 0755) == 0) return true;
#endif

    return errno == EEXIST;
}

int main(int argc, char** argv){

    std::string kernel = "baseline";

    int warmups = 0;
    int runs = 1;

    for(int i = 1; i < argc; ++i){
        if(std::strncmp(argv[i], "--kernel=", 9) == 0) kernel = std::string(argv[i] + 9);
        else if(std::strncmp(argv[i], "--warmups=", 10) == 0) warmups = std::atoi(argv[i]+10);
        else if(std::strncmp(argv[i], "--runs=", 7) == 0) runs = std::atoi(argv[i]+7);
        else if(std::strcmp(argv[i], "--help")==0){
            printf("Usage: %s [--kernel=KERNEL] [--warmups=N] [--runs=M] [--random]\n", argv[0]);
            printf("KERNEL options: baseline, ... \n");
            return 0;
        }
    }

    // Allocate and initialize host data (only true inputs and weights)
    std::vector<half> h_input((size_t)num_batches * N * d_model);
    std::vector<float> h_final_output(N * d_model);
    std::vector<half> h_expert_up_proj_weights((size_t)num_experts * d_model * up_proj_dim * d_model);
    std::vector<half> h_expert_down_proj_weights((size_t)num_experts * up_proj_dim * d_model * d_model);

    ensure_cache_dir_exists();
    initialize_host_data(h_input, h_final_output, h_expert_up_proj_weights, h_expert_down_proj_weights);

    printf("Allocating and copying data to device ... \n");
    // enable capacity-aware allocations only when running kernels other than baseline and unfused (unfused has it's own dedicated launcher)
    bool use_capacity = (kernel != "baseline");
    MoEArgs args = allocate_and_copy_to_device(h_input, h_final_output, h_expert_up_proj_weights, h_expert_down_proj_weights, use_capacity);



    printf("Running %d warmup runs... \n", warmups);
    for(int w = 0; w < warmups; ++w) solve(args);

    printf("Running %d profiling runs... \n", runs);
    CHECK_CUDA(cudaProfilerStart());
    for(int r = 0; r < runs; ++r) solve(args);
    CHECK_CUDA(cudaProfilerStop());    

    CHECK_CUDA(cudaMemcpy(h_final_output.data(), args.final_output, sizeof(float) * N * d_model, cudaMemcpyDeviceToHost));

    cleanup_device_data(args);

    printf("Profiling is complete.\n");

    return 0;
}
