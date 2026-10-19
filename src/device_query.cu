// Queries the real GPU via the CUDA runtime and writes results/raw/device_info.json.
// Derived peaks are computed from queried values; the only non-queried constant is
// FP32 lanes per SM, which the runtime does not expose (64 for compute capability 7.5,
// per the CUDA C++ Programming Guide "Arithmetic Instructions" throughput table).
#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)

static int attr(cudaDeviceAttr a, int dev) { int v = 0; cudaDeviceGetAttribute(&v, a, dev); return v; }

int main(int argc, char** argv) {
    const char* out = argc > 1 ? argv[1] : "results/raw/device_info.json";
    int dev = 0, rt = 0, drv = 0;
    CK(cudaSetDevice(dev));
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, dev));
    CK(cudaRuntimeGetVersion(&rt));
    CK(cudaDriverGetVersion(&drv));

    int smClockKHz  = attr(cudaDevAttrClockRate, dev);
    int memClockKHz = attr(cudaDevAttrMemoryClockRate, dev);
    int busWidth    = attr(cudaDevAttrGlobalMemoryBusWidth, dev);
    int fp32PerSM   = (p.major == 7 && p.minor == 5) ? 64 : -1;
    // GDDR6 reported memory clock is the DDR base; x2 for double data rate.
    double peakBW    = 2.0 * memClockKHz * 1e3 * (busWidth / 8.0) / 1e9;       // GB/s
    double peakFP32  = 2.0 * fp32PerSM * p.multiProcessorCount * smClockKHz * 1e3 / 1e9; // GFLOP/s (FMA=2)

    FILE* f = fopen(out, "w");
    if (!f) { perror(out); return 1; }
    fprintf(f, "{\n");
    fprintf(f, "  \"name\": \"%s\",\n", p.name);
    fprintf(f, "  \"compute_capability\": \"%d.%d\",\n", p.major, p.minor);
    fprintf(f, "  \"cuda_runtime_version\": %d,\n  \"cuda_driver_version\": %d,\n", rt, drv);
    fprintf(f, "  \"sm_count\": %d,\n", p.multiProcessorCount);
    fprintf(f, "  \"warp_size\": %d,\n", p.warpSize);
    fprintf(f, "  \"max_threads_per_block\": %d,\n", p.maxThreadsPerBlock);
    fprintf(f, "  \"max_block_dim\": [%d, %d, %d],\n", p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
    fprintf(f, "  \"max_threads_per_sm\": %d,\n", p.maxThreadsPerMultiProcessor);
    fprintf(f, "  \"max_blocks_per_sm\": %d,\n", attr(cudaDevAttrMaxBlocksPerMultiprocessor, dev));
    fprintf(f, "  \"max_warps_per_sm\": %d,\n", p.maxThreadsPerMultiProcessor / p.warpSize);
    fprintf(f, "  \"regs_per_sm\": %d,\n  \"regs_per_block\": %d,\n", p.regsPerMultiprocessor, p.regsPerBlock);
    fprintf(f, "  \"shared_mem_per_sm_bytes\": %zu,\n", p.sharedMemPerMultiprocessor);
    fprintf(f, "  \"shared_mem_per_block_bytes\": %zu,\n", p.sharedMemPerBlock);
    fprintf(f, "  \"shared_mem_per_block_optin_bytes\": %zu,\n", p.sharedMemPerBlockOptin);
    fprintf(f, "  \"l2_cache_bytes\": %d,\n", p.l2CacheSize);
    fprintf(f, "  \"global_mem_bytes\": %zu,\n", p.totalGlobalMem);
    fprintf(f, "  \"sm_clock_khz\": %d,\n  \"mem_clock_khz\": %d,\n", smClockKHz, memClockKHz);
    fprintf(f, "  \"mem_bus_width_bits\": %d,\n", busWidth);
    fprintf(f, "  \"tcc_driver\": %d,\n", p.tccDriver);
    fprintf(f, "  \"concurrent_kernels\": %d,\n  \"async_engine_count\": %d,\n", p.concurrentKernels, p.asyncEngineCount);
    fprintf(f, "  \"fp32_lanes_per_sm\": %d,\n", fp32PerSM);
    fprintf(f, "  \"fp32_lanes_per_sm_source\": \"CUDA C++ Programming Guide throughput table (not queryable)\",\n");
    fprintf(f, "  \"peak_dram_bw_gbs\": %.2f,\n", peakBW);
    fprintf(f, "  \"peak_dram_bw_formula\": \"2 * mem_clock_khz*1e3 * bus_width_bits/8 / 1e9\",\n");
    fprintf(f, "  \"peak_fp32_gflops\": %.2f,\n", peakFP32);
    fprintf(f, "  \"peak_fp32_formula\": \"2 * fp32_lanes_per_sm * sm_count * sm_clock_khz*1e3 / 1e9 (at reported max clock)\"\n");
    fprintf(f, "}\n");
    fclose(f);
    printf("wrote %s: %s cc%d.%d, %d SMs, %.1f GB/s, %.1f GFLOP/s\n", out, p.name, p.major, p.minor,
           p.multiProcessorCount, peakBW, peakFP32);
    return 0;
}
