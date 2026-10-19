// S02 benchmark driver: block-size / block-shape sweep with interleaved randomized order,
// per-configuration correctness check, data-movement timing, launch-overhead timing and a
// single-thread CPU baseline. Writes RAW per-repetition rows only; statistics are computed
// later by scripts/analyze.py.
//
//   bench <suite> [--sizes a,b,c] [--reps R] [--warmup W] [--cpu-reps R] [--outdir DIR] [--seed S]
//   suites: vecadd | matmul | matmul_tiled | conv3 | conv5 | overhead
#include "kernels.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <sys/stat.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
    std::exit(2); } } while (0)

// ------------------------------------------------------------------------------------------
struct Args {
    std::string suite, outdir = "results/raw";
    std::vector<long long> sizes;
    int reps = 30, warmup = 5, cpu_reps = -1;
    unsigned seed = 12345;
    double preheat_s = 2.0;
};

static std::vector<long long> parse_list(const char* s) {
    std::vector<long long> v;
    for (const char* p = s; *p;) { v.push_back(atoll(p)); while (*p && *p != ',') ++p; if (*p) ++p; }
    return v;
}

static Args parse(int argc, char** argv) {
    Args a;
    if (argc < 2) { fprintf(stderr, "usage: bench <suite> [options]\n"); std::exit(1); }
    a.suite = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string k = argv[i];
        auto nxt = [&] { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", k.c_str()); std::exit(1); } return argv[++i]; };
        if (k == "--sizes") a.sizes = parse_list(nxt());
        else if (k == "--reps") a.reps = atoi(nxt());
        else if (k == "--warmup") a.warmup = atoi(nxt());
        else if (k == "--cpu-reps") a.cpu_reps = atoi(nxt());
        else if (k == "--outdir") a.outdir = nxt();
        else if (k == "--seed") a.seed = (unsigned)atoi(nxt());
        else if (k == "--preheat") a.preheat_s = atof(nxt());
        else { fprintf(stderr, "unknown option %s\n", k.c_str()); std::exit(1); }
    }
    return a;
}

// Append-mode CSV that writes its header only when the file is new/empty.
struct Csv {
    FILE* f = nullptr;
    Csv(const std::string& path, const char* header) {
        struct stat st; bool fresh = stat(path.c_str(), &st) != 0 || st.st_size == 0;
        f = fopen(path.c_str(), "a");
        if (!f) { perror(path.c_str()); std::exit(1); }
        if (fresh) fprintf(f, "%s\n", header);
    }
    ~Csv() { if (f) fclose(f); }
};

static std::string now_iso() {
    time_t t = time(nullptr); char b[32]; strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%S", localtime(&t)); return b;
}

static cudaDeviceProp g_prop;

// ------------------------------------------------------------------------------------------
struct Cfg { int bx, by; int flag; };  // flag=1: deliberately wasteful (threads not multiple of 32)

static std::vector<Cfg> configs_1d() {
    std::vector<Cfg> c;
    for (int b : {32, 64, 96, 128, 192, 256, 384, 512, 768, 1024}) c.push_back({b, 1, 0});
    c.push_back({48, 1, 1});
    c.push_back({100, 1, 1});
    return c;
}

static std::vector<Cfg> configs_2d() {
    // grouped by thread count: square and non-square shapes with equal threads/block
    int s[][2] = {{8, 8}, {16, 4}, {4, 16}, {32, 2}, {64, 1},
                  {16, 8}, {8, 16}, {32, 4}, {128, 1},
                  {16, 16}, {32, 8}, {8, 32}, {64, 4}, {4, 64}, {128, 2}, {256, 1},
                  {32, 16}, {16, 32}, {64, 8},
                  {32, 32}, {64, 16}, {16, 64}, {128, 8}};
    std::vector<Cfg> c;
    for (auto& p : s) c.push_back({p[0], p[1], 0});
    c.push_back({12, 12, 1});  // 144 threads = 4.5 warps, rows not warp-aligned
    return c;
}

struct Occ { int regs, smem_static, blocks_per_sm; double occ; };

static Occ occupancy(const void* fn, int threads, size_t dyn_smem = 0) {
    cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, fn));
    int nb = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, fn, threads, dyn_smem));
    int warps_per_block = (threads + g_prop.warpSize - 1) / g_prop.warpSize;
    double occ = (double)nb * warps_per_block / (g_prop.maxThreadsPerMultiProcessor / g_prop.warpSize);
    return {fa.numRegs, (int)fa.sharedSizeBytes, nb, occ};
}

static float time_once(cudaEvent_t e0, cudaEvent_t e1, const std::function<void()>& launch) {
    CK(cudaEventRecord(e0));
    launch();
    CK(cudaGetLastError());
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
    return ms;
}

struct Err { double max_abs, max_rel; long long bad; };

static Err compare(const std::vector<float>& gpu, const std::vector<float>& ref, double rtol, double atol) {
    Err e{0, 0, 0};
    for (size_t i = 0; i < ref.size(); ++i) {
        double g = gpu[i], r = ref[i], d = std::fabs(g - r);
        if (!std::isfinite(g)) { e.bad++; e.max_abs = std::numeric_limits<double>::infinity(); continue; }
        e.max_abs = std::max(e.max_abs, d);
        if (std::fabs(r) > 0) e.max_rel = std::max(e.max_rel, d / std::fabs(r));
        if (d > atol + rtol * std::fabs(r)) e.bad++;
    }
    return e;
}

// Keep the GPU busy for ~seconds so it leaves the idle P-state before anything is timed.
static void preheat(double seconds) {
    if (seconds <= 0) return;
    size_t n = 1 << 24; float* d; CK(cudaMalloc(&d, 3 * n * sizeof(float)));
    CK(cudaMemset(d, 0, 3 * n * sizeof(float)));
    auto t0 = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count() < seconds) {
        for (int i = 0; i < 50; ++i) vecadd_kernel<<<(unsigned)(n / 256), 256>>>(d, d + n, d + 2 * n, n);
        CK(cudaDeviceSynchronize());
    }
    CK(cudaFree(d));
}

// Generic sweep over configurations for one (kernel, size).
struct SweepSpec {
    std::string kernel, variant;
    long long size;            // N elements (1D) or N (square side, 2D)
    long long nx, ny;          // logical thread-domain extent
    const void* fn;
    std::function<void(dim3, dim3)> launch;
    float* d_out; size_t out_elems;
    const std::vector<float>* ref;
    double rtol, atol;
};

static void sweep(const Args& a, const SweepSpec& s, const std::vector<Cfg>& cfgs, Csv& raw, Csv& cfgcsv) {
    preheat(a.preheat_s);
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    std::vector<int> valid;
    std::vector<float> host(s.out_elems);
    std::vector<dim3> grids(cfgs.size());
    for (size_t c = 0; c < cfgs.size(); ++c) {
        const Cfg& g = cfgs[c];
        int threads = g.bx * g.by;
        dim3 blk(g.bx, g.by), grd((unsigned)((s.nx + g.bx - 1) / g.bx), (unsigned)((s.ny + g.by - 1) / g.by));
        grids[c] = grd;
        Occ o = occupancy(s.fn, threads);
        long long nblocks = (long long)grd.x * grd.y;
        double waves = o.blocks_per_sm > 0 ? (double)nblocks / ((double)o.blocks_per_sm * g_prop.multiProcessorCount) : -1;
        // warm-up (discarded), then correctness on a poisoned output buffer
        float first_ms = -1;
        for (int w = 0; w < a.warmup; ++w) { float t = time_once(e0, e1, [&] { s.launch(grd, blk); }); if (w == 0) first_ms = t; }
        CK(cudaMemset(s.d_out, 0xFF, s.out_elems * sizeof(float)));  // 0xFFFFFFFF = NaN
        time_once(e0, e1, [&] { s.launch(grd, blk); });
        CK(cudaMemcpy(host.data(), s.d_out, s.out_elems * sizeof(float), cudaMemcpyDeviceToHost));
        Err e = compare(host, *s.ref, s.rtol, s.atol);
        int pass = e.bad == 0;
        if (pass) valid.push_back((int)c);
        fprintf(cfgcsv.f, "%s,%s,%lld,%d,%d,%d,%d,%u,%u,%lld,%d,%d,%d,%.6f,%.4f,%.3e,%.3e,%lld,%d,%.6f\n",
                s.kernel.c_str(), s.variant.c_str(), s.size, g.bx, g.by, threads, g.flag, grd.x, grd.y, nblocks,
                o.regs, o.smem_static, o.blocks_per_sm, o.occ, waves, e.max_abs, e.max_rel, e.bad, pass, first_ms);
        fflush(cfgcsv.f);
        if (!pass) fprintf(stderr, "  !! %s %s size %lld cfg %dx%d FAILED correctness (bad=%lld, max_abs=%.3e) -- excluded\n",
                           s.kernel.c_str(), s.variant.c_str(), s.size, g.bx, g.by, e.bad, e.max_abs);
    }
    // timed repetitions, configuration order reshuffled every repetition (thermal-drift control)
    std::mt19937 rng(a.seed + (unsigned)s.size);
    for (int r = 0; r < a.reps; ++r) {
        std::vector<int> order = valid;
        std::shuffle(order.begin(), order.end(), rng);
        for (size_t pos = 0; pos < order.size(); ++pos) {
            const Cfg& g = cfgs[order[pos]];
            dim3 blk(g.bx, g.by);
            float ms = time_once(e0, e1, [&] { s.launch(grids[order[pos]], blk); });
            fprintf(raw.f, "%s,%s,%lld,%d,%d,%d,%d,%d,%zu,%.6f,%s\n", s.kernel.c_str(), s.variant.c_str(), s.size,
                    g.bx, g.by, g.bx * g.by, g.flag, r, pos, ms, now_iso().c_str());
        }
        fflush(raw.f);
    }
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    printf("  %s/%s size=%lld: %zu/%zu configs valid, %d reps\n", s.kernel.c_str(), s.variant.c_str(), s.size,
           valid.size(), cfgs.size(), a.reps);
}

// ------------------------------------------------------------------------------------------
// Data movement: pageable and pinned H2D/D2H, timed with events.
static void time_transfers(const Args& a, const std::string& kernel, long long size, size_t in_bytes_each, int n_inputs,
                           size_t out_bytes, Csv& tr) {
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    size_t maxb = std::max(in_bytes_each, out_bytes);
    void* d; CK(cudaMalloc(&d, maxb));
    std::vector<char> pageable(maxb, 1);
    void* pinned; CK(cudaMallocHost(&pinned, maxb)); memset(pinned, 1, maxb);
    struct T { const char* dir; const char* mem; size_t bytes; int count; };
    T ts[] = {{"H2D", "pageable", in_bytes_each, n_inputs}, {"D2H", "pageable", out_bytes, 1},
              {"H2D", "pinned", in_bytes_each, n_inputs}, {"D2H", "pinned", out_bytes, 1}};
    for (auto& t : ts) {
        void* h = strcmp(t.mem, "pinned") == 0 ? pinned : (void*)pageable.data();
        for (int r = -2; r < a.reps; ++r) {   // r<0: warm-up
            float ms = time_once(e0, e1, [&] {
                for (int k = 0; k < t.count; ++k) {
                    if (t.dir[0] == 'H') CK(cudaMemcpy(d, h, t.bytes, cudaMemcpyHostToDevice));
                    else CK(cudaMemcpy(h, d, t.bytes, cudaMemcpyDeviceToHost));
                }
            });
            if (r >= 0) fprintf(tr.f, "%s,%lld,%s,%s,%zu,%d,%.6f\n", kernel.c_str(), size, t.dir, t.mem,
                                t.bytes * t.count, r, ms);
        }
    }
    fflush(tr.f);
    CK(cudaFreeHost(pinned)); CK(cudaFree(d));
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
}

template <class F>
static void time_cpu(const std::string& kernel, long long size, int reps, Csv& cpu, F&& f) {
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        f();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        fprintf(cpu.f, "%s,%lld,%d,%.4f\n", kernel.c_str(), size, r, ms);
        fflush(cpu.f);
        printf("  cpu %s size=%lld rep %d: %.2f ms\n", kernel.c_str(), size, r, ms);
    }
}

static void fill_random(std::vector<float>& v, unsigned seed) {
    std::mt19937 g(seed); std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (auto& x : v) x = u(g);
}

// ------------------------------- CPU references (single thread) --------------------------
static void cpu_vecadd(const float* a, const float* b, float* c, size_t n) { for (size_t i = 0; i < n; ++i) c[i] = a[i] + b[i]; }

static void cpu_matmul(const float* A, const float* B, float* C, int N) {  // i-k-j order (cache friendly)
    std::fill(C, C + (size_t)N * N, 0.0f);
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k) {
            float aik = A[(size_t)i * N + k];
            const float* br = B + (size_t)k * N; float* cr = C + (size_t)i * N;
            for (int j = 0; j < N; ++j) cr[j] += aik * br[j];
        }
}

static void cpu_conv(const float* in, float* out, const float* f, int W, int H, int K) {
    int R = K / 2;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float s = 0.0f;
            for (int dy = 0; dy < K; ++dy) {
                int yy = y + dy - R; if (yy < 0 || yy >= H) continue;
                for (int dx = 0; dx < K; ++dx) {
                    int xx = x + dx - R; if (xx < 0 || xx >= W) continue;
                    s += in[(size_t)yy * W + xx] * f[dy * K + dx];
                }
            }
            out[(size_t)y * W + x] = s;
        }
}

// ------------------------------------------------------------------------------------------
static const char* RAW_HDR = "kernel,variant,size,bx,by,threads,flag,rep,order_pos,time_ms,timestamp";
static const char* CFG_HDR = "kernel,variant,size,bx,by,threads,flag,grid_x,grid_y,num_blocks,regs_per_thread,"
                             "static_smem_bytes,max_active_blocks_per_sm,theoretical_occupancy,waves_per_sm,"
                             "max_abs_err,max_rel_err,n_bad,pass,first_warmup_ms";
static const char* TR_HDR = "kernel,size,direction,mem_type,bytes,rep,time_ms";
static const char* CPU_HDR = "kernel,size,rep,time_ms";

static int default_cpu_reps(double flops_or_elems) { return flops_or_elems > 5e10 ? 2 : flops_or_elems > 5e9 ? 3 : 10; }

static void suite_vecadd(const Args& a, Csv& raw, Csv& cfg, Csv& tr, Csv& cpu) {
    std::vector<long long> sizes = a.sizes.empty() ? std::vector<long long>{1LL << 20, 1LL << 24, 1LL << 27} : a.sizes;
    for (long long n : sizes) {
        std::vector<float> A(n), B(n), R(n);
        fill_random(A, 1); fill_random(B, 2);
        time_cpu("vecadd", n, a.cpu_reps > 0 ? a.cpu_reps : 10, cpu, [&] { cpu_vecadd(A.data(), B.data(), R.data(), n); });
        float *dA, *dB, *dC; size_t bytes = n * sizeof(float);
        CK(cudaMalloc(&dA, bytes)); CK(cudaMalloc(&dB, bytes)); CK(cudaMalloc(&dC, bytes));
        CK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));
        SweepSpec s{"vecadd", "naive", n, n, 1, (const void*)vecadd_kernel,
                    [&](dim3 g, dim3 b) { vecadd_kernel<<<g, b>>>(dA, dB, dC, (size_t)n); }, dC, (size_t)n, &R, 1e-6, 0.0};
        sweep(a, s, configs_1d(), raw, cfg);
        CK(cudaFree(dA)); CK(cudaFree(dB)); CK(cudaFree(dC));
        time_transfers(a, "vecadd", n, bytes, 2, bytes, tr);
    }
}

static void suite_matmul(const Args& a, bool tiled, Csv& raw, Csv& cfg, Csv& tr, Csv& cpu) {
    std::vector<long long> sizes = a.sizes.empty() ? std::vector<long long>{512, 1024, 2048, 4096} : a.sizes;
    for (long long N : sizes) {
        size_t nn = (size_t)N * N, bytes = nn * sizeof(float);
        std::vector<float> A(nn), B(nn), R(nn);
        fill_random(A, 3); fill_random(B, 4);
        double flops = 2.0 * N * N * N;
        if (tiled) cpu_matmul(A.data(), B.data(), R.data(), (int)N);  // reference only; baseline logged by 'matmul'
        else time_cpu("matmul", N, a.cpu_reps > 0 ? a.cpu_reps : default_cpu_reps(flops), cpu,
                      [&] { cpu_matmul(A.data(), B.data(), R.data(), (int)N); });
        float *dA, *dB, *dC;
        CK(cudaMalloc(&dA, bytes)); CK(cudaMalloc(&dB, bytes)); CK(cudaMalloc(&dC, bytes));
        CK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));
        int n = (int)N;
        if (!tiled) {
            SweepSpec s{"matmul", "naive", N, N, N, (const void*)matmul_naive_kernel,
                        [&](dim3 g, dim3 b) { matmul_naive_kernel<<<g, b>>>(dA, dB, dC, n); }, dC, nn, &R, 1e-4, 1e-3};
            sweep(a, s, configs_2d(), raw, cfg);
            time_transfers(a, "matmul", N, bytes, 2, bytes, tr);
        } else {
            // tiled kernel requires square blocks equal to the tile: one config per template instance
            auto run = [&](int T, const void* fn, std::function<void(dim3, dim3)> l) {
                SweepSpec s{"matmul", "tiled", N, N, N, fn, l, dC, nn, &R, 1e-4, 1e-3};
                sweep(a, s, {{T, T, 0}}, raw, cfg);
            };
            run(8, (const void*)matmul_tiled_kernel<8>, [&](dim3 g, dim3 b) { matmul_tiled_kernel<8><<<g, b>>>(dA, dB, dC, n); });
            run(16, (const void*)matmul_tiled_kernel<16>, [&](dim3 g, dim3 b) { matmul_tiled_kernel<16><<<g, b>>>(dA, dB, dC, n); });
            run(32, (const void*)matmul_tiled_kernel<32>, [&](dim3 g, dim3 b) { matmul_tiled_kernel<32><<<g, b>>>(dA, dB, dC, n); });
        }
        CK(cudaFree(dA)); CK(cudaFree(dB)); CK(cudaFree(dC));
    }
}

template <int K>
static void suite_conv(const Args& a, Csv& raw, Csv& cfg, Csv& tr, Csv& cpu) {
    std::vector<long long> sizes = a.sizes.empty() ? std::vector<long long>{1024, 2048, 4096} : a.sizes;
    // fixed, normalised, non-symmetric filter so indexing errors are detectable
    float f[25]; float sum = 0;
    for (int i = 0; i < K * K; ++i) { f[i] = 1.0f + (float)i / (K * K); sum += f[i]; }
    for (int i = 0; i < K * K; ++i) f[i] /= sum;
    CK(cudaMemcpyToSymbol(c_filter, f, K * K * sizeof(float)));
    std::string name = "conv" + std::to_string(K);
    for (long long N : sizes) {
        size_t nn = (size_t)N * N, bytes = nn * sizeof(float);
        std::vector<float> I(nn), R(nn);
        fill_random(I, 5);
        time_cpu(name, N, a.cpu_reps > 0 ? a.cpu_reps : 5, cpu, [&] { cpu_conv(I.data(), R.data(), f, (int)N, (int)N, K); });
        float *dI, *dO;
        CK(cudaMalloc(&dI, bytes)); CK(cudaMalloc(&dO, bytes));
        CK(cudaMemcpy(dI, I.data(), bytes, cudaMemcpyHostToDevice));
        int n = (int)N;
        SweepSpec s{name, "naive", N, N, N, (const void*)conv2d_kernel<K>,
                    [&](dim3 g, dim3 b) { conv2d_kernel<K><<<g, b>>>(dI, dO, n, n); }, dO, nn, &R, 1e-5, 1e-6};
        sweep(a, s, configs_2d(), raw, cfg);
        CK(cudaFree(dI)); CK(cudaFree(dO));
        time_transfers(a, name, N, bytes, 1, bytes, tr);
    }
}

// Launch / scheduling / synchronization overhead (measurable parts only).
static void suite_overhead(const Args& a) {
    Csv ov(a.outdir + "/launch_overhead.csv", "test,threads,blocks,rep,launches,time_ms_total,time_us_per_launch");
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const int L = 1000;
    std::vector<int> bs = {32, 64, 96, 128, 192, 256, 384, 512, 768, 1024};
    preheat(a.preheat_s);
    for (int w = 0; w < 100; ++w) empty_kernel<<<1, 32>>>();
    CK(cudaDeviceSynchronize());
    for (int r = 0; r < a.reps; ++r) {
        for (int b : bs) {
            // (1) GPU-side time of L back-to-back single-block empty launches
            float ms = time_once(e0, e1, [&] { for (int i = 0; i < L; ++i) empty_kernel<<<1, b>>>(); });
            fprintf(ov.f, "empty_1block_device,%d,1,%d,%d,%.6f,%.4f\n", b, r, L, ms, ms * 1e3 / L);
            // (2) empty grid covering 2^24 threads: cost of scheduling ceil(2^24/b) empty blocks
            unsigned nb = (unsigned)((1u << 24) / b);
            ms = time_once(e0, e1, [&] { empty_kernel<<<nb, b>>>(); });
            fprintf(ov.f, "empty_grid_2^24_threads,%d,%u,%d,1,%.6f,%.4f\n", b, nb, r, ms, ms * 1e3);
        }
        // (3) host-side cost of the launch API call (no sync inside the timed region)
        CK(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < L; ++i) empty_kernel<<<1, 256>>>();
        auto t1 = std::chrono::high_resolution_clock::now();
        CK(cudaDeviceSynchronize());
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        fprintf(ov.f, "host_launch_call,256,1,%d,%d,%.6f,%.4f\n", r, L, ms, ms * 1e3 / L);
        // (4) launch + cudaDeviceSynchronize round trip, one at a time
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 200; ++i) { empty_kernel<<<1, 256>>>(); CK(cudaDeviceSynchronize()); }
        t1 = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        fprintf(ov.f, "launch_plus_sync_roundtrip,256,1,%d,200,%.6f,%.4f\n", r, ms, ms * 1e3 / 200);
        // (5) cudaDeviceSynchronize on an idle device
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 1000; ++i) CK(cudaDeviceSynchronize());
        t1 = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        fprintf(ov.f, "idle_device_sync,0,0,%d,1000,%.6f,%.4f\n", r, ms, ms * 1e3 / 1000);
    }
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    printf("  overhead: %d reps done\n", a.reps);
}

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    CK(cudaSetDevice(0));
    CK(cudaGetDeviceProperties(&g_prop, 0));
    CK(cudaFree(0));  // create context outside timed regions
    printf("[%s] suite=%s reps=%d warmup=%d on %s\n", now_iso().c_str(), a.suite.c_str(), a.reps, a.warmup, g_prop.name);
    if (a.suite == "overhead") { suite_overhead(a); return 0; }
    Csv raw(a.outdir + "/kernel_times.csv", RAW_HDR);
    Csv cfg(a.outdir + "/configs.csv", CFG_HDR);
    Csv tr(a.outdir + "/transfers.csv", TR_HDR);
    Csv cpu(a.outdir + "/cpu_baseline.csv", CPU_HDR);
    if (a.suite == "vecadd") suite_vecadd(a, raw, cfg, tr, cpu);
    else if (a.suite == "matmul") suite_matmul(a, false, raw, cfg, tr, cpu);
    else if (a.suite == "matmul_tiled") suite_matmul(a, true, raw, cfg, tr, cpu);
    else if (a.suite == "conv3") suite_conv<3>(a, raw, cfg, tr, cpu);
    else if (a.suite == "conv5") suite_conv<5>(a, raw, cfg, tr, cpu);
    else { fprintf(stderr, "unknown suite %s\n", a.suite.c_str()); return 1; }
    printf("[%s] done\n", now_iso().c_str());
    return 0;
}
