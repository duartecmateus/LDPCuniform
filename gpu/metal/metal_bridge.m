#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "metal_bridge.h"

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLLibrary> g_library = nil;

static id<MTLComputePipelineState> g_pipeline_col_intersections = nil;
static id<MTLComputePipelineState> g_pipeline_rect_eliminate = nil; /* legacy fallback */
static id<MTLComputePipelineState> g_pipeline_gpu_init_pivot = nil;
static id<MTLComputePipelineState> g_pipeline_gpu_find_pivot = nil;
static id<MTLComputePipelineState> g_pipeline_gpu_swap_rows = nil;
static id<MTLComputePipelineState> g_pipeline_gpu_compute_pivot_inv = nil;
static id<MTLComputePipelineState> g_pipeline_gpu_normalize_pivot = nil;
static id<MTLComputePipelineState> g_pipeline_gpu_eliminate_column = nil;
static id<MTLComputePipelineState> g_pipeline_gpu_advance_rank = nil;

/* Binary-specific pipelines (no table lookups) */
static id<MTLComputePipelineState> g_pipeline_bin_find_pivot = nil;
static id<MTLComputePipelineState> g_pipeline_bin_swap_rows = nil;
static id<MTLComputePipelineState> g_pipeline_bin_eliminate_column = nil;
static id<MTLComputePipelineState> g_pipeline_bin_advance_rank = nil;

static int g_initialized = 0;

static uint8_t gflog_tbl[256];
static uint8_t gfexp_tbl[256];
static uint8_t gfmul_tbl[256 * 256];
static uint8_t gfinv_tbl[256];
static int gf_tables_inited = 0;

#ifndef ALG2_GPU_PIVOT_BLOCK_SIZE
#define ALG2_GPU_PIVOT_BLOCK_SIZE 8
#endif

static int setup_tables_cpu(int w)
{
    unsigned int b;
    unsigned int log;
    unsigned int x_to_w;
    unsigned int prim_poly;

    unsigned int prim_poly_4  = 023;     /* GF(16): x^4 + x + 1 */
    unsigned int prim_poly_8  = 0435;    /* GF(256): x^8 + x^4 + x^3 + x^2 + 1 */
    unsigned int prim_poly_16 = 0210013; /* GF(2^16) */
    unsigned int prim_poly_2  = 07;      /* GF(4): x^2 + x + 1 */

    switch (w) {
        case 2:  prim_poly = prim_poly_2;  break;
        case 4:  prim_poly = prim_poly_4;  break;
        case 8:  prim_poly = prim_poly_8;  break;
        case 16: prim_poly = prim_poly_16; break;
        default: return -1;
    }

    x_to_w = 1u << w;
    memset(gflog_tbl, 0, sizeof(gflog_tbl));
    memset(gfexp_tbl, 0, sizeof(gfexp_tbl));
    memset(gfmul_tbl, 0, sizeof(gfmul_tbl));
    memset(gfinv_tbl, 0, sizeof(gfinv_tbl));

    b = 1;
    for (log = 0; log < x_to_w - 1; ++log) {
        gflog_tbl[b] = (uint8_t)log;
        gfexp_tbl[log] = (uint8_t)b;
        b <<= 1;
        if (b & x_to_w) b ^= prim_poly;
    }
    gfexp_tbl[255] = gfexp_tbl[0];

    for (unsigned int a = 0; a < 256u; ++a) {
        for (unsigned int c = 0; c < 256u; ++c) {
            uint8_t value = 0;
            if (a != 0 && c != 0) {
                unsigned int la = gflog_tbl[a];
                unsigned int lb = gflog_tbl[c];
                unsigned int s = la + lb;
                if (s >= 255u) s -= 255u;
                value = gfexp_tbl[s];
            }
            gfmul_tbl[(a << 8) | c] = value;
        }
    }

    gfinv_tbl[0] = 0;
    for (unsigned int a = 1; a < 256u; ++a) {
        unsigned int la = gflog_tbl[a];
        unsigned int idx = 255u - la;
        if (idx >= 255u) idx -= 255u;
        gfinv_tbl[a] = gfexp_tbl[idx];
    }

    return 0;
}

static void ensure_gf_tables(void)
{
    if (!gf_tables_inited) {
        if (setup_tables_cpu(8) != 0) {
            fprintf(stderr, "[Metal] GF(256) table setup failed\n");
            abort();
        }
        gf_tables_inited = 1;
    }
}

static int gf16_tables_inited = 0;
static void ensure_gf16_tables(void)
{
    if (!gf16_tables_inited) {
        /* Build GF(16) multiplication and inverse tables properly.
           GF(2^4) with prim poly x^4 + x + 1 (0x13). */
        memset(gfmul_tbl, 0, sizeof(gfmul_tbl));
        memset(gfinv_tbl, 0, sizeof(gfinv_tbl));
        const unsigned int x_to_w = 16u;
        const unsigned int prim_poly = 0x13u;
        uint8_t log[16], exp[15];
        memset(log, 0, sizeof(log));
        memset(exp, 0, sizeof(exp));
        unsigned int b = 1;
        for (unsigned int i = 0; i < 15u; ++i) {
            log[b] = (uint8_t)i;
            exp[i] = (uint8_t)b;
            b <<= 1;
            if (b & x_to_w) b ^= prim_poly;
        }
        for (unsigned int a = 0; a < 16u; ++a) {
            for (unsigned int c = 0; c < 16u; ++c) {
                uint8_t v = 0;
                if (a != 0 && c != 0) {
                    unsigned int s = (unsigned int)log[a] + (unsigned int)log[c];
                    if (s >= 15u) s -= 15u;
                    v = exp[s];
                }
                gfmul_tbl[(a << 8) | c] = v;
            }
        }
        for (unsigned int a = 1; a < 16u; ++a) {
            unsigned int idx = 15u - (unsigned int)log[a];
            if (idx >= 15u) idx -= 15u;
            gfinv_tbl[a] = exp[idx];
        }
        gf16_tables_inited = 1;
    }
}

static int gf4_tables_inited = 0;
static void ensure_gf4_tables(void)
{
    if (!gf4_tables_inited) {
        /* GF(4) with prim poly x^2 + x + 1 (0x7).
           Elements: 0, 1, α, α+1.  Multiplication table: 4x4. */
        memset(gfmul_tbl, 0, sizeof(gfmul_tbl));
        memset(gfinv_tbl, 0, sizeof(gfinv_tbl));
        const unsigned int x_to_w = 4u;
        const unsigned int prim_poly = 0x7u;
        uint8_t log[4], exp[3];
        memset(log, 0, sizeof(log));
        memset(exp, 0, sizeof(exp));
        unsigned int b = 1;
        for (unsigned int i = 0; i < 3u; ++i) {
            log[b] = (uint8_t)i;
            exp[i] = (uint8_t)b;
            b <<= 1;
            if (b & x_to_w) b ^= prim_poly;
        }
        for (unsigned int a = 0; a < 4u; ++a) {
            for (unsigned int c = 0; c < 4u; ++c) {
                uint8_t v = 0;
                if (a != 0 && c != 0) {
                    unsigned int s = (unsigned int)log[a] + (unsigned int)log[c];
                    if (s >= 3u) s -= 3u;
                    v = exp[s];
                }
                gfmul_tbl[(a << 8) | c] = v;
            }
        }
        for (unsigned int a = 1; a < 4u; ++a) {
            unsigned int idx = 3u - (unsigned int)log[a];
            if (idx >= 3u) idx -= 3u;
            gfinv_tbl[a] = exp[idx];
        }
        gf4_tables_inited = 1;
    }
}

static int gf2_tables_inited = 0;
static void ensure_gf2_tables(void)
{
    if (!gf2_tables_inited) {
        /* GF(2) binary: mul is AND, inv is identity */
        memset(gfmul_tbl, 0, sizeof(gfmul_tbl));
        gfmul_tbl[(1u << 8) | 1u] = 1; /* 1 * 1 = 1 */
        memset(gfinv_tbl, 0, sizeof(gfinv_tbl));
        gfinv_tbl[1] = 1; /* inv(1) = 1 */
        gf2_tables_inited = 1;
    }
}

static NSString *find_metal_library(void)
{
    NSString *executablePath = [[NSBundle mainBundle] executablePath];
    NSString *executableDir = [executablePath stringByDeletingLastPathComponent];

    NSArray *searchPaths = @[
        [executableDir stringByAppendingPathComponent:@"kernel.metallib"],
        [executableDir stringByAppendingPathComponent:@"../kernel.metallib"],
        [executableDir stringByAppendingPathComponent:@"build/kernel.metallib"],
        @"kernel.metallib",
        @"build/kernel.metallib",
        @"./kernel.metallib"
    ];

    for (NSString *path in searchPaths) {
        if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
            return path;
        }
    }
    return nil;
}

static id<MTLComputePipelineState> make_pipeline(NSString *name)
{
    NSError *error = nil;
    id<MTLFunction> fn = [g_library newFunctionWithName:name];
    if (!fn) {
        fprintf(stderr, "[Metal] Kernel not found: %s\n", [name UTF8String]);
        return nil;
    }
    id<MTLComputePipelineState> ps = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!ps) {
        fprintf(stderr, "[Metal] Failed to create pipeline for %s\n", [name UTF8String]);
        if (error) fprintf(stderr, "[Metal] Error: %s\n", [[error description] UTF8String]);
    }
    return ps;
}

int metal_bridge_init(void)
{
    if (g_initialized) return 1;

    @autoreleasepool {
        ensure_gf_tables();

        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            fprintf(stderr, "[Metal] Failed to create Metal device\n");
            return 0;
        }

        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            fprintf(stderr, "[Metal] Failed to create command queue\n");
            return 0;
        }

        NSString *libPath = find_metal_library();
        if (!libPath) {
            fprintf(stderr, "[Metal] Could not find kernel.metallib\n");
            return 0;
        }

        NSError *error = nil;
        g_library = [g_device newLibraryWithURL:[NSURL fileURLWithPath:libPath] error:&error];
        if (!g_library) {
            fprintf(stderr, "[Metal] Failed to load Metal library from %s\n", [libPath UTF8String]);
            if (error) fprintf(stderr, "[Metal] Error: %s\n", [[error description] UTF8String]);
            return 0;
        }

        g_pipeline_col_intersections = make_pipeline(@"alg2_column_intersections");
        g_pipeline_rect_eliminate = make_pipeline(@"alg2_rect_eliminate_column");
        g_pipeline_gpu_init_pivot = make_pipeline(@"alg2_gpu_init_pivot");
        g_pipeline_gpu_find_pivot = make_pipeline(@"alg2_gpu_find_pivot");
        g_pipeline_gpu_swap_rows = make_pipeline(@"alg2_gpu_swap_rows");
        g_pipeline_gpu_compute_pivot_inv = make_pipeline(@"alg2_gpu_compute_pivot_inv");
        g_pipeline_gpu_normalize_pivot = make_pipeline(@"alg2_gpu_normalize_pivot");
        g_pipeline_gpu_eliminate_column = make_pipeline(@"alg2_gpu_eliminate_column");
        g_pipeline_gpu_advance_rank = make_pipeline(@"alg2_gpu_advance_rank");

        g_pipeline_bin_find_pivot = make_pipeline(@"bin_gpu_find_pivot");
        g_pipeline_bin_swap_rows = make_pipeline(@"bin_gpu_swap_rows");
        g_pipeline_bin_eliminate_column = make_pipeline(@"bin_gpu_eliminate_column");
        g_pipeline_bin_advance_rank = make_pipeline(@"bin_gpu_advance_rank");

        if (!g_pipeline_col_intersections || !g_pipeline_rect_eliminate ||
            !g_pipeline_gpu_init_pivot || !g_pipeline_gpu_find_pivot ||
            !g_pipeline_gpu_swap_rows || !g_pipeline_gpu_compute_pivot_inv ||
            !g_pipeline_gpu_normalize_pivot || !g_pipeline_gpu_eliminate_column ||
            !g_pipeline_gpu_advance_rank) {
            fprintf(stderr, "[Metal] One or more required alg2 kernels are missing\n");
            return 0;
        }

        g_initialized = 1;
        return 1;
    }
}

void metal_bridge_cleanup(void)
{
    g_pipeline_col_intersections = nil;
    g_pipeline_rect_eliminate = nil;
    g_pipeline_gpu_init_pivot = nil;
    g_pipeline_gpu_find_pivot = nil;
    g_pipeline_gpu_swap_rows = nil;
    g_pipeline_gpu_compute_pivot_inv = nil;
    g_pipeline_gpu_normalize_pivot = nil;
    g_pipeline_gpu_eliminate_column = nil;
    g_pipeline_gpu_advance_rank = nil;
    g_library = nil;
    g_queue = nil;
    g_device = nil;
    g_initialized = 0;
}

int alg2_metal_compute_column_intersections(const uint8_t *matrix,
                                            const uint8_t *syndrome,
                                            int rows,
                                            int cols,
                                            int *intersection_count,
                                            int *nonint_count)
{
    if (!matrix || !syndrome || !intersection_count || !nonint_count || rows <= 0 || cols <= 0) return 0;
    if (!metal_bridge_init() || !g_pipeline_col_intersections) return 0;

    @autoreleasepool {
        size_t matrix_size = (size_t)rows * (size_t)cols * sizeof(uint8_t);
        size_t syndrome_size = (size_t)rows * sizeof(uint8_t);
        size_t counts_size = (size_t)cols * sizeof(int);

        id<MTLBuffer> bufMatrix = [g_device newBufferWithBytes:matrix length:matrix_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufSyndrome = [g_device newBufferWithBytes:syndrome length:syndrome_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufCounts = [g_device newBufferWithLength:counts_size options:MTLResourceStorageModeShared];
        if (!bufMatrix || !bufSyndrome || !bufCounts) return 0;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!cmd || !enc) return 0;

        [enc setComputePipelineState:g_pipeline_col_intersections];
        [enc setBuffer:bufMatrix offset:0 atIndex:0];
        [enc setBuffer:bufSyndrome offset:0 atIndex:1];
        [enc setBuffer:bufCounts offset:0 atIndex:2];
        [enc setBytes:&rows length:sizeof(int) atIndex:3];
        [enc setBytes:&cols length:sizeof(int) atIndex:4];

        NSUInteger tg = g_pipeline_col_intersections.maxTotalThreadsPerThreadgroup;
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)cols, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.status == MTLCommandBufferStatusError) {
            fprintf(stderr, "[Metal] Column intersection command failed: %s\n",
                    [[[cmd error] description] UTF8String]);
            return 0;
        }

        memcpy(intersection_count, [bufCounts contents], counts_size);
        int ni = 0;
        for (int c = 0; c < cols; ++c) {
            if (intersection_count[c] == 0) ++ni;
        }
        *nonint_count = ni;
        return 1;
    }
}

static void encode_1d(id<MTLCommandBuffer> cmd,
                      id<MTLComputePipelineState> pipeline,
                      NSUInteger threads,
                      void (^bind)(id<MTLComputeCommandEncoder> enc))
{
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    bind(enc);
    NSUInteger tg = pipeline.maxTotalThreadsPerThreadgroup;
    if (tg > 256) tg = 256;
    if (threads < tg) tg = threads ? threads : 1;
    [enc dispatchThreads:MTLSizeMake(threads ? threads : 1, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
}

int alg2_metal_solve_overdetermined_unique(uint8_t *M,
                                           uint8_t *rhs,
                                           int total_rows,
                                           int cols,
                                           uint8_t *x)
{
    if (!M || !rhs || !x || total_rows < cols || cols <= 0) return 0;
    if (!metal_bridge_init()) return 0;

    @autoreleasepool {
        ensure_gf_tables();

        const uint32_t total_rows_u = (uint32_t)total_rows;
        const uint32_t cols_u = (uint32_t)cols;
        const uint32_t invalid_rank = 0xffffffffu;

        size_t matrix_size = (size_t)total_rows * (size_t)cols * sizeof(uint8_t);
        size_t rhs_size = (size_t)total_rows * sizeof(uint8_t);
        size_t x_size = (size_t)cols * sizeof(uint8_t);
        size_t mul_table_size = 256u * 256u * sizeof(uint8_t);
        size_t inv_table_size = 256u * sizeof(uint8_t);
        size_t pivot_map_size = (size_t)cols * sizeof(uint32_t);

        id<MTLBuffer> bufM = [g_device newBufferWithLength:matrix_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufRhs = [g_device newBufferWithLength:rhs_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufMul = [g_device newBufferWithBytes:gfmul_tbl length:mul_table_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufInv = [g_device newBufferWithBytes:gfinv_tbl length:inv_table_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufRank = [g_device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufPivotRow = [g_device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufPivotInv = [g_device newBufferWithLength:sizeof(uint8_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufPivotRankForCol = [g_device newBufferWithLength:pivot_map_size options:MTLResourceStorageModeShared];

        if (!bufM || !bufRhs || !bufMul || !bufInv || !bufRank || !bufPivotRow || !bufPivotInv || !bufPivotRankForCol) {
            return 0;
        }

        memcpy([bufM contents], M, matrix_size);
        memcpy([bufRhs contents], rhs, rhs_size);
        *((uint32_t *)[bufRank contents]) = 0u;
        *((uint32_t *)[bufPivotRow contents]) = total_rows_u;
        *((uint8_t *)[bufPivotInv contents]) = 0u;
        uint32_t *pivot_rank_for_col = (uint32_t *)[bufPivotRankForCol contents];
        for (int c = 0; c < cols; ++c) pivot_rank_for_col[c] = invalid_rank;

        const int block_size = ALG2_GPU_PIVOT_BLOCK_SIZE;
        for (int block_start = 0; block_start < cols; block_start += block_size) {
            int block_end = block_start + block_size;
            if (block_end > cols) block_end = cols;

            id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
            if (!cmd) return 0;

            for (int col = block_start; col < block_end; ++col) {
                uint32_t col_u = (uint32_t)col;

                encode_1d(cmd, g_pipeline_gpu_init_pivot, 1, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufPivotRow offset:0 atIndex:0];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:1];
                });

                encode_1d(cmd, g_pipeline_gpu_find_pivot, (NSUInteger)total_rows, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufM offset:0 atIndex:0];
                    [enc setBuffer:bufPivotRow offset:0 atIndex:1];
                    [enc setBuffer:bufRank offset:0 atIndex:2];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:3];
                    [enc setBytes:&cols_u length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&col_u length:sizeof(uint32_t) atIndex:5];
                });

                encode_1d(cmd, g_pipeline_gpu_swap_rows, (NSUInteger)cols, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufM offset:0 atIndex:0];
                    [enc setBuffer:bufRhs offset:0 atIndex:1];
                    [enc setBuffer:bufPivotRow offset:0 atIndex:2];
                    [enc setBuffer:bufRank offset:0 atIndex:3];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&cols_u length:sizeof(uint32_t) atIndex:5];
                });

                encode_1d(cmd, g_pipeline_gpu_compute_pivot_inv, 1, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufM offset:0 atIndex:0];
                    [enc setBuffer:bufInv offset:0 atIndex:1];
                    [enc setBuffer:bufPivotRow offset:0 atIndex:2];
                    [enc setBuffer:bufRank offset:0 atIndex:3];
                    [enc setBuffer:bufPivotInv offset:0 atIndex:4];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:5];
                    [enc setBytes:&cols_u length:sizeof(uint32_t) atIndex:6];
                    [enc setBytes:&col_u length:sizeof(uint32_t) atIndex:7];
                });

                encode_1d(cmd, g_pipeline_gpu_normalize_pivot, (NSUInteger)cols, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufM offset:0 atIndex:0];
                    [enc setBuffer:bufRhs offset:0 atIndex:1];
                    [enc setBuffer:bufMul offset:0 atIndex:2];
                    [enc setBuffer:bufPivotRow offset:0 atIndex:3];
                    [enc setBuffer:bufRank offset:0 atIndex:4];
                    [enc setBuffer:bufPivotInv offset:0 atIndex:5];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:6];
                    [enc setBytes:&cols_u length:sizeof(uint32_t) atIndex:7];
                    [enc setBytes:&col_u length:sizeof(uint32_t) atIndex:8];
                });

                encode_1d(cmd, g_pipeline_gpu_eliminate_column, (NSUInteger)total_rows, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufM offset:0 atIndex:0];
                    [enc setBuffer:bufRhs offset:0 atIndex:1];
                    [enc setBuffer:bufMul offset:0 atIndex:2];
                    [enc setBuffer:bufPivotRow offset:0 atIndex:3];
                    [enc setBuffer:bufRank offset:0 atIndex:4];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:5];
                    [enc setBytes:&cols_u length:sizeof(uint32_t) atIndex:6];
                    [enc setBytes:&col_u length:sizeof(uint32_t) atIndex:7];
                });

                encode_1d(cmd, g_pipeline_gpu_advance_rank, 1, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufPivotRow offset:0 atIndex:0];
                    [enc setBuffer:bufRank offset:0 atIndex:1];
                    [enc setBuffer:bufPivotRankForCol offset:0 atIndex:2];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:3];
                    [enc setBytes:&col_u length:sizeof(uint32_t) atIndex:4];
                });
            }

            [cmd commit];
            [cmd waitUntilCompleted];
            if (cmd.status == MTLCommandBufferStatusError) {
                fprintf(stderr, "[Metal] GPU pivot block failed: %s\n",
                        [[[cmd error] description] UTF8String]);
                return 0;
            }
        }

        uint32_t rank = *((uint32_t *)[bufRank contents]);
        uint8_t *dM = (uint8_t *)[bufM contents];
        uint8_t *dRhs = (uint8_t *)[bufRhs contents];

        for (int r = 0; r < total_rows; ++r) {
            int nonzero = 0;
            for (int c = 0; c < cols; ++c) {
                if (dM[(size_t)r * (size_t)cols + (size_t)c] != 0) {
                    nonzero = 1;
                    break;
                }
            }
            if (!nonzero && dRhs[r] != 0) {
                return 0;
            }
        }

        if (rank < (uint32_t)cols) {
            return 0;
        }

        memset(x, 0, x_size);
        for (int c = 0; c < cols; ++c) {
            uint32_t pr = pivot_rank_for_col[c];
            if (pr == invalid_rank || pr >= (uint32_t)total_rows) {
                return 0;
            }
            x[c] = dRhs[pr];
        }

        memcpy(M, dM, matrix_size);
        memcpy(rhs, dRhs, rhs_size);
        return 1;
    }
}

int alg2_metal_solve_overdetermined_unique_gf16(uint8_t *M,
                                                 uint8_t *rhs,
                                                 int total_rows,
                                                 int cols,
                                                 uint8_t *x)
{
    if (!M || !rhs || !x || total_rows < cols || cols <= 0) return 0;
    if (!metal_bridge_init()) return 0;

    @autoreleasepool {
        ensure_gf16_tables();

        const uint32_t total_rows_u = (uint32_t)total_rows;
        const uint32_t cols_u = (uint32_t)cols;
        const uint32_t invalid_rank = 0xffffffffu;

        size_t matrix_size = (size_t)total_rows * (size_t)cols * sizeof(uint8_t);
        size_t rhs_size = (size_t)total_rows * sizeof(uint8_t);
        size_t x_size = (size_t)cols * sizeof(uint8_t);
        size_t mul_table_size = 256u * 256u * sizeof(uint8_t);
        size_t inv_table_size = 256u * sizeof(uint8_t);
        size_t pivot_map_size = (size_t)cols * sizeof(uint32_t);

        id<MTLBuffer> bufM = [g_device newBufferWithLength:matrix_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufRhs = [g_device newBufferWithLength:rhs_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufMul = [g_device newBufferWithBytes:gfmul_tbl length:mul_table_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufInv = [g_device newBufferWithBytes:gfinv_tbl length:inv_table_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufRank = [g_device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufPivotRow = [g_device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufPivotInv = [g_device newBufferWithLength:sizeof(uint8_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufPivotRankForCol = [g_device newBufferWithLength:pivot_map_size options:MTLResourceStorageModeShared];

        if (!bufM || !bufRhs || !bufMul || !bufInv || !bufRank || !bufPivotRow || !bufPivotInv || !bufPivotRankForCol) {
            return 0;
        }

        memcpy([bufM contents], M, matrix_size);
        memcpy([bufRhs contents], rhs, rhs_size);
        *((uint32_t *)[bufRank contents]) = 0u;
        *((uint32_t *)[bufPivotRow contents]) = total_rows_u;
        *((uint8_t *)[bufPivotInv contents]) = 0u;
        uint32_t *pivot_rank_for_col = (uint32_t *)[bufPivotRankForCol contents];
        for (int c = 0; c < cols; ++c) pivot_rank_for_col[c] = invalid_rank;

        const int block_size = ALG2_GPU_PIVOT_BLOCK_SIZE;
        for (int block_start = 0; block_start < cols; block_start += block_size) {
            int block_end = block_start + block_size;
            if (block_end > cols) block_end = cols;

            id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
            if (!cmd) return 0;

            for (int col = block_start; col < block_end; ++col) {
                uint32_t col_u = (uint32_t)col;
                encode_1d(cmd, g_pipeline_gpu_init_pivot, 1, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufPivotRow offset:0 atIndex:0];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:1];
                });
                encode_1d(cmd, g_pipeline_gpu_find_pivot, (NSUInteger)total_rows, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufM offset:0 atIndex:0];
                    [enc setBuffer:bufPivotRow offset:0 atIndex:1];
                    [enc setBuffer:bufRank offset:0 atIndex:2];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:3];
                    [enc setBytes:&cols_u length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&col_u length:sizeof(uint32_t) atIndex:5];
                });
                encode_1d(cmd, g_pipeline_gpu_swap_rows, (NSUInteger)cols, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufM offset:0 atIndex:0];
                    [enc setBuffer:bufRhs offset:0 atIndex:1];
                    [enc setBuffer:bufPivotRow offset:0 atIndex:2];
                    [enc setBuffer:bufRank offset:0 atIndex:3];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&cols_u length:sizeof(uint32_t) atIndex:5];
                });
                encode_1d(cmd, g_pipeline_gpu_compute_pivot_inv, 1, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufM offset:0 atIndex:0];
                    [enc setBuffer:bufInv offset:0 atIndex:1];
                    [enc setBuffer:bufPivotRow offset:0 atIndex:2];
                    [enc setBuffer:bufRank offset:0 atIndex:3];
                    [enc setBuffer:bufPivotInv offset:0 atIndex:4];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:5];
                    [enc setBytes:&cols_u length:sizeof(uint32_t) atIndex:6];
                    [enc setBytes:&col_u length:sizeof(uint32_t) atIndex:7];
                });
                encode_1d(cmd, g_pipeline_gpu_normalize_pivot, (NSUInteger)cols, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufM offset:0 atIndex:0];
                    [enc setBuffer:bufRhs offset:0 atIndex:1];
                    [enc setBuffer:bufMul offset:0 atIndex:2];
                    [enc setBuffer:bufPivotRow offset:0 atIndex:3];
                    [enc setBuffer:bufRank offset:0 atIndex:4];
                    [enc setBuffer:bufPivotInv offset:0 atIndex:5];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:6];
                    [enc setBytes:&cols_u length:sizeof(uint32_t) atIndex:7];
                    [enc setBytes:&col_u length:sizeof(uint32_t) atIndex:8];
                });
                encode_1d(cmd, g_pipeline_gpu_eliminate_column, (NSUInteger)total_rows, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufM offset:0 atIndex:0];
                    [enc setBuffer:bufRhs offset:0 atIndex:1];
                    [enc setBuffer:bufMul offset:0 atIndex:2];
                    [enc setBuffer:bufPivotRow offset:0 atIndex:3];
                    [enc setBuffer:bufRank offset:0 atIndex:4];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:5];
                    [enc setBytes:&cols_u length:sizeof(uint32_t) atIndex:6];
                    [enc setBytes:&col_u length:sizeof(uint32_t) atIndex:7];
                });
                encode_1d(cmd, g_pipeline_gpu_advance_rank, 1, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufPivotRow offset:0 atIndex:0];
                    [enc setBuffer:bufRank offset:0 atIndex:1];
                    [enc setBuffer:bufPivotRankForCol offset:0 atIndex:2];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:3];
                    [enc setBytes:&col_u length:sizeof(uint32_t) atIndex:4];
                });
            }

            [cmd commit];
            [cmd waitUntilCompleted];
            if (cmd.status == MTLCommandBufferStatusError) {
                fprintf(stderr, "[Metal] GPU pivot block (GF16) failed\n");
                return 0;
            }
        }

        uint32_t rank = *((uint32_t *)[bufRank contents]);
        uint8_t *dM = (uint8_t *)[bufM contents];
        uint8_t *dRhs = (uint8_t *)[bufRhs contents];

        for (int r = 0; r < total_rows; ++r) {
            int nonzero = 0;
            for (int c = 0; c < cols; ++c)
                if (dM[(size_t)r * (size_t)cols + (size_t)c] != 0) { nonzero = 1; break; }
            if (!nonzero && dRhs[r] != 0) return 0;
        }

        if (rank < (uint32_t)cols) return 0;

        memset(x, 0, x_size);
        for (int c = 0; c < cols; ++c) {
            uint32_t pr = pivot_rank_for_col[c];
            if (pr == invalid_rank || pr >= (uint32_t)total_rows) return 0;
            x[c] = dRhs[pr];
        }

        memcpy(M, dM, matrix_size);
        memcpy(rhs, dRhs, rhs_size);
        return 1;
    }
}

int alg2_metal_solve_overdetermined_unique_gf4(uint8_t *M,
                                                uint8_t *rhs,
                                                int total_rows,
                                                int cols,
                                                uint8_t *x)
{
    if (!M || !rhs || !x || total_rows < cols || cols <= 0) return 0;
    if (!metal_bridge_init()) return 0;

    @autoreleasepool {
        ensure_gf4_tables();

        const uint32_t total_rows_u = (uint32_t)total_rows;
        const uint32_t cols_u = (uint32_t)cols;
        const uint32_t invalid_rank = 0xffffffffu;

        size_t matrix_size = (size_t)total_rows * (size_t)cols * sizeof(uint8_t);
        size_t rhs_size = (size_t)total_rows * sizeof(uint8_t);
        size_t x_size = (size_t)cols * sizeof(uint8_t);
        size_t mul_table_size = 256u * 256u * sizeof(uint8_t);
        size_t inv_table_size = 256u * sizeof(uint8_t);
        size_t pivot_map_size = (size_t)cols * sizeof(uint32_t);

        id<MTLBuffer> bufM = [g_device newBufferWithLength:matrix_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufRhs = [g_device newBufferWithLength:rhs_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufMul = [g_device newBufferWithBytes:gfmul_tbl length:mul_table_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufInv = [g_device newBufferWithBytes:gfinv_tbl length:inv_table_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufRank = [g_device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufPivotRow = [g_device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufPivotInv = [g_device newBufferWithLength:sizeof(uint8_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufPivotRankForCol = [g_device newBufferWithLength:pivot_map_size options:MTLResourceStorageModeShared];

        if (!bufM || !bufRhs || !bufMul || !bufInv || !bufRank || !bufPivotRow || !bufPivotInv || !bufPivotRankForCol) {
            return 0;
        }

        memcpy([bufM contents], M, matrix_size);
        memcpy([bufRhs contents], rhs, rhs_size);
        *((uint32_t *)[bufRank contents]) = 0u;
        *((uint32_t *)[bufPivotRow contents]) = total_rows_u;
        *((uint8_t *)[bufPivotInv contents]) = 0u;
        uint32_t *pivot_rank_for_col = (uint32_t *)[bufPivotRankForCol contents];
        for (int c = 0; c < cols; ++c) pivot_rank_for_col[c] = invalid_rank;

        const int block_size = ALG2_GPU_PIVOT_BLOCK_SIZE;
        for (int block_start = 0; block_start < cols; block_start += block_size) {
            int block_end = block_start + block_size;
            if (block_end > cols) block_end = cols;

            id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
            if (!cmd) return 0;

            for (int col = block_start; col < block_end; ++col) {
                uint32_t col_u = (uint32_t)col;
                encode_1d(cmd, g_pipeline_gpu_init_pivot, 1, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufPivotRow offset:0 atIndex:0];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:1];
                });
                encode_1d(cmd, g_pipeline_gpu_find_pivot, (NSUInteger)total_rows, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufM offset:0 atIndex:0];
                    [enc setBuffer:bufPivotRow offset:0 atIndex:1];
                    [enc setBuffer:bufRank offset:0 atIndex:2];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:3];
                    [enc setBytes:&cols_u length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&col_u length:sizeof(uint32_t) atIndex:5];
                });
                encode_1d(cmd, g_pipeline_gpu_swap_rows, (NSUInteger)cols, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufM offset:0 atIndex:0];
                    [enc setBuffer:bufRhs offset:0 atIndex:1];
                    [enc setBuffer:bufPivotRow offset:0 atIndex:2];
                    [enc setBuffer:bufRank offset:0 atIndex:3];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&cols_u length:sizeof(uint32_t) atIndex:5];
                });
                encode_1d(cmd, g_pipeline_gpu_compute_pivot_inv, 1, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufM offset:0 atIndex:0];
                    [enc setBuffer:bufInv offset:0 atIndex:1];
                    [enc setBuffer:bufPivotRow offset:0 atIndex:2];
                    [enc setBuffer:bufRank offset:0 atIndex:3];
                    [enc setBuffer:bufPivotInv offset:0 atIndex:4];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:5];
                    [enc setBytes:&cols_u length:sizeof(uint32_t) atIndex:6];
                    [enc setBytes:&col_u length:sizeof(uint32_t) atIndex:7];
                });
                encode_1d(cmd, g_pipeline_gpu_normalize_pivot, (NSUInteger)cols, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufM offset:0 atIndex:0];
                    [enc setBuffer:bufRhs offset:0 atIndex:1];
                    [enc setBuffer:bufMul offset:0 atIndex:2];
                    [enc setBuffer:bufPivotRow offset:0 atIndex:3];
                    [enc setBuffer:bufRank offset:0 atIndex:4];
                    [enc setBuffer:bufPivotInv offset:0 atIndex:5];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:6];
                    [enc setBytes:&cols_u length:sizeof(uint32_t) atIndex:7];
                    [enc setBytes:&col_u length:sizeof(uint32_t) atIndex:8];
                });
                encode_1d(cmd, g_pipeline_gpu_eliminate_column, (NSUInteger)total_rows, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufM offset:0 atIndex:0];
                    [enc setBuffer:bufRhs offset:0 atIndex:1];
                    [enc setBuffer:bufMul offset:0 atIndex:2];
                    [enc setBuffer:bufPivotRow offset:0 atIndex:3];
                    [enc setBuffer:bufRank offset:0 atIndex:4];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:5];
                    [enc setBytes:&cols_u length:sizeof(uint32_t) atIndex:6];
                    [enc setBytes:&col_u length:sizeof(uint32_t) atIndex:7];
                });
                encode_1d(cmd, g_pipeline_gpu_advance_rank, 1, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufPivotRow offset:0 atIndex:0];
                    [enc setBuffer:bufRank offset:0 atIndex:1];
                    [enc setBuffer:bufPivotRankForCol offset:0 atIndex:2];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:3];
                    [enc setBytes:&col_u length:sizeof(uint32_t) atIndex:4];
                });
            }

            [cmd commit];
            [cmd waitUntilCompleted];
            if (cmd.status == MTLCommandBufferStatusError) {
                fprintf(stderr, "[Metal] GPU pivot block (GF4) failed\n");
                return 0;
            }
        }

        uint32_t rank = *((uint32_t *)[bufRank contents]);
        uint8_t *dM = (uint8_t *)[bufM contents];
        uint8_t *dRhs = (uint8_t *)[bufRhs contents];

        for (int r = 0; r < total_rows; ++r) {
            int nonzero = 0;
            for (int c = 0; c < cols; ++c)
                if (dM[(size_t)r * (size_t)cols + (size_t)c] != 0) { nonzero = 1; break; }
            if (!nonzero && dRhs[r] != 0) return 0;
        }

        if (rank < (uint32_t)cols) return 0;

        memset(x, 0, x_size);
        for (int c = 0; c < cols; ++c) {
            uint32_t pr = pivot_rank_for_col[c];
            if (pr == invalid_rank || pr >= (uint32_t)total_rows) return 0;
            x[c] = dRhs[pr];
        }

        memcpy(M, dM, matrix_size);
        memcpy(rhs, dRhs, rhs_size);
        return 1;
    }
}

int alg2_metal_solve_overdetermined_unique_gf2(uint8_t *M,
                                                uint8_t *rhs,
                                                int total_rows,
                                                int cols,
                                                uint8_t *x)
{
    if (!M || !rhs || !x || total_rows < cols || cols <= 0) return 0;
    if (!metal_bridge_init()) return 0;

    @autoreleasepool {
        ensure_gf2_tables();

        const uint32_t total_rows_u = (uint32_t)total_rows;
        const uint32_t cols_u = (uint32_t)cols;
        const uint32_t invalid_rank = 0xffffffffu;

        size_t matrix_size = (size_t)total_rows * (size_t)cols * sizeof(uint8_t);
        size_t rhs_size = (size_t)total_rows * sizeof(uint8_t);
        size_t x_size = (size_t)cols * sizeof(uint8_t);
        size_t mul_table_size = 256u * 256u * sizeof(uint8_t);
        size_t inv_table_size = 256u * sizeof(uint8_t);
        size_t pivot_map_size = (size_t)cols * sizeof(uint32_t);

        id<MTLBuffer> bufM = [g_device newBufferWithLength:matrix_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufRhs = [g_device newBufferWithLength:rhs_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufMul = [g_device newBufferWithBytes:gfmul_tbl length:mul_table_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufInv = [g_device newBufferWithBytes:gfinv_tbl length:inv_table_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufRank = [g_device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufPivotRow = [g_device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufPivotInv = [g_device newBufferWithLength:sizeof(uint8_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufPivotRankForCol = [g_device newBufferWithLength:pivot_map_size options:MTLResourceStorageModeShared];

        if (!bufM || !bufRhs || !bufMul || !bufInv || !bufRank || !bufPivotRow || !bufPivotInv || !bufPivotRankForCol) {
            return 0;
        }

        memcpy([bufM contents], M, matrix_size);
        memcpy([bufRhs contents], rhs, rhs_size);
        *((uint32_t *)[bufRank contents]) = 0u;
        *((uint32_t *)[bufPivotRow contents]) = total_rows_u;
        *((uint8_t *)[bufPivotInv contents]) = 0u;
        uint32_t *pivot_rank_for_col = (uint32_t *)[bufPivotRankForCol contents];
        for (int c = 0; c < cols; ++c) pivot_rank_for_col[c] = invalid_rank;

        const int block_size = ALG2_GPU_PIVOT_BLOCK_SIZE;
        for (int block_start = 0; block_start < cols; block_start += block_size) {
            int block_end = block_start + block_size;
            if (block_end > cols) block_end = cols;

            id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
            if (!cmd) return 0;

            for (int col = block_start; col < block_end; ++col) {
                uint32_t col_u = (uint32_t)col;
                encode_1d(cmd, g_pipeline_gpu_init_pivot, 1, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufPivotRow offset:0 atIndex:0];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:1];
                });
                encode_1d(cmd, g_pipeline_gpu_find_pivot, (NSUInteger)total_rows, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufM offset:0 atIndex:0];
                    [enc setBuffer:bufPivotRow offset:0 atIndex:1];
                    [enc setBuffer:bufRank offset:0 atIndex:2];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:3];
                    [enc setBytes:&cols_u length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&col_u length:sizeof(uint32_t) atIndex:5];
                });
                encode_1d(cmd, g_pipeline_gpu_swap_rows, (NSUInteger)cols, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufM offset:0 atIndex:0];
                    [enc setBuffer:bufRhs offset:0 atIndex:1];
                    [enc setBuffer:bufPivotRow offset:0 atIndex:2];
                    [enc setBuffer:bufRank offset:0 atIndex:3];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&cols_u length:sizeof(uint32_t) atIndex:5];
                });
                encode_1d(cmd, g_pipeline_gpu_compute_pivot_inv, 1, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufM offset:0 atIndex:0];
                    [enc setBuffer:bufInv offset:0 atIndex:1];
                    [enc setBuffer:bufPivotRow offset:0 atIndex:2];
                    [enc setBuffer:bufRank offset:0 atIndex:3];
                    [enc setBuffer:bufPivotInv offset:0 atIndex:4];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:5];
                    [enc setBytes:&cols_u length:sizeof(uint32_t) atIndex:6];
                    [enc setBytes:&col_u length:sizeof(uint32_t) atIndex:7];
                });
                encode_1d(cmd, g_pipeline_gpu_normalize_pivot, (NSUInteger)cols, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufM offset:0 atIndex:0];
                    [enc setBuffer:bufRhs offset:0 atIndex:1];
                    [enc setBuffer:bufMul offset:0 atIndex:2];
                    [enc setBuffer:bufPivotRow offset:0 atIndex:3];
                    [enc setBuffer:bufRank offset:0 atIndex:4];
                    [enc setBuffer:bufPivotInv offset:0 atIndex:5];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:6];
                    [enc setBytes:&cols_u length:sizeof(uint32_t) atIndex:7];
                    [enc setBytes:&col_u length:sizeof(uint32_t) atIndex:8];
                });
                encode_1d(cmd, g_pipeline_gpu_eliminate_column, (NSUInteger)total_rows, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufM offset:0 atIndex:0];
                    [enc setBuffer:bufRhs offset:0 atIndex:1];
                    [enc setBuffer:bufMul offset:0 atIndex:2];
                    [enc setBuffer:bufPivotRow offset:0 atIndex:3];
                    [enc setBuffer:bufRank offset:0 atIndex:4];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:5];
                    [enc setBytes:&cols_u length:sizeof(uint32_t) atIndex:6];
                    [enc setBytes:&col_u length:sizeof(uint32_t) atIndex:7];
                });
                encode_1d(cmd, g_pipeline_gpu_advance_rank, 1, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufPivotRow offset:0 atIndex:0];
                    [enc setBuffer:bufRank offset:0 atIndex:1];
                    [enc setBuffer:bufPivotRankForCol offset:0 atIndex:2];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:3];
                    [enc setBytes:&col_u length:sizeof(uint32_t) atIndex:4];
                });
            }

            [cmd commit];
            [cmd waitUntilCompleted];
            if (cmd.status == MTLCommandBufferStatusError) {
                fprintf(stderr, "[Metal] GPU pivot block (GF2) failed\n");
                return 0;
            }
        }

        uint32_t rank = *((uint32_t *)[bufRank contents]);
        uint8_t *dM = (uint8_t *)[bufM contents];
        uint8_t *dRhs = (uint8_t *)[bufRhs contents];

        for (int r = 0; r < total_rows; ++r) {
            int nonzero = 0;
            for (int c = 0; c < cols; ++c)
                if (dM[(size_t)r * (size_t)cols + (size_t)c] != 0) { nonzero = 1; break; }
            if (!nonzero && dRhs[r] != 0) return 0;
        }

        if (rank < (uint32_t)cols) return 0;

        memset(x, 0, x_size);
        for (int c = 0; c < cols; ++c) {
            uint32_t pr = pivot_rank_for_col[c];
            if (pr == invalid_rank || pr >= (uint32_t)total_rows) return 0;
            x[c] = dRhs[pr];
        }

        memcpy(M, dM, matrix_size);
        memcpy(rhs, dRhs, rhs_size);
        return 1;
    }
}

/*
 * Direct binary (GF2) elimination — no table lookups, pure XOR.
 * No normalize_pivot or compute_pivot_inv stages needed — skips 2 buffers and 3 dispatch steps.
 */
int alg2_metal_solve_overdetermined_unique_bin(uint8_t *M,
                                                uint8_t *rhs,
                                                int total_rows,
                                                int cols,
                                                uint8_t *x)
{
    if (!M || !rhs || !x || total_rows < cols || cols <= 0) return 0;
    if (!metal_bridge_init()) return 0;

    @autoreleasepool {
        const uint32_t total_rows_u = (uint32_t)total_rows;
        const uint32_t cols_u = (uint32_t)cols;
        const uint32_t invalid_rank = 0xffffffffu;

        size_t matrix_size = (size_t)total_rows * (size_t)cols * sizeof(uint8_t);
        size_t rhs_size = (size_t)total_rows * sizeof(uint8_t);
        size_t x_size = (size_t)cols * sizeof(uint8_t);
        size_t pivot_map_size = (size_t)cols * sizeof(uint32_t);

        id<MTLBuffer> bufM = [g_device newBufferWithLength:matrix_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufRhs = [g_device newBufferWithLength:rhs_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufRank = [g_device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufPivotRow = [g_device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufPivotRankForCol = [g_device newBufferWithLength:pivot_map_size options:MTLResourceStorageModeShared];

        if (!bufM || !bufRhs || !bufRank || !bufPivotRow || !bufPivotRankForCol) {
            return 0;
        }

        memcpy([bufM contents], M, matrix_size);
        memcpy([bufRhs contents], rhs, rhs_size);
        *((uint32_t *)[bufRank contents]) = 0u;
        uint32_t *pivot_rank_for_col_bin = (uint32_t *)[bufPivotRankForCol contents];
        for (int c = 0; c < cols; ++c) pivot_rank_for_col_bin[c] = invalid_rank;

        const int block_size = ALG2_GPU_PIVOT_BLOCK_SIZE;
        for (int block_start = 0; block_start < cols; block_start += block_size) {
            int block_end = block_start + block_size;
            if (block_end > cols) block_end = cols;

            id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
            if (!cmd) return 0;

            for (int col = block_start; col < block_end; ++col) {
                uint32_t col_u = (uint32_t)col;

                /* Use existing GPU init_pivot kernel for proper synchronization */
                encode_1d(cmd, g_pipeline_gpu_init_pivot, 1, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufPivotRow offset:0 atIndex:0];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:1];
                });

                encode_1d(cmd, g_pipeline_bin_find_pivot, (NSUInteger)total_rows, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufM offset:0 atIndex:0];
                    [enc setBuffer:bufPivotRow offset:0 atIndex:1];
                    [enc setBuffer:bufRank offset:0 atIndex:2];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:3];
                    [enc setBytes:&cols_u length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&col_u length:sizeof(uint32_t) atIndex:5];
                });

                encode_1d(cmd, g_pipeline_bin_swap_rows, (NSUInteger)cols, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufM offset:0 atIndex:0];
                    [enc setBuffer:bufRhs offset:0 atIndex:1];
                    [enc setBuffer:bufPivotRow offset:0 atIndex:2];
                    [enc setBuffer:bufRank offset:0 atIndex:3];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&cols_u length:sizeof(uint32_t) atIndex:5];
                });

                encode_1d(cmd, g_pipeline_bin_eliminate_column, (NSUInteger)total_rows, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufM offset:0 atIndex:0];
                    [enc setBuffer:bufRhs offset:0 atIndex:1];
                    [enc setBuffer:bufPivotRow offset:0 atIndex:2];
                    [enc setBuffer:bufRank offset:0 atIndex:3];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&cols_u length:sizeof(uint32_t) atIndex:5];
                    [enc setBytes:&col_u length:sizeof(uint32_t) atIndex:6];
                });

                encode_1d(cmd, g_pipeline_bin_advance_rank, 1, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:bufPivotRow offset:0 atIndex:0];
                    [enc setBuffer:bufRank offset:0 atIndex:1];
                    [enc setBuffer:bufPivotRankForCol offset:0 atIndex:2];
                    [enc setBytes:&total_rows_u length:sizeof(uint32_t) atIndex:3];
                    [enc setBytes:&col_u length:sizeof(uint32_t) atIndex:4];
                });
            }

            [cmd commit];
            [cmd waitUntilCompleted];
            if (cmd.status == MTLCommandBufferStatusError) {
                return 0;
            }
        }

        uint32_t rank = *((uint32_t *)[bufRank contents]);
        uint8_t *dM = (uint8_t *)[bufM contents];
        uint8_t *dRhs = (uint8_t *)[bufRhs contents];

        for (int r = 0; r < total_rows; ++r) {
            int nonzero = 0;
            for (int c = 0; c < cols; ++c)
                if (dM[(size_t)r * (size_t)cols + (size_t)c]) { nonzero = 1; break; }
            if (!nonzero && dRhs[r]) return 0;
        }

        if (rank < (uint32_t)cols) return 0;

        memset(x, 0, x_size);
        for (int c = 0; c < cols; ++c) {
            uint32_t pr = ((uint32_t *)[bufPivotRankForCol contents])[c];
            if (pr == invalid_rank || pr >= (uint32_t)total_rows) return 0;
            x[c] = dRhs[pr];
        }

        memcpy(M, dM, matrix_size);
        memcpy(rhs, dRhs, rhs_size);
        return 1;
    }
}
