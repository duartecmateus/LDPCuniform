#include <metal_stdlib>
using namespace metal;

static inline uchar gf256_mul_metal(uchar a,
                                    uchar b,
                                    device const uchar *gfmul)
{
    return gfmul[((uint)a << 8) | (uint)b];
}

/*
 * For each column, count how many nonzero entries of that column occur on rows
 * where syndrome[row] is nonzero.
 */
kernel void alg2_column_intersections(device const uchar *matrix [[buffer(0)]],
                                      device const uchar *syndrome [[buffer(1)]],
                                      device int *intersection_count [[buffer(2)]],
                                      constant int &rows [[buffer(3)]],
                                      constant int &cols [[buffer(4)]],
                                      uint col [[thread_position_in_grid]])
{
    if ((int)col >= cols) return;

    int count = 0;
    for (int r = 0; r < rows; ++r) {
        if (syndrome[r] != 0 && matrix[(uint)r * (uint)cols + col] != 0) {
            ++count;
        }
    }
    intersection_count[col] = count;
}

/*
 * Legacy rectangular row update for one pivot.  Kept for compatibility.
 * Buffer 2 is now the 256x256 multiplication table.
 */
kernel void alg2_rect_eliminate_column(device uchar *M [[buffer(0)]],
                                       device uchar *rhs [[buffer(1)]],
                                       device const uchar *gfmul [[buffer(2)]],
                                       device const uchar *unused [[buffer(3)]],
                                       constant int &total_rows [[buffer(4)]],
                                       constant int &cols [[buffer(5)]],
                                       constant int &pivot_rank [[buffer(6)]],
                                       constant int &col [[buffer(7)]],
                                       uint row [[thread_position_in_grid]])
{
    (void)unused;
    if ((int)row >= total_rows) return;
    if ((int)row == pivot_rank) return;

    device uchar *target = M + (uint)row * (uint)cols;
    device uchar *pivot  = M + (uint)pivot_rank * (uint)cols;

    uchar factor = target[col];
    if (factor == 0) return;

    target[col] = 0;
    uint fbase = ((uint)factor << 8);
    for (int j = col + 1; j < cols; ++j) {
        target[j] ^= gfmul[fbase | (uint)pivot[j]];
    }
    rhs[row] ^= gfmul[fbase | (uint)rhs[pivot_rank]];
}

/*
 * GPU-resident pivot-step kernels for rectangular GF(256) elimination.
 * These kernels keep pivot search, row swap, normalization, elimination, and
 * rank update on the GPU.  The host can encode several columns into one command
 * buffer to reduce CPU/GPU synchronization overhead.
 */
kernel void alg2_gpu_init_pivot(device atomic_uint *pivot_row [[buffer(0)]],
                                constant uint &total_rows [[buffer(1)]],
                                uint tid [[thread_position_in_grid]])
{
    if (tid == 0) {
        atomic_store_explicit(pivot_row, total_rows, memory_order_relaxed);
    }
}

kernel void alg2_gpu_find_pivot(device const uchar *M [[buffer(0)]],
                                device atomic_uint *pivot_row [[buffer(1)]],
                                device const uint *rank_buf [[buffer(2)]],
                                constant uint &total_rows [[buffer(3)]],
                                constant uint &cols [[buffer(4)]],
                                constant uint &col [[buffer(5)]],
                                uint row [[thread_position_in_grid]])
{
    if (row >= total_rows) return;
    uint rank = rank_buf[0];
    if (row < rank) return;

    if (M[row * cols + col] != 0) {
        atomic_fetch_min_explicit(pivot_row, row, memory_order_relaxed);
    }
}

kernel void alg2_gpu_swap_rows(device uchar *M [[buffer(0)]],
                               device uchar *rhs [[buffer(1)]],
                               device atomic_uint *pivot_row [[buffer(2)]],
                               device const uint *rank_buf [[buffer(3)]],
                               constant uint &total_rows [[buffer(4)]],
                               constant uint &cols [[buffer(5)]],
                               uint j [[thread_position_in_grid]])
{
    if (j >= cols) return;

    uint pivot = atomic_load_explicit(pivot_row, memory_order_relaxed);
    if (pivot >= total_rows) return;

    uint rank = rank_buf[0];
    if (pivot == rank) return;

    uchar tmp = M[rank * cols + j];
    M[rank * cols + j] = M[pivot * cols + j];
    M[pivot * cols + j] = tmp;

    if (j == 0) {
        uchar tb = rhs[rank];
        rhs[rank] = rhs[pivot];
        rhs[pivot] = tb;
    }
}

kernel void alg2_gpu_compute_pivot_inv(device uchar *M [[buffer(0)]],
                                        device const uchar *gfinv [[buffer(1)]],
                                        device atomic_uint *pivot_row [[buffer(2)]],
                                        device const uint *rank_buf [[buffer(3)]],
                                        device uchar *pivot_inv [[buffer(4)]],
                                        constant uint &total_rows [[buffer(5)]],
                                        constant uint &cols [[buffer(6)]],
                                        constant uint &col [[buffer(7)]],
                                        uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;

    uint pivot = atomic_load_explicit(pivot_row, memory_order_relaxed);
    if (pivot >= total_rows) {
        pivot_inv[0] = 0;
        return;
    }

    uint rank = rank_buf[0];
    uchar pv = M[rank * cols + col];
    if (pv == 0) {
        pivot_inv[0] = 0;
        atomic_store_explicit(pivot_row, total_rows, memory_order_relaxed);
        return;
    }
    pivot_inv[0] = gfinv[(uint)pv];
}

kernel void alg2_gpu_normalize_pivot(device uchar *M [[buffer(0)]],
                                     device uchar *rhs [[buffer(1)]],
                                     device const uchar *gfmul [[buffer(2)]],
                                     device atomic_uint *pivot_row [[buffer(3)]],
                                     device const uint *rank_buf [[buffer(4)]],
                                     device const uchar *pivot_inv [[buffer(5)]],
                                     constant uint &total_rows [[buffer(6)]],
                                     constant uint &cols [[buffer(7)]],
                                     constant uint &col [[buffer(8)]],
                                     uint j [[thread_position_in_grid]])
{
    if (j >= cols) return;

    uint pivot = atomic_load_explicit(pivot_row, memory_order_relaxed);
    if (pivot >= total_rows) return;

    uchar inv = pivot_inv[0];
    if (inv == 0) return;

    uint rank = rank_buf[0];
    device uchar *prow = M + rank * cols;

    if (j >= col) {
        prow[j] = gfmul[((uint)prow[j] << 8) | (uint)inv];
    }
    if (j == 0) {
        rhs[rank] = gfmul[((uint)rhs[rank] << 8) | (uint)inv];
    }
}

kernel void alg2_gpu_eliminate_column(device uchar *M [[buffer(0)]],
                                      device uchar *rhs [[buffer(1)]],
                                      device const uchar *gfmul [[buffer(2)]],
                                      device atomic_uint *pivot_row [[buffer(3)]],
                                      device const uint *rank_buf [[buffer(4)]],
                                      constant uint &total_rows [[buffer(5)]],
                                      constant uint &cols [[buffer(6)]],
                                      constant uint &col [[buffer(7)]],
                                      uint row [[thread_position_in_grid]])
{
    if (row >= total_rows) return;

    uint pivot = atomic_load_explicit(pivot_row, memory_order_relaxed);
    if (pivot >= total_rows) return;

    uint rank = rank_buf[0];
    if (row == rank) return;

    device uchar *target = M + row * cols;
    device uchar *pivotr = M + rank * cols;

    uchar factor = target[col];
    if (factor == 0) return;

    target[col] = 0;
    uint fbase = ((uint)factor << 8);
    for (uint j = col + 1; j < cols; ++j) {
        target[j] ^= gfmul[fbase | (uint)pivotr[j]];
    }
    rhs[row] ^= gfmul[fbase | (uint)rhs[rank]];
}

kernel void alg2_gpu_advance_rank(device atomic_uint *pivot_row [[buffer(0)]],
                                  device uint *rank_buf [[buffer(1)]],
                                  device uint *pivot_rank_for_col [[buffer(2)]],
                                  constant uint &total_rows [[buffer(3)]],
                                  constant uint &col [[buffer(4)]],
                                  uint tid [[thread_position_in_grid]])
{
    if (tid == 0) {
        uint pivot = atomic_load_explicit(pivot_row, memory_order_relaxed);
        if (pivot < total_rows) {
            uint old_rank = rank_buf[0];
            pivot_rank_for_col[col] = old_rank;
            rank_buf[0] = old_rank + 1;
        }
    }
}

/*
 * Binary (GF(2)) GPU-accelerated elimination kernels.
 * No multiplication table — uses direct XOR only.
 */

kernel void bin_gpu_find_pivot(device const uchar *M [[buffer(0)]],
                               device atomic_uint *pivot_row [[buffer(1)]],
                               device const uint *rank_buf [[buffer(2)]],
                               constant uint &total_rows [[buffer(3)]],
                               constant uint &cols [[buffer(4)]],
                               constant uint &col [[buffer(5)]],
                               uint row [[thread_position_in_grid]])
{
    if (row >= total_rows) return;
    uint rank = rank_buf[0];
    if (row < rank) return;
    if (M[row * cols + col] != 0) {
        atomic_fetch_min_explicit(pivot_row, row, memory_order_relaxed);
    }
}

kernel void bin_gpu_swap_rows(device uchar *M [[buffer(0)]],
                              device uchar *rhs [[buffer(1)]],
                              device atomic_uint *pivot_row [[buffer(2)]],
                              device const uint *rank_buf [[buffer(3)]],
                              constant uint &total_rows [[buffer(4)]],
                              constant uint &cols [[buffer(5)]],
                              uint j [[thread_position_in_grid]])
{
    if (j >= cols) return;
    uint pivot = atomic_load_explicit(pivot_row, memory_order_relaxed);
    if (pivot >= total_rows) return;
    uint rank = rank_buf[0];
    if (pivot == rank) return;
    uchar tmp = M[rank * cols + j];
    M[rank * cols + j] = M[pivot * cols + j];
    M[pivot * cols + j] = tmp;
    if (j == 0) {
        uchar tb = rhs[rank];
        rhs[rank] = rhs[pivot];
        rhs[pivot] = tb;
    }
}

/* Vectorized XOR elimination for binary using uchar8 (8 bytes at a time) */
kernel void bin_gpu_eliminate_column(device uchar *M [[buffer(0)]],
                                     device uchar *rhs [[buffer(1)]],
                                     device atomic_uint *pivot_row [[buffer(2)]],
                                     device const uint *rank_buf [[buffer(3)]],
                                     constant uint &total_rows [[buffer(4)]],
                                     constant uint &cols [[buffer(5)]],
                                     constant uint &col [[buffer(6)]],
                                     uint row [[thread_position_in_grid]])
{
    if (row >= total_rows) return;
    uint pivot = atomic_load_explicit(pivot_row, memory_order_relaxed);
    if (pivot >= total_rows) return;
    uint rank = rank_buf[0];
    if (row == rank) return;

    device uchar *target = M + row * cols;
    device uchar *pivotr = M + rank * cols;

    uchar f = target[col];
    if (f == 0) return;
    target[col] = 0;

    uint j = col + 1;
    /* 8 bytes at a time using vec type (no alignment required) */
    for (; j + 8 <= cols; j += 8) {
        device vec<uchar, 8> *t8 = (device vec<uchar, 8> *)(target + j);
        device vec<uchar, 8> *p8 = (device vec<uchar, 8> *)(pivotr + j);
        *t8 ^= *p8;
    }
    for (; j < cols; ++j) {
        target[j] ^= pivotr[j];
    }
    rhs[row] ^= rhs[rank];
}

kernel void bin_gpu_advance_rank(device atomic_uint *pivot_row [[buffer(0)]],
                                 device uint *rank_buf [[buffer(1)]],
                                 device uint *pivot_rank_for_col [[buffer(2)]],
                                 constant uint &total_rows [[buffer(3)]],
                                 constant uint &col [[buffer(4)]],
                                 uint tid [[thread_position_in_grid]])
{
    if (tid == 0) {
        uint pivot = atomic_load_explicit(pivot_row, memory_order_relaxed);
        if (pivot < total_rows) {
            uint old_rank = rank_buf[0];
            pivot_rank_for_col[col] = old_rank;
            rank_buf[0] = old_rank + 1;
        }
    }
}
