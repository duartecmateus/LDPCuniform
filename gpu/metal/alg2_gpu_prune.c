// alg2_gpu_prune.c — GPU-accelerated pruning decoder.
// Uses Metal for column-intersection computation and elimination on
// a compact reduced system.

#include "alg2.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__APPLE__) && defined(ENABLE_ALG2_METAL)
#include "metal_bridge.h"
#endif

#include "gf256.h"

static int hamming_weight(const gf256_t *v, int n)
{
    int w = 0;
    for (int i = 0; i < n; ++i)
        if (v[i] != 0) ++w;
    return w;
}

/* CPU Gauss-Jordan for overdetermined/rectangular systems. */
static int solve_overdetermined_cpu(gf256_t *M, gf256_t *rhs,
                                     int rows, int cols, gf256_t *x)
{
    if (rows < cols) return 0;
    gf256_init();
    int *pivot = (int *)calloc((size_t)cols, sizeof(int));
    if (!pivot) return 0;
    for (int i = 0; i < cols; ++i) pivot[i] = -1;
    int rank = 0;
    for (int c = 0; c < cols && rank < rows; ++c) {
        int p = -1;
        for (int r = rank; r < rows; ++r)
            if (M[(size_t)r * (size_t)cols + (size_t)c] != 0) { p = r; break; }
        if (p < 0) continue;
        if (p != rank) {
            for (int j = c; j < cols; ++j) {
                gf256_t t = M[(size_t)rank * (size_t)cols + (size_t)j];
                M[(size_t)rank * (size_t)cols + (size_t)j] = M[(size_t)p * (size_t)cols + (size_t)j];
                M[(size_t)p * (size_t)cols + (size_t)j] = t;
            }
            gf256_t t = rhs[rank]; rhs[rank] = rhs[p]; rhs[p] = t;
        }
        gf256_t pv = M[(size_t)rank * (size_t)cols + (size_t)c];
        if (pv == 0) { free(pivot); return 0; }
        gf256_t inv = gf256_inv(pv);
        for (int j = c; j < cols; ++j)
            M[(size_t)rank * (size_t)cols + (size_t)j] = gf256_mul(M[(size_t)rank * (size_t)cols + (size_t)j], inv);
        rhs[rank] = gf256_mul(rhs[rank], inv);
        for (int r = 0; r < rows; ++r) {
            if (r == rank) continue;
            gf256_t f = M[(size_t)r * (size_t)cols + (size_t)c];
            if (f == 0) continue;
            for (int j = c; j < cols; ++j)
                M[(size_t)r * (size_t)cols + (size_t)j] = gf256_add(
                    M[(size_t)r * (size_t)cols + (size_t)j],
                    gf256_mul(f, M[(size_t)rank * (size_t)cols + (size_t)j]));
            rhs[r] = gf256_add(rhs[r], gf256_mul(f, rhs[rank]));
        }
        pivot[c] = rank++;
    }
    memset(x, 0, (size_t)cols * sizeof(gf256_t));
    for (int c = 0; c < cols; ++c)
        if (pivot[c] >= 0) x[c] = rhs[pivot[c]];
    free(pivot);
    return 1;
}

/* Use Metal GPU solver when available. */
static int solve_overdetermined(gf256_t *M, gf256_t *rhs,
                                 int rows, int cols, gf256_t *x)
{
#if defined(__APPLE__) && defined(ENABLE_ALG2_METAL)
    if (cols >= 1024 && metal_bridge_init()) {
        return alg2_metal_solve_overdetermined_unique(
            (uint8_t *)M, (uint8_t *)rhs, rows, cols, (uint8_t *)x);
    }
#endif
    return solve_overdetermined_cpu(M, rhs, rows, cols, x);
}

/*
 * alg2_gpu_prune — GPU-accelerated pruning decoder.
 *
 * Algorithm matches the CPU pruning approach:
 * 1. Compute column-syndrome intersections (GPU via Metal when cols ≥ 1024)
 * 2. Build "sure" set: columns with intersection > threshold (likely errors)
 * 3. Build "candidate" set: columns with intersection ≤ threshold (unlikely)
 * 4. Construct a compact rows × target system from the sure columns
 * 5. Solve the overdetermined system (GPU via Metal)
 * 6. Map the solution back to the original column positions
 *
 * The system is overdetermined (target = rows/2) so the solver favours
 * sparse solutions that match the syndrome.
 */
int alg2_gpu_prune(const gf256_t *A, const gf256_t *v, gf256_t *x,
                   int rows, int cols,
                   int max_weight, int max_tries,
                   int max_syndrome_intersections)
{
    static int seeded = 0;
    if (!seeded) { srand(54321u); seeded = 1; }

    if (!A || !v || !x || rows <= 0 || cols <= 0 || rows > cols ||
        max_weight < 0 || max_tries <= 0)
        return 0;

    const int extra_margin = (rows > 100) ? 50 : (rows / 2);
    const int target = rows - extra_margin;  // square-ish system: rows × target
    if (target <= 0) return 0;
    if (max_syndrome_intersections < 0) max_syndrome_intersections = 0;

    int *intersection = (int *)calloc((size_t)cols, sizeof(int));
    int *cands = (int *)malloc((size_t)cols * sizeof(int));

    if (!intersection || !cands) {
        free(intersection); free(cands);
        return 0;
    }

    // Step 1: column-syndrome intersections (Metal GPU when cols ≥ 1024)
#if defined(__APPLE__) && defined(ENABLE_ALG2_METAL)
    if (cols >= 1024 && metal_bridge_init()) {
        int dummy = 0;
        alg2_metal_compute_column_intersections(
            (const uint8_t *)A, (const uint8_t *)v,
            rows, cols, intersection, &dummy);
    } else
#endif
    {
        memset(intersection, 0, (size_t)cols * sizeof(int));
        for (int col = 0; col < cols; ++col)
            for (int r = 0; r < rows; ++r)
                if (A[(size_t)r * (size_t)cols + (size_t)col] != 0 && v[r] != 0)
                    intersection[col]++;
    }

    // Step 2: classify. Sort sure columns by intersection count descending.
    typedef struct { int idx, inter; } IdxInter;
    IdxInter *sure_sorted = (IdxInter *)malloc((size_t)cols * sizeof(IdxInter));
    int sure_count = 0, cand_count = 0;
    for (int i = 0; i < cols; ++i) {
        if (intersection[i] > max_syndrome_intersections) {
            sure_sorted[sure_count].idx = i;
            sure_sorted[sure_count].inter = intersection[i];
            sure_count++;
        } else {
            cands[cand_count++] = i;
        }
    }
    // Sort sure columns by intersection descending
    for (int i = 0; i < sure_count - 1; ++i)
        for (int j = i + 1; j < sure_count; ++j)
            if (sure_sorted[j].inter > sure_sorted[i].inter) {
                IdxInter t = sure_sorted[i]; sure_sorted[i] = sure_sorted[j]; sure_sorted[j] = t;
            }

    printf("[alg2_gpu] cols=%d, sure=%d (inter>%d), candidates=%d, rows=%d, target=%d\n",
           cols, sure_count, max_syndrome_intersections, cand_count, rows, target);

    // Step 3: iteratively try subsets of top sure columns
    for (int attempt = 0; attempt < max_tries; ++attempt) {
        int n_sel = (sure_count < target) ? sure_count : target;
        if (n_sel < 1) break;

        int *selected = (int *)malloc((size_t)n_sel * sizeof(int));
        gf256_t *M = (gf256_t *)malloc((size_t)rows * (size_t)n_sel * sizeof(gf256_t));
        gf256_t *cand_x = (gf256_t *)calloc((size_t)cols, sizeof(gf256_t));
        if (!selected || !M || !cand_x) {
            free(selected); free(M); free(cand_x);
            break;
        }

        // Match CPU algorithm:
        // 1. Use all sure columns
        // 2. If not enough, fill with random candidates
        // 3. If too many sure, shuffle within top 2*target for later attempts
        int orig_sure = (sure_count < target) ? sure_count : target;
        for (int i = 0; i < orig_sure; ++i) selected[i] = sure_sorted[i].idx;
        
        if (sure_count >= target) {
            // We have enough sure — use first target, shuffle in later attempts
            if (attempt > 0) {
                for (int i = target - 1; i > 0; --i) {
                    int j = rand() % (i + 1);
                    IdxInter t = sure_sorted[i]; sure_sorted[i] = sure_sorted[j]; sure_sorted[j] = t;
                }
                for (int i = 0; i < target; ++i) selected[i] = sure_sorted[i].idx;
            }
        } else {
            // Not enough sure — fill rest with random candidates
            int need = target - sure_count;
            if (need > cand_count) {
                free(selected); free(M); free(cand_x);
                break;
            }
            // Shuffle candidates and take first `need`
            for (int i = cand_count - 1; i > 0; --i) {
                int j = rand() % (i + 1);
                int t = cands[i]; cands[i] = cands[j]; cands[j] = t;
            }
            for (int i = 0; i < need; ++i)
                selected[sure_count + i] = cands[i];
        }

        // Build compact system: rows × n_sel
        memset(M, 0, (size_t)rows * (size_t)n_sel * sizeof(gf256_t));
        for (int r = 0; r < rows; ++r)
            for (int k = 0; k < n_sel; ++k)
                M[(size_t)r * (size_t)n_sel + (size_t)k] =
                    A[(size_t)r * (size_t)cols + (size_t)selected[k]];

        // Solve
        if (!solve_overdetermined(M, (gf256_t *)v, rows, n_sel, cand_x))
            { free(selected); free(M); free(cand_x); continue; }

        // Map back
        memset(x, 0, (size_t)cols * sizeof(gf256_t));
        for (int k = 0; k < n_sel; ++k)
            x[selected[k]] = cand_x[k];

        int w = hamming_weight(x, cols);
        if (w <= max_weight) {
            printf("[alg2_gpu] OK attempt %d, weight=%d\n", attempt + 1, w);
            free(intersection); free(sure_sorted); free(cands);
            free(selected); free(M); free(cand_x);
            return attempt + 1;
        }
        free(selected); free(M); free(cand_x);
    }

    free(intersection); free(sure_sorted); free(cands);
    return 0;
}
