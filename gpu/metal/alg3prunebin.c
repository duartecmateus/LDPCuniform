#include "alg3prunebin.h"
#include "metal_bridge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

static void compute_intersections_sparse(const SparseMatrix *H,
                                         const uint8_t *syndrome,
                                         int *intersection_count)
{
    for (int j = 0; j < H->cols; ++j) {
        int count = 0;
        const SparseCol *sc = &H->col[j];
        for (int k = 0; k < sc->c; ++k) {
            if (syndrome[sc->rows[k]] != 0) ++count;
        }
        intersection_count[j] = count;
    }
}

typedef struct { int idx; int inter; } Cand;

static int cmp_cand_desc(const void *a, const void *b)
{
    const Cand *ca = (const Cand *)a;
    const Cand *cb = (const Cand *)b;
    if (ca->inter != cb->inter)
        return (ca->inter < cb->inter) - (ca->inter > cb->inter);
    return (rand() % 3) - 1;
}

/* XOR a pivot row into another row using word-wise transfer */
static inline void xor_rows(uint8_t *restrict row,
                             const uint8_t *restrict pr,
                             int cols, int start_col)
{
    int j = start_col;
    for (; j + 8 <= cols; j += 8)
        *(uint64_t *)(row + j) ^= *(const uint64_t *)(pr + j);
    for (; j < cols; ++j)
        row[j] ^= pr[j];
}

static void swap_rows(uint8_t *M, uint8_t *rhs, int cols, int r1, int r2)
{
    if (r1 == r2) return;
    uint8_t *row1 = M + (size_t)r1 * (size_t)cols;
    uint8_t *row2 = M + (size_t)r2 * (size_t)cols;
    int j = 0;
    for (; j + 8 <= cols; j += 8) {
        uint64_t t = *(uint64_t *)(row1 + j);
        *(uint64_t *)(row1 + j) = *(uint64_t *)(row2 + j);
        *(uint64_t *)(row2 + j) = t;
    }
    for (; j < cols; ++j) { uint8_t t = row1[j]; row1[j] = row2[j]; row2[j] = t; }
    uint8_t tb = rhs[r1]; rhs[r1] = rhs[r2]; rhs[r2] = tb;
}

static int solve_overdetermined_bin(uint8_t *restrict M,
                                     uint8_t *restrict rhs,
                                     int total_rows, int cols,
                                     uint8_t *restrict x,
                                     int *restrict pivot_row)
{
    if (!pivot_row) return 0;
    for (int c = 0; c < cols; ++c) pivot_row[c] = -1;
    int rank = 0;
    for (int col = 0; col < cols && rank < total_rows; ++col) {
        int pivot = -1;
        for (int r = rank; r < total_rows; ++r)
            if (M[(size_t)r * (size_t)cols + (size_t)col]) { pivot = r; break; }
        if (pivot < 0) continue;
        swap_rows(M, rhs, cols, rank, pivot);
        for (int r = 0; r < total_rows; ++r) {
            if (r == rank) continue;
            if (M[(size_t)r * (size_t)cols + (size_t)col]) {
                xor_rows(M + (size_t)r * (size_t)cols,
                         M + (size_t)rank * (size_t)cols,
                         cols, col);
                rhs[r] ^= rhs[rank];
            }
        }
        pivot_row[col] = rank;
        ++rank;
    }
    for (int r = rank; r < total_rows; ++r) if (rhs[r]) return 0;
    if (rank < cols) return 0;
    for (int c = 0; c < cols; ++c) {
        if (pivot_row[c] < 0) return 0;
        x[c] = rhs[pivot_row[c]];
    }
    return 1;
}

static void build_pruned_system(const SparseMatrix *H,
                                const uint8_t *syndrome,
                                const int *kept_indices,
                                int kept_count,
                                uint8_t *restrict M,
                                uint8_t *restrict rhs)
{
    memset(M, 0, (size_t)H->rows * (size_t)kept_count * sizeof(uint8_t));
    for (int ki = 0; ki < kept_count; ++ki) {
        int orig_col = kept_indices[ki];
        const SparseCol *sc = &H->col[orig_col];
        for (int k = 0; k < sc->c; ++k) {
            int r = sc->rows[k];
            M[(size_t)r * (size_t)kept_count + (size_t)ki] = 1;
        }
    }
    memcpy(rhs, syndrome, (size_t)H->rows * sizeof(uint8_t));
}

int alg3_prunebin(const SparseMatrix *H,
                  const uint8_t *syndrome,
                  uint8_t *e_out,
                  int threshold_v,
                  int max_weight,
                  int max_tries)
{
    static int seeded = 0;
    if (!seeded) { srand((unsigned)time(NULL)); seeded = 1; }
    if (!H || !syndrome || !e_out || H->rows <= 0 || H->cols <= 0 ||
        threshold_v < 0 || max_weight < 0 || max_tries <= 0) {
        fprintf(stderr, "[alg3_prunebin] Invalid arguments.\n");
        return 0;
    }

    int *intersection = (int *)malloc((size_t)H->cols * sizeof(int));
    Cand *candidates = (Cand *)malloc((size_t)H->cols * sizeof(Cand));
    int *selected = (int *)malloc((size_t)H->cols * sizeof(int));
    if (!intersection || !candidates || !selected) {
        free(intersection); free(candidates); free(selected);
        return 0;
    }
    compute_intersections_sparse(H, syndrome, intersection);

    int sure_count = 0, cand_count = 0;
    for (int j = 0; j < H->cols; ++j) {
        if (intersection[j] > threshold_v) selected[sure_count++] = j;
        else { candidates[cand_count].idx = j; candidates[cand_count].inter = intersection[j]; ++cand_count; }
    }
    const int orig_sure_count = sure_count;
    const int extra_margin = (H->rows > 100) ? 50 : (H->rows / 2);
    int target_cols = H->rows - extra_margin;
    if (target_cols < 1) target_cols = 1;
    /* Use only sure columns */
    if (target_cols > sure_count) target_cols = sure_count;

    printf("[alg3_prunebin] cols=%d, sure(>%d)=%d, candidates=%d, rows=%d, target=%d\n",
           H->cols, threshold_v, sure_count, cand_count, H->rows, target_cols);

    if (sure_count >= target_cols) {
        for (int i = target_cols; i < sure_count; ++i) {
            int j = rand() % (i + 1);
            if (j < target_cols) { int t = selected[j]; selected[j] = selected[i]; selected[i] = t; }
        }
        sure_count = target_cols;
    } else {
        qsort(candidates, (size_t)cand_count, sizeof(Cand), cmp_cand_desc);
        {
            int i = 0;
            while (i < cand_count) {
                int j = i;
                while (j < cand_count && candidates[j].inter == candidates[i].inter) ++j;
                for (int k = j - 1; k > i; --k) {
                    int r = i + rand() % (k - i + 1);
                    Cand tmp = candidates[k]; candidates[k] = candidates[r]; candidates[r] = tmp;
                }
                i = j;
            }
        }
        int need = target_cols - sure_count;
        if (need > cand_count) {
            fprintf(stderr, "[alg3_prunebin] Not enough candidates.\n");
            free(intersection); free(candidates); free(selected);
            return 0;
        }
        for (int i = 0; i < need; ++i) selected[sure_count + i] = candidates[i].idx;
        sure_count = target_cols;
    }

    const int kept_count = sure_count;
    uint8_t *M = (uint8_t *)malloc((size_t)H->rows * (size_t)kept_count * sizeof(uint8_t));
    uint8_t *rhs = (uint8_t *)malloc((size_t)H->rows * sizeof(uint8_t));
    uint8_t *x_reduced = (uint8_t *)malloc((size_t)kept_count * sizeof(uint8_t));
    if (!M || !rhs || !x_reduced) {
        free(intersection); free(candidates); free(selected);
        free(M); free(rhs); free(x_reduced);
        return 0;
    }

    int metal_ok = metal_bridge_init();
    if (!metal_ok) {
        fprintf(stderr, "[alg3_prunebin] Metal init failed.\n");
    }

    for (int attempt = 0; attempt < max_tries; ++attempt) {
        if (sure_count == target_cols && orig_sure_count > target_cols) {
            for (int i = target_cols; i < orig_sure_count; ++i) {
                int j = rand() % (i + 1);
                if (j < target_cols) { int t = selected[j]; selected[j] = selected[i]; selected[i] = t; }
            }
        }
        build_pruned_system(H, syndrome, selected, kept_count, M, rhs);
        int solved = 0;
        if (metal_ok) solved = alg2_metal_solve_overdetermined_unique_bin(M, rhs, H->rows, kept_count, x_reduced);
        if (!solved) continue;
        memset(e_out, 0, (size_t)H->cols * sizeof(uint8_t));
        for (int ki = 0; ki < kept_count; ++ki) e_out[selected[ki]] = x_reduced[ki];
        if (count_nonzero(e_out, H->cols) <= max_weight) {
            printf("[alg3_prunebin] Success attempt %d: kept=%d, weight=%d\n",
                   attempt+1, kept_count, count_nonzero(e_out, H->cols));
            free(intersection); free(candidates); free(selected);
            free(M); free(rhs); free(x_reduced);
            return attempt + 1;
        }
    }
    printf("[alg3_prunebin] Failed after %d attempts.\n", max_tries);
    free(intersection); free(candidates); free(selected);
    free(M); free(rhs); free(x_reduced);
    return 0;
}
