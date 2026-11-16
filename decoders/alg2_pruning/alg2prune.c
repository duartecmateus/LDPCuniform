#include "alg2prune.h"
#include "gf256.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

/*
 * GF(256) sparse matrix-vector multiply: syndrome = H * e.
 */
void sparse_matvec_gf256(const SparseMatrix *H, const uint8_t *e, uint8_t *syndrome)
{
    memset(syndrome, 0, (size_t)H->rows * sizeof(uint8_t));
    for (int j = 0; j < H->cols; ++j) {
        if (e[j] == 0) continue;
        uint8_t ej = e[j];
        const SparseCol *sc = &H->col[j];
        for (int k = 0; k < sc->c; ++k) {
            int r = sc->rows[k];
            uint8_t val = sc->values[k];
            if (val == 1) {
                syndrome[r] ^= ej;
            } else {
                syndrome[r] ^= gf256_mul(val, ej);
            }
        }
    }
}

/*
 * Compute, for each column, how many of its c non-zero rows intersect
 * the syndrome support.  Uses sparse column representation: O(c) per column.
 */
static void compute_intersections_sparse(const SparseMatrix *H,
                                         const uint8_t *syndrome,
                                         int *intersection_count)
{
    for (int j = 0; j < H->cols; ++j) {
        int count = 0;
        const SparseCol *sc = &H->col[j];
        for (int k = 0; k < sc->c; ++k) {
            if (syndrome[sc->rows[k]] != 0) {
                ++count;
            }
        }
        intersection_count[j] = count;
    }
}

/* Swap two rows using word-wise transfer (8 bytes at a time) */
static void swap_rows(uint8_t *restrict M, uint8_t *restrict rhs,
                       int cols, int r1, int r2)
{
    if (r1 == r2) return;
    uint8_t *restrict row1 = M + (size_t)r1 * (size_t)cols;
    uint8_t *restrict row2 = M + (size_t)r2 * (size_t)cols;
    int j = 0;
    for (; j + 8 <= cols; j += 8) {
        uint64_t t = *(uint64_t *)(row1 + j);
        *(uint64_t *)(row1 + j) = *(uint64_t *)(row2 + j);
        *(uint64_t *)(row2 + j) = t;
    }
    for (; j < cols; ++j) {
        uint8_t t = row1[j];
        row1[j] = row2[j];
        row2[j] = t;
    }
    uint8_t tb = rhs[r1];
    rhs[r1] = rhs[r2];
    rhs[r2] = tb;
}

/*
 * Overdetermined Gauss-Jordan for GF(256).
 * M: total_rows x cols dense matrix, total_rows >= cols.
 * pivot_row is pre-allocated workspace of size cols.
 * Returns 1 on unique solution, 0 otherwise.
 */
static int solve_overdetermined_gf256(uint8_t *restrict M,
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
        for (int r = rank; r < total_rows; ++r) {
            if (M[(size_t)r * (size_t)cols + (size_t)col] != 0) {
                pivot = r;
                break;
            }
        }
        if (pivot < 0) continue;

        swap_rows(M, rhs, cols, rank, pivot);

        uint8_t *restrict pr = M + (size_t)rank * (size_t)cols;
        uint8_t pv = pr[col];
        if (pv == 0) return 0;

        if (pv != 1) {
            uint8_t inv = gf256_inv(pv);
            for (int j = col; j < cols; ++j)
                pr[j] = gf256_mul(pr[j], inv);
            rhs[rank] = gf256_mul(rhs[rank], inv);
        }

        for (int r = 0; r < total_rows; ++r) {
            if (r == rank) continue;
            uint8_t *restrict row = M + (size_t)r * (size_t)cols;
            uint8_t factor = row[col];
            if (factor == 0) continue;
            for (int j = col; j < cols; ++j)
                row[j] ^= gf256_mul(factor, pr[j]);
            rhs[r] ^= gf256_mul(factor, rhs[rank]);
        }

        pivot_row[col] = rank;
        ++rank;
    }

    /* Consistency check: only scan rows beyond rank (guaranteed all-zero rows) */
    for (int r = rank; r < total_rows; ++r) {
        if (rhs[r] != 0) return 0;
    }

    if (rank < cols) return 0;

    for (int c = 0; c < cols; ++c) {
        int pr_idx = pivot_row[c];
        if (pr_idx < 0) return 0;
        x[c] = rhs[pr_idx];
    }

    return 1;
}

/*
 * Build the pruned dense system from the kept columns.
 * M: rows x kept_count dense matrix (filled from sparse columns)
 * rhs: syndrome vector (copied from input)
 */
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
            M[(size_t)r * (size_t)kept_count + (size_t)ki] = sc->values[k];
        }
    }

    memcpy(rhs, syndrome, (size_t)H->rows * sizeof(uint8_t));
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

int alg2_prune(const SparseMatrix *H,
               const uint8_t *syndrome,
               uint8_t *e_out,
               int threshold_v,
               int max_weight,
               int max_tries)
{
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned)time(NULL));
        seeded = 1;
    }

    if (!H || !syndrome || !e_out || H->rows <= 0 || H->cols <= 0 ||
        threshold_v < 0 || max_weight < 0 || max_tries <= 0) {
        fprintf(stderr, "[alg2_prune] Invalid arguments.\n");
        return 0;
    }

    gf256_init();

    int *intersection = (int *)malloc((size_t)H->cols * sizeof(int));
    int *kept = (int *)malloc((size_t)H->cols * sizeof(int));
    Cand *candidates = (Cand *)malloc((size_t)H->cols * sizeof(Cand));
    int *selected = (int *)malloc((size_t)H->cols * sizeof(int));
    if (!intersection || !kept || !candidates || !selected) {
        fprintf(stderr, "[alg2_prune] Allocation failed.\n");
        free(intersection); free(kept); free(candidates); free(selected);
        return 0;
    }

    /* Step 1: sparse intersections */
    compute_intersections_sparse(H, syndrome, intersection);

    /* Step 2: separate into sure (>v) and candidates (<=v) */
    int sure_count = 0;
    int cand_count = 0;
    for (int j = 0; j < H->cols; ++j) {
        if (intersection[j] > threshold_v) {
            selected[sure_count++] = j;
        } else {
            candidates[cand_count].idx = j;
            candidates[cand_count].inter = intersection[j];
            ++cand_count;
        }
    }

    const int orig_sure_count = sure_count;

    /* Overdetermined system: target_cols = rows - margin */
    const int extra_margin = (H->rows > 100) ? 50 : (H->rows / 2);
    int target_cols = H->rows - extra_margin;
    if (target_cols < 1) target_cols = 1;

    printf("[alg2_prune] cols=%d, sure (inter>%d)=%d, candidates=%d, rows=%d, target=%d\n",
           H->cols, threshold_v, sure_count, cand_count, H->rows, target_cols);

    if (sure_count >= target_cols) {
        for (int i = target_cols; i < sure_count; ++i) {
            int j = rand() % (i + 1);
            if (j < target_cols) {
                int tmp = selected[j];
                selected[j] = selected[i];
                selected[i] = tmp;
            }
        }
        sure_count = target_cols;
    } else {
        /* qsort candidates by intersection descending */
        qsort(candidates, (size_t)cand_count, sizeof(Cand), cmp_cand_desc);

        /* Shuffle within equal-intersection buckets */
        {
            int i = 0;
            while (i < cand_count) {
                int j = i;
                while (j < cand_count && candidates[j].inter == candidates[i].inter) ++j;
                for (int k = j - 1; k > i; --k) {
                    int r = i + rand() % (k - i + 1);
                    Cand tmp = candidates[k];
                    candidates[k] = candidates[r];
                    candidates[r] = tmp;
                }
                i = j;
            }
        }

        int need = target_cols - sure_count;
        if (need > cand_count) {
            fprintf(stderr, "[alg2_prune] Not enough candidates: need %d more from %d.\n",
                    need, cand_count);
            free(intersection); free(kept); free(candidates); free(selected);
            return 0;
        }
        for (int i = 0; i < need; ++i)
            selected[sure_count + i] = candidates[i].idx;
        sure_count = target_cols;
    }

    const int kept_count = sure_count;
    printf("[alg2_prune] selected %d columns for a %d x %d system\n",
           kept_count, H->rows, kept_count);

    /* Pre-allocate all buffers once and reuse across attempts */
    uint8_t *M = (uint8_t *)malloc((size_t)H->rows * (size_t)kept_count * sizeof(uint8_t));
    uint8_t *rhs = (uint8_t *)malloc((size_t)H->rows * sizeof(uint8_t));
    uint8_t *x_reduced = (uint8_t *)malloc((size_t)kept_count * sizeof(uint8_t));
    int *pivot_row = (int *)malloc((size_t)kept_count * sizeof(int));
    if (!M || !rhs || !x_reduced || !pivot_row) {
        fprintf(stderr, "[alg2_prune] Allocation failed.\n");
        free(intersection); free(kept); free(candidates); free(selected);
        free(M); free(rhs); free(x_reduced); free(pivot_row);
        return 0;
    }

    for (int attempt = 0; attempt < max_tries; ++attempt) {
        /* Reshuffle sure-column selection each attempt if we have extras */
        if (sure_count == target_cols && orig_sure_count > target_cols) {
            for (int i = target_cols; i < orig_sure_count; ++i) {
                int j = rand() % (i + 1);
                if (j < target_cols) {
                    int tmp = selected[j];
                    selected[j] = selected[i];
                    selected[i] = tmp;
                }
            }
        }

        build_pruned_system(H, syndrome, selected, kept_count, M, rhs);

        if (!solve_overdetermined_gf256(M, rhs, H->rows, kept_count,
                                         x_reduced, pivot_row)) {
            continue;
        }

        /* Map back to full error vector */
        memset(e_out, 0, (size_t)H->cols * sizeof(uint8_t));
        for (int ki = 0; ki < kept_count; ++ki) {
            e_out[selected[ki]] = x_reduced[ki];
        }

        if (count_nonzero(e_out, H->cols) <= max_weight) {
            printf("[alg2_prune] Success on attempt %d: kept=%d, weight=%d\n",
                   attempt + 1, kept_count, count_nonzero(e_out, H->cols));
            free(intersection); free(kept); free(candidates); free(selected);
            free(M); free(rhs); free(x_reduced); free(pivot_row);
            return attempt + 1;
        }
    }

    printf("[alg2_prune] Failed after %d attempts (kept=%d).\n", max_tries, kept_count);
    free(intersection); free(kept); free(candidates); free(selected);
    free(M); free(rhs); free(x_reduced); free(pivot_row);
    return 0;
}
