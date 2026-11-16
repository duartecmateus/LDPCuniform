#include "alg2augmentbin.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

/* Sparse intersection: O(c) per column */
static void compute_intersections_sparse(const SparseMatrix *H,
                                         const uint8_t *syndrome,
                                         int *intersection_count)
{
    for (int j = 0; j < H->cols; ++j) {
        int count = 0;
        const SparseCol *sc = &H->col[j];
        for (int k = 0; k < sc->c; ++k) {
            if (syndrome[sc->rows[k]] != 0)
                ++count;
        }
        intersection_count[j] = count;
    }
}

/* Word-wise XOR of a pivot row into another row (uint64_t, 8 bytes at a time) */
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

/* Word-wise row swap */
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
 * Overdetermined Gauss-Jordan for GF(2), word-wise XOR.
 * M: total_rows x cols (total_rows >= cols).
 * pivot_row: pre-allocated workspace of size cols.
 * Returns 1 on unique solution, 0 otherwise.
 */
static int solve_binary(uint8_t *restrict M, uint8_t *restrict rhs,
                          int total_rows, int cols,
                          uint8_t *restrict x, int *restrict pivot_row)
{
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
        for (int r = 0; r < total_rows; ++r) {
            if (r == rank) continue;
            uint8_t *restrict row = M + (size_t)r * (size_t)cols;
            if (row[col] == 0) continue;
            xor_rows(row, pr, cols, col);
            rhs[r] ^= rhs[rank];
        }

        pivot_row[col] = rank;
        ++rank;
    }

    /* Consistency: check rows beyond rank (all-zero rows) */
    for (int r = rank; r < total_rows; ++r)
        if (rhs[r] != 0) return 0;

    if (rank < cols) return 0;

    for (int c = 0; c < cols; ++c) {
        int pr = pivot_row[c];
        if (pr < 0) return 0;
        x[c] = rhs[pr];
    }
    return 1;
}

/*
 * Build the augmented system.
 *
 * Original: A (rows x cols) dense, from sparse columns.
 * Added: unit rows (one 1 at selected[k], rest 0) with RHS = 0.
 *
 * Total rows: rows + added_rows
 * Cols: cols
 *
 * We construct by:
 *   1. Copy H rows into M (built from sparse columns)
 *   2. Zero out added_rows × cols region
 *   3. Set M[rows + k][selected[k]] = 1 for each added row
 *   4. Copy syndrome into first `rows` entries of rhs, 0 for the rest
 */
static void build_augmented_system(const SparseMatrix *H,
                                    const uint8_t *syndrome,
                                    const int *selected,
                                    int added_rows,
                                    uint8_t *restrict M,
                                    uint8_t *restrict rhs)
{
    const int rows = H->rows;
    const int cols = H->cols;

    /* Initialize M to zero */
    memset(M, 0, (size_t)(rows + added_rows) * (size_t)cols * sizeof(uint8_t));

    /* Fill from sparse columns */
    for (int j = 0; j < cols; ++j) {
        const SparseCol *sc = &H->col[j];
        for (int k = 0; k < sc->c; ++k) {
            int r = sc->rows[k];
            /* Binary: value is always 1 */
            M[(size_t)r * (size_t)cols + (size_t)j] = 1;
        }
    }

    /* Add unit rows for selected columns */
    for (int k = 0; k < added_rows; ++k) {
        int out_row = rows + k;
        int basis_col = selected[k];
        M[(size_t)out_row * (size_t)cols + (size_t)basis_col] = 1;
    }

    /* RHS: syndrome + zeros */
    memcpy(rhs, syndrome, (size_t)rows * sizeof(uint8_t));
    if (added_rows > 0)
        memset(rhs + rows, 0, (size_t)added_rows * sizeof(uint8_t));
}

/* Candidate pair for sorting */
typedef struct { int idx; int inter; } Cand;

/* Sort by intersection ascending (lower = better for augmentation) */
static int cmp_cand_asc(const void *a, const void *b)
{
    const Cand *ca = (const Cand *)a;
    const Cand *cb = (const Cand *)b;
    if (ca->inter != cb->inter)
        return (ca->inter > cb->inter) - (ca->inter < cb->inter);
    return (rand() % 3) - 1;
}

int alg2_augment_bin(const SparseMatrix *H,
                     const uint8_t *syndrome,
                     uint8_t *e_out,
                     int max_intersection,
                     int extra_added_rows,
                     int max_weight,
                     int max_tries)
{
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned)time(NULL));
        seeded = 1;
    }

    if (!H || !syndrome || !e_out || H->rows <= 0 || H->cols <= 0 ||
        max_intersection < 0 || extra_added_rows < 0 ||
        max_weight < 0 || max_tries <= 0) {
        fprintf(stderr, "[alg2_augment_bin] Invalid arguments.\n");
        return 0;
    }

    if (H->field != 0) {
        fprintf(stderr, "[alg2_augment_bin] Matrix is not binary (field=%d).\n", H->field);
        return 0;
    }

    int rows = H->rows;
    int cols = H->cols;

    int *intersection = (int *)malloc((size_t)cols * sizeof(int));
    int *selected = (int *)malloc((size_t)cols * sizeof(int));
    Cand *candidates = (Cand *)malloc((size_t)cols * sizeof(Cand));
    if (!intersection || !selected || !candidates) {
        fprintf(stderr, "[alg2_augment_bin] Allocation failed.\n");
        free(intersection); free(selected); free(candidates);
        return 0;
    }

    /* Step 1: sparse intersections */
    compute_intersections_sparse(H, syndrome, intersection);

    /* Step 2: collect candidates (intersection <= max_intersection) */
    int cand_count = 0;
    for (int j = 0; j < cols; ++j) {
        if (intersection[j] <= max_intersection) {
            candidates[cand_count].idx = j;
            candidates[cand_count].inter = intersection[j];
            ++cand_count;
        }
    }

    /* Sort by intersection ascending (lowest = safest to zero-constrain) */
    qsort(candidates, (size_t)cand_count, sizeof(Cand), cmp_cand_asc);

    /* Number of added constraints needed */
    const int minimum_added = cols - rows;
    const int added_rows = minimum_added + extra_added_rows;
    const int total_rows = rows + added_rows;

    printf("[alg2_augment_bin] cols=%d, rows=%d, candidates(inter<=%d)=%d, "
           "added=%d(min=%d+extra=%d), system=%dx%d\n",
           cols, rows, max_intersection, cand_count,
           added_rows, minimum_added, extra_added_rows,
           total_rows, cols);

    if (added_rows > cand_count) {
        fprintf(stderr, "[alg2_augment_bin] Not enough candidates: "
                "need %d added rows, have %d candidates.\n",
                added_rows, cand_count);
        free(intersection); free(selected); free(candidates);
        return 0;
    }

    /* Pre-allocate buffers */
    uint8_t *M = (uint8_t *)malloc((size_t)total_rows * (size_t)cols * sizeof(uint8_t));
    uint8_t *rhs = (uint8_t *)malloc((size_t)total_rows * sizeof(uint8_t));
    uint8_t *x_full = (uint8_t *)malloc((size_t)cols * sizeof(uint8_t));
    int *pivot_row = (int *)malloc((size_t)cols * sizeof(int));
    if (!M || !rhs || !x_full || !pivot_row) {
        fprintf(stderr, "[alg2_augment_bin] Allocation failed.\n");
        free(intersection); free(selected); free(candidates);
        free(M); free(rhs); free(x_full); free(pivot_row);
        return 0;
    }

    for (int attempt = 0; attempt < max_tries; ++attempt) {
        /* On each attempt, rotate candidates to try different subsets
           (preserves low-intersection priority while exploring variety) */
        if (cand_count > 0 && added_rows < cand_count && attempt > 0) {
            int rot = 1 + rand() % (cand_count / 2);
            for (int i = 0; i < rot; ++i) {
                Cand tmp = candidates[0];
                for (int k = 0; k + 1 < cand_count; ++k)
                    candidates[k] = candidates[k + 1];
                candidates[cand_count - 1] = tmp;
            }
        }

        /* Select the first `added_rows` candidates */
        for (int i = 0; i < added_rows; ++i)
            selected[i] = candidates[i].idx;

        /* Build and solve */
        build_augmented_system(H, syndrome, selected, added_rows, M, rhs);

        if (!solve_binary(M, rhs, total_rows, cols, x_full, pivot_row))
            continue;

        /* Check weight */
        int w = count_nonzero(x_full, cols);
        if (w <= max_weight) {
            /* Verify: syndrome matches */
            uint8_t *s_check = (uint8_t *)malloc((size_t)rows * sizeof(uint8_t));
            sparse_matvec(H, x_full, s_check);
            int match = compare_vectors(syndrome, s_check, rows);
            free(s_check);

            if (match) {
                memcpy(e_out, x_full, (size_t)cols * sizeof(uint8_t));
                printf("[alg2_augment_bin] Success attempt %d: weight=%d\n",
                       attempt + 1, w);
                free(intersection); free(selected); free(candidates);
                free(M); free(rhs); free(x_full); free(pivot_row);
                return attempt + 1;
            }
        }
    }

    printf("[alg2_augment_bin] Failed after %d attempts (added=%d, candidates=%d).\n",
           max_tries, added_rows, cand_count);
    free(intersection); free(selected); free(candidates);
    free(M); free(rhs); free(x_full); free(pivot_row);
    return 0;
}
