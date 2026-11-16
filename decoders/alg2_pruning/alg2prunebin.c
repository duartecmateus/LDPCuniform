#include "alg2prunebin.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

/*
 * Compute, for each column, how many of its c non-zero rows intersect
 * the syndrome support.  Binary version — values are always 1.
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

typedef struct { int idx; int inter; } Cand;

static int cmp_cand_desc(const void *a, const void *b)
{
    const Cand *ca = (const Cand *)a;
    const Cand *cb = (const Cand *)b;
    /* Sort by intersection descending */
    if (ca->inter != cb->inter)
        return (ca->inter < cb->inter) - (ca->inter > cb->inter);
    /* Random tie-break within equal buckets */
    return (rand() % 3) - 1;
}

/* XOR a pivot row into another row, processing 8 bytes at a time via uint64_t.
   Both row and pr are uint8_t arrays of length cols. */
static inline void xor_rows(uint8_t *restrict row,
                             const uint8_t *restrict pr,
                             int cols, int start_col)
{
    int j = start_col;
    /* Aligned word-wise XOR */
    for (; j + 8 <= cols; j += 8) {
        *(uint64_t *)(row + j) ^= *(const uint64_t *)(pr + j);
    }
    /* Remaining bytes */
    for (; j < cols; ++j)
        row[j] ^= pr[j];
}

static void swap_rows(uint8_t *M, uint8_t *rhs, int cols, int r1, int r2)
{
    if (r1 == r2) return;
    /* Swap in 8-byte chunks */
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
 * Overdetermined Gauss-Jordan for GF(2) (binary/XOR).
 * M: total_rows x cols dense matrix (uint8_t, values 0 or 1).
 * Returns 1 on unique solution, 0 otherwise.
 * pivot_row is a pre-allocated workspace of size cols.
 */
static int solve_overdetermined_binary(uint8_t *M, uint8_t *rhs,
                                        int total_rows, int cols,
                                        uint8_t *x,
                                        int *pivot_row)
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

        uint8_t *pr = M + (size_t)rank * (size_t)cols;
        for (int r = 0; r < total_rows; ++r) {
            if (r == rank) continue;
            uint8_t *row = M + (size_t)r * (size_t)cols;
            if (row[col] == 0) continue;
            /* XOR the pivot row into this row (8 bytes at a time) */
            xor_rows(row, pr, cols, col);
            rhs[r] ^= rhs[rank];
        }

        pivot_row[col] = rank;
        ++rank;
    }

    /* Consistency check — only scan rows beyond rank (all-zero rows) */
    for (int r = rank; r < total_rows; ++r) {
        if (rhs[r] != 0) return 0;
    }

    if (rank < cols) return 0;

    for (int c = 0; c < cols; ++c) {
        int pr = pivot_row[c];
        if (pr < 0) return 0;
        x[c] = rhs[pr];
    }

    return 1;
}

static void build_pruned_system(const SparseMatrix *H,
                                const uint8_t *syndrome,
                                const int *kept_indices,
                                int kept_count,
                                uint8_t *M,
                                uint8_t *rhs)
{
    memset(M, 0, (size_t)H->rows * (size_t)kept_count * sizeof(uint8_t));

    for (int ki = 0; ki < kept_count; ++ki) {
        int orig_col = kept_indices[ki];
        const SparseCol *sc = &H->col[orig_col];
        for (int k = 0; k < sc->c; ++k) {
            int r = sc->rows[k];
            /* Binary: value is always 1 */
            M[(size_t)r * (size_t)kept_count + (size_t)ki] = 1;
        }
    }

    memcpy(rhs, syndrome, (size_t)H->rows * sizeof(uint8_t));
}

int alg2_prunebin(const SparseMatrix *H,
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
        fprintf(stderr, "[alg2_prunebin] Invalid arguments.\n");
        return 0;
    }

    if (H->field != 0) {
        fprintf(stderr, "[alg2_prunebin] Matrix is not binary (field=%d).\n", H->field);
        return 0;
    }

    int *intersection = (int *)malloc((size_t)H->cols * sizeof(int));
    /* For sure+candidate selection we need up to H->cols indices */
    int *selected = (int *)malloc((size_t)H->cols * sizeof(int));
    if (!intersection || !selected) {
        fprintf(stderr, "[alg2_prunebin] Allocation failed.\n");
        free(intersection); free(selected);
        return 0;
    }

    /* Step 1: sparse intersections */
    compute_intersections_sparse(H, syndrome, intersection);

    /* Step 2: separate into sure (intersection > v) and candidates (intersection <= v) */
    Cand *candidates = (Cand *)malloc((size_t)H->cols * sizeof(Cand));
    if (!candidates) {
        free(intersection); free(selected);
        return 0;
    }

    int sure_count = 0;
    int cand_count = 0;
    for (int j = 0; j < H->cols; ++j) {
        if (intersection[j] > threshold_v) {
            selected[sure_count++] = j;   /* sure columns go first */
        } else {
            candidates[cand_count].idx = j;
            candidates[cand_count].inter = intersection[j];
            ++cand_count;
        }
    }

    printf("[alg2_prunebin] cols=%d, sure (inter>%d)=%d, candidates=%d, rows=%d\n",
           H->cols, threshold_v, sure_count, cand_count, H->rows);

    /* Remember original sure count for reshuffling logic */
    const int orig_sure_count = sure_count;

    /* Build an overdetermined system: target_cols < H->rows for solver robustness.
     * We leave extra_margin extra rows beyond the square minimum (at least 50). */
    const int extra_margin = (H->rows > 100) ? 50 : (H->rows / 2);
    int target_cols = H->rows - extra_margin;
    if (target_cols < 1) target_cols = 1;

    if (sure_count >= target_cols) {
        /* Too many sure columns: subsample randomly among sure-set columns.
         * Shuffle the first target_cols entries of selected[].
         * The shuffle is deterministic on each attempt but differs across attempts
         * because we re-seed. */
        for (int i = target_cols; i < sure_count; ++i) {
            int j = rand() % (i + 1);
            if (j < target_cols) {
                int tmp = selected[j];
                selected[j] = selected[i];
                selected[i] = tmp;
            }
        }
        /* Trim to target_cols */
        sure_count = target_cols;
    } else {
        /* Sort candidates by intersection descending using qsort (O(N log N)) */
        qsort(candidates, (size_t)cand_count, sizeof(Cand), cmp_cand_desc);

        /* Shuffle within equal-intersection buckets for randomness on ties */
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
            fprintf(stderr,
                    "[alg2_prunebin] Not enough candidates: need %d more from %d candidates.\n",
                    need, cand_count);
            free(intersection); free(selected); free(candidates);
            return 0;
        }

        /* Append top candidates */
        for (int i = 0; i < need; ++i) {
            selected[sure_count + i] = candidates[i].idx;
        }

        sure_count = target_cols;  /* now we have exactly target_cols */
    }

    const int kept_count = sure_count;
    printf("[alg2_prunebin] selected %d columns for a %d x %d system\n",
           kept_count, H->rows, kept_count);

    /* Pre-allocate all buffers once and reuse across attempts */
    uint8_t *M = (uint8_t *)malloc((size_t)H->rows * (size_t)kept_count * sizeof(uint8_t));
    uint8_t *rhs = (uint8_t *)malloc((size_t)H->rows * sizeof(uint8_t));
    uint8_t *x_reduced = (uint8_t *)malloc((size_t)kept_count * sizeof(uint8_t));
    int *pivot_row = (int *)malloc((size_t)kept_count * sizeof(int));
    if (!M || !rhs || !x_reduced || !pivot_row) {
        fprintf(stderr, "[alg2_prunebin] Allocation failed.\n");
        free(intersection); free(selected); free(candidates);
        free(M); free(rhs); free(x_reduced); free(pivot_row);
        return 0;
    }

    for (int attempt = 0; attempt < max_tries; ++attempt) {
        /* Reshuffle sure-column selection if we have extras */
        if (orig_sure_count > target_cols) {
            for (int i = target_cols; i < orig_sure_count; ++i) {
                int j = rand() % (i + 1);
                if (j < target_cols) {
                    int tmp = selected[j];
                    selected[j] = selected[i];
                    selected[i] = tmp;
                }
            }
        }

        /* Also try different candidate subsets on each attempt.
           Rotate the candidate list by a random offset so different
           portions of the top candidates get picked. */
        if (cand_count > 0) {
            int cand_start = (orig_sure_count < target_cols) ? orig_sure_count : target_cols;
            int cand_end = target_cols;
            int cand_in_system = cand_end - cand_start;
            if (cand_in_system > 0 && cand_in_system < cand_count) {
                /* Rotate by a random offset to try different subsets */
                int rot = rand() % cand_count;
                if (rot > 0) {
                    for (int i = 0; i < rot && i < cand_count; ++i) {
                        Cand tmp = candidates[0];
                        for (int k = 0; k + 1 < cand_count; ++k)
                            candidates[k] = candidates[k + 1];
                        candidates[cand_count - 1] = tmp;
                    }
                }
                /* Re-fill candidate portion of selected[] */
                for (int i = 0; i < cand_in_system; ++i)
                    selected[cand_start + i] = candidates[i].idx;
            }
        }

        build_pruned_system(H, syndrome, selected, kept_count, M, rhs);

        if (!solve_overdetermined_binary(M, rhs, H->rows, kept_count,
                                          x_reduced, pivot_row)) {
            continue;
        }

        /* Map back: e_out[selected[ki]] = x_reduced[ki] */
        memset(e_out, 0, (size_t)H->cols * sizeof(uint8_t));
        for (int ki = 0; ki < kept_count; ++ki) {
            e_out[selected[ki]] = x_reduced[ki];
        }

        if (count_nonzero(e_out, H->cols) <= max_weight) {
            printf("[alg2_prunebin] Success on attempt %d: kept=%d, weight=%d\n",
                   attempt + 1, kept_count, count_nonzero(e_out, H->cols));
            free(intersection); free(selected); free(candidates);
            free(M); free(rhs); free(x_reduced); free(pivot_row);
            return attempt + 1;
        }
    }

    printf("[alg2_prunebin] Failed after %d attempts (kept=%d).\n", max_tries, kept_count);
    free(intersection); free(selected); free(candidates);
    free(M); free(rhs); free(x_reduced); free(pivot_row);
    return 0;
}
