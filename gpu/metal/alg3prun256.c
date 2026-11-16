#include "alg3prun256.h"
#include "metal_bridge.h"
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
 * Compute, for each column, intersection count with syndrome support.
 * Uses sparse column representation: O(c) per column.
 * Optionally uses Metal GPU for the dense computation if matrix is large enough.
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
    if (ca->inter != cb->inter)
        return (ca->inter < cb->inter) - (ca->inter > cb->inter);
    return (rand() % 3) - 1;
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

int alg3_prune256(const SparseMatrix *H,
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
        fprintf(stderr, "[alg3_prune256] Invalid arguments.\n");
        return 0;
    }

    gf256_init();

    int *intersection = (int *)malloc((size_t)H->cols * sizeof(int));
    int *kept = (int *)malloc((size_t)H->cols * sizeof(int));
    Cand *candidates = (Cand *)malloc((size_t)H->cols * sizeof(Cand));
    int *selected = (int *)malloc((size_t)H->cols * sizeof(int));
    if (!intersection || !kept || !candidates || !selected) {
        fprintf(stderr, "[alg3_prune256] Allocation failed.\n");
        free(intersection); free(kept); free(candidates); free(selected);
        return 0;
    }

    /* Step 1: sparse intersections (CPU — O(n*c), fast) */
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
    /* Use only sure columns — avoid adding random candidates */
    if (target_cols > sure_count) target_cols = sure_count;

    printf("[alg3_prune256] cols=%d, sure (inter>%d)=%d, candidates=%d, rows=%d, target=%d\n",
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
            fprintf(stderr, "[alg3_prune256] Not enough candidates: need %d more from %d.\n",
                    need, cand_count);
            free(intersection); free(kept); free(candidates); free(selected);
            return 0;
        }
        for (int i = 0; i < need; ++i)
            selected[sure_count + i] = candidates[i].idx;
        sure_count = target_cols;
    }

    const int kept_count = sure_count;
    printf("[alg3_prune256] selected %d columns for a %d x %d system\n",
           kept_count, H->rows, kept_count);

    /* Pre-allocate buffers for the pruned dense system */
    uint8_t *M = (uint8_t *)malloc((size_t)H->rows * (size_t)kept_count * sizeof(uint8_t));
    uint8_t *rhs = (uint8_t *)malloc((size_t)H->rows * sizeof(uint8_t));
    uint8_t *x_reduced = (uint8_t *)malloc((size_t)kept_count * sizeof(uint8_t));
    if (!M || !rhs || !x_reduced) {
        fprintf(stderr, "[alg3_prune256] Allocation failed.\n");
        free(intersection); free(kept); free(candidates); free(selected);
        free(M); free(rhs); free(x_reduced);
        return 0;
    }

    /* Initialize Metal bridge once */
    int metal_ok = metal_bridge_init();
    if (!metal_ok) {
        fprintf(stderr, "[alg3_prune256] Metal init failed; falling back would require CPU path.\n");
        /* The algorithm will fail gracefully since solve_overdetermined_unique returns 0 on Metal failure */
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

        int solved = 0;
        if (metal_ok) {
            /* GPU accelerated solve — uses Metal Gauss-Jordan elimination */
            solved = alg2_metal_solve_overdetermined_unique(M, rhs, H->rows, kept_count, x_reduced);
        }
        /* Note: no CPU fallback here; Metal is required. If metal_ok is false,
           solved stays 0 and we continue to the next attempt or fail. */

        if (!solved) {
            continue;
        }

        /* Map back to full error vector */
        memset(e_out, 0, (size_t)H->cols * sizeof(uint8_t));
        for (int ki = 0; ki < kept_count; ++ki) {
            e_out[selected[ki]] = x_reduced[ki];
        }

        if (count_nonzero(e_out, H->cols) <= max_weight) {
            printf("[alg3_prune256] Success on attempt %d: kept=%d, weight=%d\n",
                   attempt + 1, kept_count, count_nonzero(e_out, H->cols));
            free(intersection); free(kept); free(candidates); free(selected);
            free(M); free(rhs); free(x_reduced);
            return attempt + 1;
        }
    }

    printf("[alg3_prune256] Failed after %d attempts (kept=%d).\n", max_tries, kept_count);
    free(intersection); free(kept); free(candidates); free(selected);
    free(M); free(rhs); free(x_reduced);
    return 0;
}
