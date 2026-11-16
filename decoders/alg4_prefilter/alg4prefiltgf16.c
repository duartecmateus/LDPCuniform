/*
 * alg4prefiltgf16.c — Pruning decoder for GF(16) with consistency ranking.
 *
 * Ranks columns by how many incident rows agree on the most common error
 * value in {1, a, a^2, a^3} (single-bit-error patterns). True error
 * columns rank highly (~5/6 agreement); wrong columns rank low.
 * No hard cutoff — the top columns are selected and solved via LU.
 */

#include "alg4prefiltgf16.h"
#include "lu_solve.h"
#include "gf16.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

void sparse_matvec_gf16(const SparseMatrix *H, const uint8_t *e, uint8_t *syndrome)
{
    memset(syndrome, 0, (size_t)H->rows * sizeof(uint8_t));
    for (int j = 0; j < H->cols; ++j) {
        if (e[j] == 0) continue;
        uint8_t ej = e[j];
        const SparseCol *sc = &H->col[j];
        for (int k = 0; k < sc->c; ++k) {
            int r = sc->rows[k];
            uint8_t val = sc->values[k];
            if (val == 1) syndrome[r] ^= ej;
            else syndrome[r] ^= gf16_mul(val, ej);
        }
    }
}

/* ---- Ranking info per column ---- */
typedef struct {
    int idx;
    int inter;
    uint8_t best_val;
    int best_cnt;
} RankedCol;

/* Likely GF(16) error values: {1, a, a^2, a^3} = {1, 2, 4, 8} */
static const uint8_t likely_set[4] = {1, 2, 4, 8};

static void rank_all_columns(const SparseMatrix *H,
                              const uint8_t *syndrome,
                              RankedCol *ranks)
{
    /* Precompute which GF(16) elements map to the likely set */
    static int8_t likely_idx[16] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    static int init_done = 0;
    if (!init_done) {
        for (int i = 0; i < 4; i++) likely_idx[likely_set[i]] = (int8_t)i;
        init_done = 1;
    }

    for (int j = 0; j < H->cols; ++j) {
        const SparseCol *sc = &H->col[j];
        int votes[5] = {0, 0, 0, 0, 0};

        for (int k = 0; k < sc->c; ++k) {
            int r = sc->rows[k];
            if (syndrome[r] == 0) continue;
            uint8_t val = sc->values[k];
            uint8_t e_hat = gf16_mul(syndrome[r], gf16_inv(val));
            int idx = likely_idx[e_hat & 0xF];
            if (idx >= 0) votes[idx]++;
            else votes[4]++;
        }

        ranks[j].idx = j;
        ranks[j].inter = 0;
        for (int k = 0; k < sc->c; ++k)
            if (syndrome[sc->rows[k]] != 0) ++ranks[j].inter;

        ranks[j].best_val = 0;
        ranks[j].best_cnt = 0;
        for (int i = 0; i < 4; i++) {
            if (votes[i] > ranks[j].best_cnt) {
                ranks[j].best_cnt = votes[i];
                ranks[j].best_val = likely_set[i];
            }
        }
    }
}

static int cmp_rank(const void *a, const void *b)
{
    const RankedCol *ra = (const RankedCol *)a;
    const RankedCol *rb = (const RankedCol *)b;
    if (ra->best_cnt != rb->best_cnt)
        return (ra->best_cnt < rb->best_cnt) ? 1 : -1;
    if (ra->inter != rb->inter)
        return (ra->inter < rb->inter) ? 1 : -1;
    return (rand() % 3) - 1;
}

static void build_pruned_system(const SparseMatrix *H,
                                const uint8_t *syndrome,
                                const RankedCol *selected,
                                int kept_count,
                                uint8_t *restrict M,
                                uint8_t *restrict rhs)
{
    memset(M, 0, (size_t)H->rows * (size_t)kept_count * sizeof(uint8_t));
    for (int ki = 0; ki < kept_count; ++ki) {
        int orig_col = selected[ki].idx;
        const SparseCol *sc = &H->col[orig_col];
        for (int k = 0; k < sc->c; ++k) {
            int r = sc->rows[k];
            M[(size_t)r * (size_t)kept_count + (size_t)ki] = sc->values[k];
        }
    }
    memcpy(rhs, syndrome, (size_t)H->rows * sizeof(uint8_t));
}

int alg4prefilt_gf16(const SparseMatrix *H,
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
        fprintf(stderr, "[alg4prefilt_gf16] Invalid arguments.\n");
        return 0;
    }
    gf16_init();

    int n = H->cols, m = H->rows;

    RankedCol *ranks = (RankedCol *)malloc((size_t)n * sizeof(RankedCol));
    if (!ranks) return 0;

    rank_all_columns(H, syndrome, ranks);
    qsort(ranks, (size_t)n, sizeof(RankedCol), cmp_rank);

    int with_agree_4 = 0, with_agree_3 = 0, with_agree_2 = 0;
    for (int i = 0; i < n; i++) {
        if (ranks[i].best_cnt >= 4) with_agree_4++;
        if (ranks[i].best_cnt >= 3) with_agree_3++;
        if (ranks[i].best_cnt >= 2) with_agree_2++;
    }
    printf("[alg4prefilt_gf16] ranking: cnt>=4=%d, cnt>=3=%d, cnt>=2=%d, "
           "rows=%d, target=%d\n",
           with_agree_4, with_agree_3, with_agree_2, m,
           (m > 100) ? m - 50 : m / 2);

    const int extra_margin = (m > 100) ? 50 : (m / 2);
    int target_cols = m - extra_margin;
    if (target_cols < 1) target_cols = 1;
    if (target_cols > n) target_cols = n;

    uint8_t *M = (uint8_t *)malloc((size_t)m * (size_t)target_cols * sizeof(uint8_t));
    uint8_t *rhs = (uint8_t *)malloc((size_t)m * sizeof(uint8_t));
    uint8_t *x_reduced = (uint8_t *)malloc((size_t)target_cols * sizeof(uint8_t));
    if (!M || !rhs || !x_reduced) {
        free(ranks); free(M); free(rhs); free(x_reduced); return 0;
    }

    build_pruned_system(H, syndrome, ranks, target_cols, M, rhs);
    int solved = solve_lu_gf16(M, rhs, x_reduced, m, target_cols);

    if (solved) {
        memset(e_out, 0, (size_t)n * sizeof(uint8_t));
        for (int ki = 0; ki < target_cols; ++ki)
            e_out[ranks[ki].idx] = x_reduced[ki];
        if (count_nonzero(e_out, n) <= max_weight) {
            printf("[alg4prefilt_gf16] Success: kept=%d, weight=%d\n",
                   target_cols, count_nonzero(e_out, n));
            free(ranks); free(M); free(rhs); free(x_reduced);
            return 1;
        }
    }

    printf("[alg4prefilt_gf16] Failed (LU=%s, weight=%d vs max=%d).\n",
           solved ? "solved" : "singular",
           solved ? count_nonzero(e_out, n) : -1, max_weight);
    free(ranks); free(M); free(rhs); free(x_reduced);
    return 0;
}
