/*
 * alg4prefiltgf4.c — Pruning decoder for GF(4) with successive cancellation.
 *
 * 1. Rank columns by consistency (how many incident rows agree on error value)
 * 2. Pin the most confident columns: those with best_cnt >= pin_thresh
 *    For each pinned column j with agreed value e_j:
 *      - Add equation: 1·e_j = e_j (virtual row)
 *      - Subtract H[r,j]·e_j from the syndrome of all incident rows
 * 3. Build the pruned system from remaining (unpinned) columns
 * 4. Solve the smaller, cleaner system via LU
 *
 * This mirrors BP's peeling decoder: confident variables are decided first,
 * their contribution is removed, and the residual problem is simpler.
 *
 * threshold_v encoding: (pin_thresh << 8) | max_pin_violations
 *   Default: pin_thresh = 5 (0.46% FP rate), max_viol = 2
 */

#include "alg4prefiltgf4.h"
#include "lu_solve.h"
#include "gf4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

void sparse_matvec_gf4(const SparseMatrix *H, const uint8_t *e, uint8_t *syndrome)
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
            else syndrome[r] ^= gf4_mul(val, ej);
        }
    }
}

/* ---- Per-column ranking info ---- */
typedef struct {
    int idx;
    int inter;        /* incident rows with non-zero syndrome */
    uint8_t best_val; /* most agreed error value */
    int best_cnt;     /* # of rows agreeing */
} ColRank;

static void rank_all(const SparseMatrix *H, const uint8_t *syndrome, ColRank *r)
{
    for (int j = 0; j < H->cols; ++j) {
        const SparseCol *sc = &H->col[j];
        int votes[4] = {0,0,0,0};
        for (int k = 0; k < sc->c; ++k) {
            int ri = sc->rows[k];
            if (syndrome[ri] == 0) continue;
            votes[gf4_mul(syndrome[ri], gf4_inv(sc->values[k])) & 3]++;
        }
        r[j].idx = j;
        r[j].inter = 0;
        for (int k = 0; k < sc->c; ++k)
            if (syndrome[sc->rows[k]] != 0) r[j].inter++;
        r[j].best_val = 0;
        r[j].best_cnt = 0;
        for (int i = 1; i < 4; i++)
            if (votes[i] > r[j].best_cnt) {
                r[j].best_cnt = votes[i];
                r[j].best_val = (uint8_t)i;
            }
    }
}

static int cmp_rank(const void *a, const void *b)
{
    const ColRank *ra = (const ColRank *)a, *rb = (const ColRank *)b;
    if (ra->best_cnt != rb->best_cnt)
        return (ra->best_cnt < rb->best_cnt) ? 1 : -1;
    if (ra->inter != rb->inter)
        return (ra->inter < rb->inter) ? 1 : -1;
    return (rand() % 3) - 1;
}

int alg4prefilt_gf4(const SparseMatrix *H,
                    const uint8_t *syndrome,
                    uint8_t *e_out,
                    int threshold_v,
                    int max_weight,
                    int max_tries)
{
    static int seeded = 0;
    if (!seeded) { srand((unsigned)time(NULL)); seeded = 1; }
    if (!H || !syndrome || !e_out || H->rows <= 0 || H->cols <= 0) return 0;
    gf4_init();

    /* Decode params */
    int pin_thresh = (threshold_v >> 8) & 0xFF;
    if (pin_thresh == 0) pin_thresh = 5;
    int max_viol = threshold_v & 0xFF;
    if (max_viol == 0) max_viol = 2;

    int n = H->cols, m = H->rows;

    ColRank *ranks = (ColRank *)malloc((size_t)n * sizeof(ColRank));
    int *pin_to_sel = (int *)malloc((size_t)n * sizeof(int));  /* pinned col idx → selected col idx */
    uint8_t *pin_val = (uint8_t *)malloc((size_t)n);
    int *sel_to_orig = (int *)malloc((size_t)n * sizeof(int));
    if (!ranks || !pin_to_sel || !pin_val || !sel_to_orig) {
        free(ranks); free(pin_to_sel); free(pin_val); free(sel_to_orig);
        return 0;
    }

    rank_all(H, syndrome, ranks);
    qsort(ranks, (size_t)n, sizeof(ColRank), cmp_rank);

    /* ---- Stage 1: Pin confident columns ---- */
    int np = 0;  /* number pinned */
    for (int i = 0; i < n && ranks[i].best_cnt >= pin_thresh
                      && (ranks[i].best_val == 1 || ranks[i].best_val == 2); ++i) {
        pin_to_sel[np] = ranks[i].idx;  /* original column index */
        pin_val[np] = ranks[i].best_val;
        np++;
    }

    /* Build residual syndrome: remove pinned columns' contribution */
    uint8_t *residual = (uint8_t *)malloc((size_t)m);
    memcpy(residual, syndrome, (size_t)m);
    for (int pi = 0; pi < np; ++pi) {
        int oc = pin_to_sel[pi];
        const SparseCol *sc = &H->col[oc];
        for (int k = 0; k < sc->c; ++k) {
            int r = sc->rows[k];
            residual[r] ^= gf4_mul(sc->values[k], pin_val[pi]);
        }
    }

    /* ---- Stage 2: Select unpinned columns ---- */
    /* Mark pinned columns as used */
    int *used = (int *)calloc((size_t)n, sizeof(int));
    for (int pi = 0; pi < np; ++pi) used[pin_to_sel[pi]] = 1;

    /* Take remaining top-ranked unpinned columns */
    const int extra_margin = (m > 100) ? 50 : (m / 2);
    int target = m - extra_margin;
    if (target < 1) target = 1;
    if (target > n) target = n;

    int ns = 0;  /* selected (unpinned) count */
    for (int i = 0; i < n && ns < target - np; ++i) {
        if (!used[ranks[i].idx]) {
            sel_to_orig[ns++] = ranks[i].idx;
        }
    }
    /* If not enough unpinned, take pinned columns — should be rare */
    for (int i = 0; i < n && ns < target - np; ++i) {
        /* already sorted, take any remaining */
    }

    int kept = np + ns;
    if (kept < 1) { free(ranks); free(pin_to_sel); free(pin_val);
                    free(sel_to_orig); free(residual); free(used); return 0; }

    printf("[alg4prefilt_gf4] pinned=%d, unpinned=%d, total=%d, rows=%d\n",
           np, ns, kept, m);

    /* ---- Stage 3: Build and solve the system ---- */
    /* System: [H_unpinned] · x = residual, with pinned cols as virtual rows */
    int sys_rows = m + np;
    uint8_t *M = (uint8_t *)calloc((size_t)sys_rows * (size_t)kept, 1);
    uint8_t *rhs = (uint8_t *)malloc((size_t)sys_rows);
    uint8_t *x = (uint8_t *)malloc((size_t)kept);

    if (!M || !rhs || !x) {
        free(ranks); free(pin_to_sel); free(pin_val); free(sel_to_orig);
        free(residual); free(used); free(M); free(rhs); free(x);
        return 0;
    }

    /* Original H rows for unpinned columns */
    for (int ki = 0; ki < ns; ++ki) {
        int oc = sel_to_orig[ki];
        const SparseCol *sc = &H->col[oc];
        for (int k = 0; k < sc->c; ++k) {
            int r = sc->rows[k];
            M[(size_t)r * kept + ki] = sc->values[k];
        }
    }
    memcpy(rhs, residual, (size_t)m);

    /* Virtual rows for pinned columns (equation: 1·e = pin_val) */
    for (int pi = 0; pi < np; ++pi) {
        int vr = m + pi;
        int ki = ns + pi;  /* pinned column is at position ns+pi in the system */
        M[(size_t)vr * kept + ki] = 1;
        rhs[vr] = pin_val[pi];
    }

    int solved = solve_lu_gf4(M, rhs, x, sys_rows, kept);

    if (solved) {
        memset(e_out, 0, (size_t)n);
        for (int ki = 0; ki < ns; ++ki)
            e_out[sel_to_orig[ki]] = x[ki];
        for (int pi = 0; pi < np; ++pi)
            e_out[pin_to_sel[pi]] = pin_val[pi];

        /* Check virtual-row violations */
        int nv = 0;
        for (int pi = 0; pi < np; ++pi) {
            if (x[ns + pi] != pin_val[pi]) nv++;
        }

        if (nv <= max_viol && count_nonzero(e_out, n) <= max_weight) {
            printf("[alg4prefilt_gf4] Success: kept=%d (pinned=%d), "
                   "weight=%d, violations=%d\n",
                   kept, np, count_nonzero(e_out, n), nv);
            free(ranks); free(pin_to_sel); free(pin_val); free(sel_to_orig);
            free(residual); free(used); free(M); free(rhs); free(x);
            return 1;
        } else if (nv > 0) {
            printf("[alg4prefilt_gf4] %d violations, weight=%d vs max=%d\n",
                   nv, count_nonzero(e_out, n), max_weight);
        }
    } else {
        printf("[alg4prefilt_gf4] Singular (sys %d×%d). Trying with %d pinned...\n",
               sys_rows, kept, np);
    }

    /* Fallback: try without pinned columns */
    if (ns > 0 && ns <= m) {
        memset(M, 0, (size_t)m * (size_t)ns * sizeof(uint8_t));
        for (int ki = 0; ki < ns; ++ki) {
            int oc = sel_to_orig[ki];
            const SparseCol *sc = &H->col[oc];
            for (int k = 0; k < sc->c; ++k)
                M[(size_t)sc->rows[k] * ns + ki] = sc->values[k];
        }
        memcpy(rhs, residual, (size_t)m);  /* still using cleaned residual */
        solved = solve_lu_gf4(M, rhs, x, m, ns);
        if (solved) {
            memset(e_out, 0, (size_t)n);
            for (int ki = 0; ki < ns; ++ki) e_out[sel_to_orig[ki]] = x[ki];
            for (int pi = 0; pi < np; ++pi) e_out[pin_to_sel[pi]] = pin_val[pi];
            if (count_nonzero(e_out, n) <= max_weight) {
                printf("[alg4prefilt_gf4] Success (fallback, no pin rows): "
                       "weight=%d\n", count_nonzero(e_out, n));
                free(ranks); free(pin_to_sel); free(pin_val); free(sel_to_orig);
                free(residual); free(used); free(M); free(rhs); free(x);
                return 1;
            }
        }
    }

    printf("[alg4prefilt_gf4] Failed.\n");
    free(ranks); free(pin_to_sel); free(pin_val); free(sel_to_orig);
    free(residual); free(used); free(M); free(rhs); free(x);
    return 0;
}
