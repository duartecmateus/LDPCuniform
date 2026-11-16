/*
 * validate_gf16.c — GF(16) pre-filter validation with binary-to-GF(16)
 * error generation. Measures true-error capture rate with the value filter
 * {1,2,4,8} and without (any-value agreement).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#include "sparse_utils.h"
#include "gf16.h"

typedef struct { int idx; int inter; uint8_t best_val; int best_cnt; int best_any; } RC;

static int cmp_rank(const void *a, const void *b)
{
    const RC *ra = (const RC *)a, *rb = (const RC *)b;
    if (ra->best_cnt != rb->best_cnt)
        return (ra->best_cnt < rb->best_cnt) ? 1 : -1;
    if (ra->best_any != rb->best_any)
        return (ra->best_any < rb->best_any) ? 1 : -1;
    if (ra->inter != rb->inter)
        return (ra->inter < rb->inter) ? 1 : -1;
    return (rand() % 3) - 1;
}

static void matvec_gf16(const SparseMatrix *H, const uint8_t *e, uint8_t *s)
{
    memset(s, 0, (size_t)H->rows);
    for (int j = 0; j < H->cols; j++) {
        if (!e[j]) continue;
        const SparseCol *sc = &H->col[j];
        for (int k = 0; k < sc->c; k++)
            s[sc->rows[k]] ^= (sc->values[k] == 1) ? e[j] : gf16_mul(sc->values[k], e[j]);
    }
}

/* Generate binary errors at BER, convert to GF(16) symbols (4 bits each) */
static int binary_to_gf16_errors(uint8_t *e_out, int n_syms, double ber)
{
    int total_errs = 0;
    memset(e_out, 0, n_syms);
    for (int i = 0; i < n_syms; i++) {
        uint8_t sym = 0;
        for (int b = 0; b < 4; b++) {
            if ((double)rand() / RAND_MAX < ber) sym |= (1 << b);
        }
        e_out[i] = sym;
        if (sym) total_errs++;
    }
    return total_errs;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <matrix> <ber_pct_list> <trials>\n", argv[0]);
        return 1;
    }
    srand(12345);
    SparseMatrix H;
    if (!load_sparse_matrix(argv[1], &H)) return 1;
    gf16_init();

    int n = H.cols, m = H.rows;
    uint8_t *e_true = (uint8_t *)malloc(n);
    uint8_t *syndrome = (uint8_t *)malloc(m);
    uint8_t likely_set[4] = {1, 2, 4, 8};
    int8_t likely_idx[16];
    for (int i = 0; i < 16; i++) likely_idx[i] = -1;
    for (int i = 0; i < 4; i++) likely_idx[likely_set[i]] = (int8_t)i;

    double ber_vals[5];
    int n_ber = 0;
    char *cp = strdup(argv[2]);
    for (char *t = strtok(cp, ","); t && n_ber < 5; t = strtok(NULL, ","))
        ber_vals[n_ber++] = atof(t);
    free(cp);
    int trials = atoi(argv[3]);

    printf("Matrix: %d x %d, c=%d, bits=%d\n", m, n, H.c, n * 4);
    printf("Binary BER: "); for (int b = 0; b < n_ber; b++) printf("%.1f%% ", ber_vals[b]);
    printf("\nTrials: %d\n\n", trials);

    for (int bi = 0; bi < n_ber; bi++) {
        double ber = ber_vals[bi] / 100.0;
        int tot_capt_set = 0, tot_capt_any = 0, tot_in_set = 0, tot_errs = 0;
        double tot_time = 0;

        for (int tr = 0; tr < trials; tr++) {
            int n_err = binary_to_gf16_errors(e_true, n, ber);
            tot_errs += n_err;
            matvec_gf16(&H, e_true, syndrome);

            double t0 = (double)clock() / CLOCKS_PER_SEC * 1000;
            RC *ranks = (RC *)malloc(n * sizeof(RC));
            for (int j = 0; j < n; j++) {
                const SparseCol *sc = &H.col[j];
                int votes[5] = {0,0,0,0,0}, votes_all[16] = {0};
                ranks[j].inter = 0;

                for (int k = 0; k < sc->c; k++) {
                    int r = sc->rows[k];
                    if (syndrome[r] == 0) continue;
                    ranks[j].inter++;
                    uint8_t e_hat = gf16_mul(syndrome[r], gf16_inv(sc->values[k]));
                    int idx = likely_idx[e_hat & 0xF];
                    if (idx >= 0) votes[idx]++;
                    votes_all[e_hat & 0xF]++;
                }

                ranks[j].best_val = 0;
                ranks[j].best_cnt = 0;
                for (int i = 0; i < 4; i++)
                    if (votes[i] > ranks[j].best_cnt) {
                        ranks[j].best_cnt = votes[i];
                        ranks[j].best_val = likely_set[i];
                    }

                ranks[j].best_any = 0;
                for (int i = 1; i < 16; i++)
                    if (votes_all[i] > ranks[j].best_any)
                        ranks[j].best_any = votes_all[i];
            }
            qsort(ranks, n, sizeof(RC), cmp_rank);
            tot_time += (double)clock() / CLOCKS_PER_SEC * 1000 - t0;

            /* Count errors that are in {1,2,4,8} */
            int in_set = 0;
            for (int j = 0; j < n; j++)
                if (e_true[j] && (likely_idx[e_true[j] & 0xF] >= 0)) in_set++;
            tot_in_set += in_set;

            /* Captured within top n_err with value filter vs any-value */
            int cap_set = 0, cap_any = 0;
            for (int i = 0; i < n && i < n_err; i++) {
                if (e_true[ranks[i].idx]) {
                    if (ranks[i].best_cnt > 0) cap_set++;
                    cap_any++;
                }
            }
            tot_capt_set += cap_set;
            tot_capt_any += cap_any;
            free(ranks);
        }

        double avg_err = (double)tot_errs / trials;
        int avg_in_set = tot_in_set / trials;
        printf("--- BER=%.1f%% (avg %.0f sym errs, ~%d in {1,a,a^2,a^3}) ---\n",
               ber * 100, avg_err, avg_in_set);
        printf("  Ranking:           %.2f ms avg\n", tot_time / trials);
        printf("  Value-filter cap:  %.1f / %d (%.0f%% of set errs)\n",
               (double)tot_capt_set / trials, avg_in_set,
               100.0 * tot_capt_set / (trials > 0 && avg_in_set > 0 ? trials * avg_in_set : 1));
        printf("  Any-value cap:     %.1f / %.0f (%.0f%% of all errs)\n",
               (double)tot_capt_any / trials, avg_err,
               100.0 * tot_capt_any / (trials * avg_err));
        printf("\n");
    }

    free(e_true); free(syndrome);
    free_sparse_matrix(&H);
    return 0;
}
