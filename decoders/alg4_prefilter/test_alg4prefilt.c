/*
 * test_alg4prefilt.c — Test harness for alg4LDprefilt.
 *
 * Generates uniform binary errors at specified BER, maps pairs of consecutive
 * bits to GF(4) symbols, then runs the pre-filter decoder on the syndrome.
 *
 * For GF(16): maps 4 consecutive bits to GF(16) symbols.
 *
 * Tests at BER 1%, 2%, 3% with multiple random trials.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#include "sparse_utils.h"
#include "gf16.h"
#include "gf4.h"
#include "lu_solve.h"
#include "alg4prefiltgf16.h"
#include "alg4prefiltgf4.h"

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1000.0 * (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e6;
}

/* ---- Binary to GF(4) symbol conversion ---- */
/* Map 2 consecutive bits [b1 b0] to GF(4) symbol:
 *   00 -> 0  (no error)
 *   01 -> 1  (α^0, single-bit error)
 *   10 -> 2  (α^1, single-bit error)
 *   11 -> 3  (α^0+α^1, double-bit error — rare at low BER)
 */
static void binary_errors_to_gf4(const uint8_t *binary_errors, int n_bits,
                                  uint8_t *gf4_errors, int n_gf4)
{
    /* n_bits = 2 * n_gf4 */
    for (int i = 0; i < n_gf4; i++) {
        int b0 = binary_errors[2 * i] ? 1 : 0;
        int b1 = binary_errors[2 * i + 1] ? 1 : 0;
        gf4_errors[i] = (uint8_t)((b1 << 1) | b0);
    }
}

/* ---- Binary to GF(16) symbol conversion ---- */
/* Map 4 consecutive bits to GF(16) element in polynomial basis.
 * The 4 single-bit-error patterns are {1, 2, 4, 8}.
 */
static void binary_errors_to_gf16(const uint8_t *binary_errors, int n_bits,
                                   uint8_t *gf16_errors, int n_gf16)
{
    for (int i = 0; i < n_gf16; i++) {
        uint8_t sym = 0;
        for (int b = 0; b < 4; b++) {
            if (binary_errors[4 * i + b])
                sym |= (1 << b);
        }
        gf16_errors[i] = sym;
    }
}

/* ---- Generate uniform binary errors at a given BER ---- */
/* Returns the number of bit errors. */
static int generate_binary_errors(uint8_t *bits, int n_bits, double ber)
{
    int count = 0;
    for (int i = 0; i < n_bits; i++) {
        double r = (double)rand() / (double)RAND_MAX;
        if (r < ber) {
            bits[i] = 1;
            count++;
        } else {
            bits[i] = 0;
        }
    }
    return count;
}

/* ---- Count GF(4) symbol types ---- */
static void count_gf4_errors(const uint8_t *e, int n, int *stats)
{
    /* stats[0] = count of value 1, stats[1] = count of value 2,
       stats[2] = count of value 3 */
    memset(stats, 0, 3 * sizeof(int));
    for (int i = 0; i < n; i++) {
        if (e[i] == 1) stats[0]++;
        else if (e[i] == 2) stats[1]++;
        else if (e[i] == 3) stats[2]++;
    }
}

/* ---- Count GF(16) symbol types: {1,2,4,8} vs rest ---- */
static void count_gf16_errors(const uint8_t *e, int n, int *stats)
{
    /* stats[0] = count in {1,2,4,8}, stats[1] = count of other non-zero */
    memset(stats, 0, 2 * sizeof(int));
    for (int i = 0; i < n; i++) {
        if (e[i] == 0) continue;
        if (e[i] == 1 || e[i] == 2 || e[i] == 4 || e[i] == 8)
            stats[0]++;
        else
            stats[1]++;
    }
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <matrix_file> <field> <ber_pct_list> <trials> [inter_thresh] [agree_thresh] [max_tries]\n", argv[0]);
        fprintf(stderr, "  field: 4=GF(4), 16=GF(16)\n");
        fprintf(stderr, "  ber_pct_list: comma-separated, e.g. \"1,2,3\"\n");
        fprintf(stderr, "  Defaults: inter_thresh=2, agree_thresh=3(GF4)/2(GF16), max_tries=20\n");
        fprintf(stderr, "\nExample: %s H_GF4_n1000_R0.60.txt 4 1,2,3 50\n", argv[0]);
        return 1;
    }

    const char *matrix_file = argv[1];
    int field = atoi(argv[2]);
    const char *ber_list = argv[3];
    int trials = atoi(argv[4]);
    int thr = (argc >= 6) ? atoi(argv[5]) : 2;
    int agree = (argc >= 7) ? atoi(argv[6]) : 0;
    int max_tries = (argc >= 8) ? atoi(argv[7]) : 20;

    /* Encode threshold_v: (agree << 8) | inter_thresh
       If agree == 0, default is field-dependent (3 for GF(4), 2 for GF(16)) */
    if (agree > 0) thr = (agree << 8) | (thr & 0xFF);

    if (field != 4 && field != 16) {
        fprintf(stderr, "[test] Field must be 4 (GF(4)) or 16 (GF(16)).\n");
        return 1;
    }

    srand(12345);
    SparseMatrix H;
    if (!load_sparse_matrix(matrix_file, &H)) {
        fprintf(stderr, "[test] Failed to load matrix: %s\n", matrix_file);
        return 1;
    }

    /* Parse BER list */
    double ber_vals[10];
    int n_ber = 0;
    char *list_copy = strdup(ber_list);
    char *token = strtok(list_copy, ",");
    while (token && n_ber < 10) {
        ber_vals[n_ber++] = atof(token);
        token = strtok(NULL, ",");
    }
    free(list_copy);

    /* Compute n_bits based on field */
    int bits_per_sym = (field == 4) ? 2 : 4;
    int n_syms = H.cols;
    int n_bits = n_syms * bits_per_sym;

    printf("================================================================\n");
    printf("alg4LDprefilt Test (consistency pre-filter)\n");
    printf("================================================================\n");
    printf("Matrix:  %s\n", matrix_file);
    printf("Size:    %d x %d (c=%d, field=GF(%d))\n", H.rows, H.cols, H.c, field);
    printf("Bits:    %d (%d bits/symbol)\n", n_bits, bits_per_sym);
    printf("BER:     ");
    for (int b = 0; b < n_ber; b++) printf("%.1f%%%s", ber_vals[b], (b < n_ber-1) ? ", " : "");
    printf("\nTrials:  %d per BER, threshold_v=%d, max_tries=%d\n", trials, thr, max_tries);
    printf("----------------------------------------------------------------\n\n");

    /* Allocate */
    uint8_t *binary_err = (uint8_t *)malloc(n_bits);
    uint8_t *e_true = (uint8_t *)malloc(n_syms);
    uint8_t *syndrome = (uint8_t *)malloc(H.rows);
    uint8_t *e_recv = (uint8_t *)calloc(n_syms, 1);
    if (!binary_err || !e_true || !syndrome || !e_recv) {
        fprintf(stderr, "[test] Allocation failed.\n");
        free(binary_err); free(e_true); free(syndrome); free(e_recv);
        free_sparse_matrix(&H); return 1;
    }

    /* Initialize field */
    if (field == 4) gf4_init(); else gf16_init();

    /* For each BER */
    for (int bi = 0; bi < n_ber; bi++) {
        double ber = ber_vals[bi] / 100.0;
        int passed = 0, failed = 0;
        double total_time = 0.0;
        int total_prefilter_pass = 0, total_inter_pass = 0;

        for (int tr = 0; tr < trials; tr++) {
            /* Generate binary errors */
            generate_binary_errors(binary_err, n_bits, ber);

            /* Convert to GF symbols */
            if (field == 4)
                binary_errors_to_gf4(binary_err, n_bits, e_true, n_syms);
            else
                binary_errors_to_gf16(binary_err, n_bits, e_true, n_syms);

            /* Compute syndrome = H * e */
            if (field == 4)
                sparse_matvec_gf4(&H, e_true, syndrome);
            else
                sparse_matvec_gf16(&H, e_true, syndrome);

            /* Count actual error types */
            int sym_errs = count_nonzero(e_true, n_syms);
            int gf4_stats[3] = {0};
            int gf16_stats[2] = {0};
            if (field == 4) count_gf4_errors(e_true, n_syms, gf4_stats);
            else count_gf16_errors(e_true, n_syms, gf16_stats);

            /* Compute max_weight = actual + margin */
            int max_weight = sym_errs + 20;

            /* Run decoder */
            memset(e_recv, 0, n_syms);
            double t0 = now_ms();
            int attempts;
            if (field == 4)
                attempts = alg4prefilt_gf4(&H, syndrome, e_recv, thr, max_weight, max_tries);
            else
                attempts = alg4prefilt_gf16(&H, syndrome, e_recv, thr, max_weight, max_tries);
            double elapsed = now_ms() - t0;

            int ok = (attempts > 0 && compare_vectors(e_true, e_recv, n_syms));
            if (ok) passed++; else failed++;
            total_time += elapsed;

            if (tr % 10 == 0 || tr == trials - 1) {
                printf("  [BER=%.1f%% trial %3d/%d] sym_errs=%d (1:%d, 2:%d, 3:%d) %s in %.1fms\n",
                       ber * 100, tr + 1, trials, sym_errs,
                       gf4_stats[0], gf4_stats[1], gf4_stats[2],
                       ok ? "PASS" : "FAIL", elapsed);
            }
        }

        printf("\n--- Summary for BER=%.1f%% (%d trials) ---\n", ber * 100, trials);
        printf("  Passed:  %d/%d (%.1f%%)\n", passed, trials,
               100.0 * passed / trials);
        printf("  Avg time: %.2f ms\n", total_time / trials);
        printf("\n");
    }

    free(binary_err); free(e_true); free(syndrome); free(e_recv);
    free_sparse_matrix(&H);

    printf("================================================================\n");
    printf("Done.\n");
    return 0;
}
