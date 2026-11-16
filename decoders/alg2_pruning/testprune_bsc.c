#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <errno.h>

#include "gf256.h"
#include "sparse_utils.h"
#include "alg2prune.h"

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1000.0 * (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e6;
}

/* Generate bit-level BSC errors, then pack into GF(256) symbols.
   size_bits = total bits (e.g. 2000 for n=250 symbols × 8 bits)
   n_symbols = number of GF(256) symbols */
static void generate_bsc_error_gf256(uint8_t *e_sym, int n_symbols,
                                      int size_bits, double bit_error_rate)
{
    int n_bits = size_bits;
    uint8_t *bits = (uint8_t*)calloc(n_bits, 1);
    if (!bits) { fprintf(stderr, "alloc failed\n"); exit(1); }

    int nz = (int)((double)n_bits * bit_error_rate / 100.0 + 0.5);
    if (nz <= 0) nz = 1;
    int placed = 0;
    while (placed < nz) {
        int pos = rand() % n_bits;
        if (bits[pos]) continue;
        bits[pos] = 1;
        ++placed;
    }

    memset(e_sym, 0, n_symbols);
    for (int s = 0; s < n_symbols; s++) {
        uint8_t val = 0;
        for (int b = 0; b < 8; b++) {
            if (s * 8 + b < n_bits && bits[s * 8 + b])
                val |= (1 << b);
        }
        e_sym[s] = val;
    }
    free(bits);
}

int main(int argc, char **argv)
{
    if (argc != 7) {
        fprintf(stderr, "Usage: %s <matrix> <bit_error_pct> <threshold_v> "
                        "<max_weight> <max_tries> <seed>\n", argv[0]);
        return 1;
    }

    const char *matrix_file = argv[1];
    double bit_error_pct = atof(argv[2]);
    int threshold_v = atoi(argv[3]);
    int max_weight = atoi(argv[4]);
    int max_tries = atoi(argv[5]);
    int seed = atoi(argv[6]);

    srand(seed);

    SparseMatrix H;
    if (!load_sparse_matrix(matrix_file, &H)) return 1;

    int rows = H.rows, cols = H.cols;
    int total_bits = cols * 8;  /* GF(256) → 8 bits per symbol */

    printf("Matrix: %s  %d x %d  c=%d\n", matrix_file, rows, cols, H.c);
    printf("Bit error: %.2f%%  → %d bits, %d GF(256) symbols\n",
           bit_error_pct, total_bits, cols);

    uint8_t *e_orig = (uint8_t*)calloc(cols, 1);
    uint8_t *e_recv = (uint8_t*)calloc(cols, 1);
    uint8_t *syndrome = (uint8_t*)malloc(rows);

    /* Generate realistic bit-level errors, packed into symbols */
    generate_bsc_error_gf256(e_orig, cols, total_bits, bit_error_pct);
    int sym_err_weight = count_nonzero(e_orig, cols);
    int bit_errs = 0;
    for (int j = 0; j < cols; j++)
        for (int b = 0; b < 8; b++)
            if (e_orig[j] & (1 << b)) bit_errs++;

    printf("BSC errors: %d bit flips across %d/%d symbols (%.1f%% symb err rate)\n",
           bit_errs, sym_err_weight, cols,
           100.0 * sym_err_weight / cols);

    /* Syndrome */
    sparse_matvec_gf256(&H, e_orig, syndrome);
    int synd_w = count_nonzero(syndrome, rows);
    printf("Syndrome weight: %d / %d\n", synd_w, rows);

    /* Decode */
    double t0 = now_ms();
    int attempts = alg2_prune(&H, syndrome, e_recv,
                              threshold_v, max_weight, max_tries);
    double elapsed = now_ms() - t0;

    int success = (attempts > 0 && compare_vectors(e_orig, e_recv, cols));

    printf("Decoder: attempts=%d  time=%.3fms  success=%s\n",
           attempts, elapsed, success ? "YES" : "NO");

    if (!success) {
        /* Show which symbols are wrong */
        int mismatches = 0;
        for (int j = 0; j < cols; j++)
            if (e_orig[j] != e_recv[j]) mismatches++;
        printf("  Mismatched symbols: %d / %d\n", mismatches, cols);
    }

    free(e_orig); free(e_recv); free(syndrome);
    free_sparse_matrix(&H);
    return success ? 0 : 1;
}
