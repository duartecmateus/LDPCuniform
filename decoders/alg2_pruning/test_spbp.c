/* Test sparse BP decoder on large PEG matrices.
   Usage: test_spbp <matrix> <err_pct> <max_iters> [seed]
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "sparse_utils.h"
#include "spbp.h"

static void gen_binary_errors(uint8_t *e, int size, double pct)
{
    int nz = (int)((double)size * pct / 100.0 + 0.5);
    if (nz <= 0) nz = 1;
    memset(e, 0, (size_t)size);
    int placed = 0;
    while (placed < nz) {
        int pos = rand() % size;
        if (e[pos]) continue;
        e[pos] = 1;
        ++placed;
    }
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <matrix> <err_pct> <max_iters> [seed]\n", argv[0]);
        return 1;
    }

    const char *mf = argv[1];
    double err_pct = atof(argv[2]);
    int max_iters = atoi(argv[3]);
    int seed = (argc > 4) ? atoi(argv[4]) : 54321;

    srand(seed);

    SparseMatrix H;
    if (!load_sparse_matrix(mf, &H)) return 1;

    int rows = H.rows, cols = H.cols;
    printf("Matrix: %s  %d x %d  c=%d\n", mf, rows, cols, H.c);
    printf("Error: %.2f%% BSC, max iters=%d\n", err_pct, max_iters);

    uint8_t *e = (uint8_t *)malloc(cols);
    if (!e) { fprintf(stderr, "alloc failed\n"); return 1; }

    gen_binary_errors(e, cols, err_pct);
    int err_w = 0;
    for (int j = 0; j < cols; j++) if (e[j]) err_w++;

    /* LLR: all-zero transmitted, received = e */
    double *llr = (double *)malloc(cols * sizeof(double));
    if (!llr) { fprintf(stderr, "alloc failed\n"); return 1; }

    double llr0 = log((100.0 - err_pct) / err_pct);
    for (int j = 0; j < cols; j++)
        llr[j] = e[j] ? -llr0 : llr0;

    printf("Errors: %d / %d  (%.2f%%)\n", err_w, cols, 100.0 * err_w / cols);

    /* Run SPA */
    printf("\n--- SPA (Sum-Product) ---\n");
    BPResult r_spa = spbp_decode_spa(&H, llr, max_iters);

    printf("  Converged: %s  Time: %.3fms  Iters: %d\n",
           r_spa.converged ? "YES" : "NO",
           r_spa.time_ms, r_spa.iters_used);

    /* 0-codeword transmitted → correct means decoded == 0 everywhere */
    int is_zero = 1;
    for (int j = 0; j < cols; j++) {
        if (r_spa.decoded[j]) { is_zero = 0; break; }
    }

    /* Also verify syndrome directly */
    uint8_t *synd_check = (uint8_t*)calloc(rows, 1);
    for (int j = 0; j < cols; j++) {
        if (!r_spa.decoded[j]) continue;
        for (int k = 0; k < H.col[j].c; k++) {
            int r = H.col[j].rows[k];
            synd_check[r] ^= 1;
        }
    }
    int synd_ok = 1;
    for (int r = 0; r < rows; r++) { if (synd_check[r]) { synd_ok = 0; break; } }
    free(synd_check);

    printf("  All-zero codeword: %s  (syndrome check: %s)\n",
           is_zero ? "YES" : "NO",
           synd_ok ? "PASS" : "FAIL");

    /* Check: did BP recover the error vector? (ê = e means decoded = 0) */
    int match = is_zero;
    printf("  Correct (ê = e): %s\n", match ? "YES" : "NO");
    if (!is_zero) {
        int nz = 0;
        for (int j = 0; j < cols && nz < 5; j++)
            if (r_spa.decoded[j]) { printf("  decoded[%d]=%d\n", j, r_spa.decoded[j]); nz++; }
    }

    /* Run Min-Sum */
    printf("\n--- Min-Sum ---\n");
    BPResult r_ms = spbp_decode_ms(&H, llr, max_iters);

    printf("  Converged: %s  Time: %.3fms  Iters: %d\n",
           r_ms.converged ? "YES" : "NO",
           r_ms.time_ms, r_ms.iters_used);

    int ms_match = 1;
    for (int j = 0; j < cols; j++) {
        if (r_ms.decoded[j] != e[j]) { ms_match = 0; break; }
    }
    printf("  Correct: %s\n", ms_match ? "YES" : "NO");

    printf("\n--- Summary ---\n");
    printf("%-18s %10s %10s %10s\n", "Decoder", "Time(ms)", "Converged", "Correct");
    printf("%-18s %10.3f %10s %10s\n", "SPA", r_spa.time_ms,
           r_spa.converged ? "YES" : "NO", match ? "YES" : "NO");
    printf("%-18s %10.3f %10s %10s\n", "Min-Sum", r_ms.time_ms,
           r_ms.converged ? "YES" : "NO", ms_match ? "YES" : "NO");

    spbp_free(&r_spa);
    spbp_free(&r_ms);
    free(e);
    free(llr);
    free_sparse_matrix(&H);

    return 0;
}
