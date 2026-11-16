#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "sparse_utils.h"
#include "alg3prunebin.h"
#include "ldpcdec.h"

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1000.0 * (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e6;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <matrix_file> <error_rate_pct>\n", argv[0]);
        return 1;
    }

    const char *mf = argv[1];
    double err_pct = atof(argv[2]);

    srand(54321);
    SparseMatrix H;
    if (!load_sparse_matrix(mf, &H)) { fprintf(stderr, "Failed to load %s\n", mf); return 1; }

    printf("=============================================\n");
    printf("BP vs Metal Pruning — %d×%d  c=%d  field=%d  err=%.2f%%\n",
           H.rows, H.cols, H.c, H.field, err_pct);
    printf("=============================================\n\n");

    int n_err = (int)(H.cols * err_pct / 100.0 + 0.5);
    if (n_err <= 0) n_err = 1;

    /* Generate error vector */
    uint8_t *e = (uint8_t *)calloc(H.cols, 1);
    for (int p = 0; p < n_err; ) {
        int pos = rand() % H.cols;
        if (!e[pos]) { e[pos] = 1; p++; }
    }

    uint8_t *syndrome = (uint8_t *)malloc(H.rows);
    uint8_t *e_recv = (uint8_t *)calloc(H.cols, 1);
    if (!syndrome || !e_recv) { free(e); free(syndrome); free(e_recv); return 1; }

    /* Syndrome */
    sparse_matvec(&H, e, syndrome);

    int err_wgt = count_nonzero(e, H.cols);
    int syn_wgt = count_nonzero(syndrome, H.rows);
    printf("Error weight: %d / %d   Syndrome weight: %d / %d\n\n",
           err_wgt, H.cols, syn_wgt, H.rows);

    /* ── Metal Pruning ── */
    printf("── Metal Pruning (alg3_prunebin) ──\n");
    double t0 = now_ms();
    int att = alg3_prunebin(&H, syndrome, e_recv, 4, n_err + 20, 20);
    double dt = now_ms() - t0;
    int prune_ok = (att > 0 && compare_vectors(e, e_recv, H.cols));
    printf("  Time: %.3f ms  Attempts: %d  Success: %s\n\n",
           dt, att, prune_ok ? "YES" : "NO");

    /* ── BP Decoders ── */
    printf("── BP Decoders ──\n");

    /* Build dense matrix for BP (needs H.cols columns for the full code) */
    int **dense = (int **)malloc(H.rows * sizeof(int *));
    for (int r = 0; r < H.rows; r++) {
        dense[r] = (int *)calloc(H.cols, sizeof(int));
    }
    for (int j = 0; j < H.cols; j++)
        for (int k = 0; k < H.col[j].c; k++)
            dense[H.col[j].rows[k]][j] = 1;

    double llr0 = err_pct > 0 ? log((100.0 - err_pct) / err_pct) : 10.0;
    double *llr = (double *)malloc(H.cols * sizeof(double));
    int *dec = (int *)malloc(H.cols * sizeof(int));

    /* SPA */
    printf("  SPA:\n");
    for (int r = 0; r < 3; r++) {
        for (int j = 0; j < H.cols; j++) llr[j] = e[j] ? -llr0 : llr0;
        int iters;
        t0 = now_ms();
        int ok = ldpc_decode_spa(H.rows, H.cols, dense, llr, dec, &iters);
        dt = now_ms() - t0;
        uint8_t *e_hat = (uint8_t *)malloc(H.cols);
        for (int j = 0; j < H.cols; j++) e_hat[j] = e[j] ^ (uint8_t)dec[j];
        int ok_vec = compare_vectors(e, e_hat, H.cols);
        free(e_hat);
        printf("    Time: %.3f ms  Converged: %s  Correct: %s  Iters: %d\n",
               dt, ok ? "YES" : "NO", ok_vec ? "YES" : "NO", iters);
    }

    /* Min-Sum */
    printf("  Min-Sum:\n");
    for (int r = 0; r < 3; r++) {
        for (int j = 0; j < H.cols; j++) llr[j] = e[j] ? -llr0 : llr0;
        int iters;
        t0 = now_ms();
        int ok = ldpc_decode_min_sum(H.rows, H.cols, dense, llr, dec, &iters);
        dt = now_ms() - t0;
        uint8_t *e_hat = (uint8_t *)malloc(H.cols);
        for (int j = 0; j < H.cols; j++) e_hat[j] = e[j] ^ (uint8_t)dec[j];
        int ok_vec = compare_vectors(e, e_hat, H.cols);
        free(e_hat);
        printf("    Time: %.3f ms  Converged: %s  Correct: %s  Iters: %d\n",
               dt, ok ? "YES" : "NO", ok_vec ? "YES" : "NO", iters);
    }

    /* Cleanup */
    for (int r = 0; r < H.rows; r++) free(dense[r]);
    free(dense);
    free(llr);
    free(dec);
    free(e);
    free(e_recv);
    free(syndrome);
    free_sparse_matrix(&H);

    return 0;
}
