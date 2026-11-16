#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "sparse_utils.h"
#include "alg2prune.h"
#include "gf256.h"
#include "ldpcdec.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1000.0 * (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e6;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <matrix_file> <field> <error_rate_pct> <threshold_v>\n", argv[0]);
        return 1;
    }
    const char *mf = argv[1];
    int field = atoi(argv[2]);
    double err_pct = atof(argv[3]);
    int thr = atoi(argv[4]);

    srand(54321);
    SparseMatrix H;
    if (!load_sparse_matrix(mf, &H)) return 1;

    int n_err = (int)(H.cols * err_pct / 100.0 + 0.5);
    if (n_err <= 0) n_err = 1;
    uint8_t *e = (uint8_t*)calloc(H.cols, 1);
    for (int p = 0; p < n_err; ) {
        int pos = rand() % H.cols;
        if (!e[pos]) { e[pos] = field ? (uint8_t)((rand()%255)+1) : 1; p++; }
    }
    uint8_t *syndrome = (uint8_t*)malloc(H.rows);
    uint8_t *e_recv = (uint8_t*)calloc(H.cols, 1);

    /* F256 pruning on GF256 matrix */
    double t0, dt_prune = 0;
    int ok_prune = 0;
    if (field == 1) {
        sparse_matvec_gf256(&H, e, syndrome);
        gf256_init();
        t0 = now_ms();
        int att = alg2_prune(&H, syndrome, e_recv, thr, n_err + 20, 20);
        dt_prune = now_ms() - t0;
        ok_prune = (att > 0 && compare_vectors(e, e_recv, H.cols));
        memset(e_recv, 0, H.cols);
    }

    /* Build dense matrix for BP decoders */
    double llr0 = err_pct > 0 ? log((100.0 - err_pct) / err_pct) : 10.0;
    double *llr = (double*)malloc(H.cols * sizeof(double));
    int *dec = (int*)malloc(H.cols * sizeof(int));
    
    /* Binary error vector for BP (all-zero codeword) */
    uint8_t *e_bin = (uint8_t*)calloc(H.cols, 1);
    for (int j = 0; j < H.cols; j++) e_bin[j] = e[j] ? 1 : 0;

    int **dense = (int**)malloc(H.rows * sizeof(int*));
    for (int r = 0; r < H.rows; r++) {
        dense[r] = (int*)calloc(H.cols, sizeof(int));
        for (int j = 0; j < H.cols; j++)
            for (int k = 0; k < H.col[j].c; k++)
                if (H.col[j].rows[k] == r) dense[r][j] = 1;
    }

    /* SPA */
    for (int j = 0; j < H.cols; j++) llr[j] = e_bin[j] ? -llr0 : llr0;
    int iters;
    t0 = now_ms();
    int ok_spa = ldpc_decode_spa(H.rows, H.cols, dense, llr, dec, &iters);
    double dt_spa = now_ms() - t0;
    int spa_allzero = 1;
    for (int j = 0; j < H.cols; j++) if (dec[j]) { spa_allzero = 0; break; }

    /* Min-Sum */
    for (int j = 0; j < H.cols; j++) llr[j] = e_bin[j] ? -llr0 : llr0;
    t0 = now_ms();
    int ok_ms = ldpc_decode_min_sum(H.rows, H.cols, dense, llr, dec, &iters);
    double dt_ms = now_ms() - t0;
    int ms_allzero = 1;
    for (int j = 0; j < H.cols; j++) if (dec[j]) { ms_allzero = 0; break; }

    /* CSV output */
    printf("%s,%d,%d,%d,%d,%.4f,%d,%.3f,%d,%d,%.3f,%d,%d,%.3f,%d,%d\n",
           mf, H.rows, H.cols, H.c, field, err_pct, thr,
           dt_prune, ok_prune, ok_spa, dt_spa, spa_allzero,
           ok_ms, dt_ms, ms_allzero, iters);

    for (int r = 0; r < H.rows; r++) free(dense[r]);
    free(dense); free(e); free(e_bin); free(e_recv); free(syndrome);
    free(llr); free(dec);
    free_sparse_matrix(&H);
    return 0;
}
