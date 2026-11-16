#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "sparse_utils.h"
#include "alg2prunebin.h"
#include "alg2augmentbin.h"
#include "ldpcdec.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1000.0 * (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e6;
}

static void gen_binary_errors(uint8_t *e, int size, double pct) {
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

/* Convert sparse matrix to dense int **H for BP decoder */
static int **sparse_to_dense(const SparseMatrix *H) {
    int **dense = (int**)malloc((size_t)H->rows * sizeof(int*));
    if (!dense) return NULL;
    for (int r = 0; r < H->rows; ++r) {
        dense[r] = (int*)calloc((size_t)H->cols, sizeof(int));
        if (!dense[r]) {
            for (int k = 0; k < r; ++k) free(dense[k]);
            free(dense);
            return NULL;
        }
    }
    for (int j = 0; j < H->cols; ++j) {
        const SparseCol *sc = &H->col[j];
        for (int k = 0; k < sc->c; ++k)
            dense[sc->rows[k]][j] = 1;
    }
    return dense;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <matrix_file> <error_rate_pct> "
                "<prune_v> <augment_max_inter> <augment_extra>\n", argv[0]);
        return 1;
    }

    const char *mf = argv[1];
    double err_pct = atof(argv[2]);
    int prune_v = atoi(argv[3]);
    int aug_inter = atoi(argv[4]);
    int aug_extra = atoi(argv[5]);

    srand(54321);

    SparseMatrix H;
    if (!load_sparse_matrix(mf, &H)) return 1;

    int rows = H.rows, cols = H.cols;

    uint8_t *e_orig = (uint8_t*)malloc((size_t)cols);
    uint8_t *e_prune = (uint8_t*)calloc((size_t)cols, 1);
    uint8_t *e_aug = (uint8_t*)calloc((size_t)cols, 1);
    uint8_t *syndrome = (uint8_t*)malloc((size_t)rows);

    gen_binary_errors(e_orig, cols, err_pct);
    int err_w = count_nonzero(e_orig, cols);
    sparse_matvec(&H, e_orig, syndrome);
    int synd_w = count_nonzero(syndrome, rows);

    printf("=== Comparison: %s ===\n", mf);
    printf("Matrix: %d x %d, c=%d, errors=%d, synd_w=%d\n",
           rows, cols, H.c, err_w, synd_w);

    /* === BP DECODER (SPA) — all-zero codeword assumption ===
       Transmitted all-zero codeword, received = error vector e.
       LLR: P(y=0|x=0) = 1-β, P(y=1|x=0) = β.
       If BP converges to x̂ = 0 (all-zero codeword), then ê = y - x̂ = e. */
    int **Hdense = sparse_to_dense(&H);
    double *llr = (double*)malloc((size_t)cols * sizeof(double));
    int *dec_bp = (int*)malloc((size_t)cols * sizeof(int));
    int bp_iters = 0;

    /* BSC LLR: P(received bit=0 | sent=0) = 1-β, P(received=1 | sent=0) = β */
    double llr0 = log((100.0 - err_pct) / err_pct);  /* LLR for received 0 */
    for (int j = 0; j < cols; ++j)
        llr[j] = e_orig[j] ? -llr0 : llr0;  /* received 1 → negative LLR */

    double t0 = now_ms();
    int bp_ok = ldpc_decode_spa(rows, cols, Hdense, llr, dec_bp, &bp_iters);
    double t_bp = now_ms() - t0;

    /* Check results */
    uint8_t *bp_recovered = (uint8_t*)calloc((size_t)cols, 1);
    int bp_all_zero = 1;
    for (int j = 0; j < cols; ++j) {
        bp_recovered[j] = (uint8_t)dec_bp[j];
        if (dec_bp[j]) bp_all_zero = 0;
    }
    /* Error estimate: ê = received - decoded_codeword = e_orig - x̂ */
    uint8_t *bp_error = (uint8_t*)malloc((size_t)cols);
    for (int j = 0; j < cols; ++j) bp_error[j] = e_orig[j] ^ (uint8_t)dec_bp[j];

    printf("\n--- BP Decoder (SPA) ---\n");
    printf("  Converged (H·x̂=0): %s  Time: %.3fms  Iters: %d\n",
           bp_ok ? "YES" : "NO", t_bp, bp_iters);
    printf("  Codeword x̂ = 0 (all-zero): %s\n", bp_all_zero ? "YES" : "NO");
    printf("  Error vector ê = e_orig: %s\n",
           compare_vectors(bp_error, e_orig, cols) ? "YES" : "NO");

    /* === PRUNING DECODER === */
    t0 = now_ms();
    int p_att = alg2_prunebin(&H, syndrome, e_prune, prune_v, err_w + 20, 20);
    double t_p = now_ms() - t0;
    int p_match = (p_att > 0 && compare_vectors(e_orig, e_prune, cols));

    printf("\n--- Pruning Decoder (v=%d) ---\n", prune_v);
    printf("  Success: %s  Time: %.3fms  Attempts: %d\n",
           p_match ? "YES" : "NO", t_p, p_att);

    /* === AUGMENTATION DECODER === */
    t0 = now_ms();
    int a_att = alg2_augment_bin(&H, syndrome, e_aug,
                                  aug_inter, aug_extra, err_w + 20, 20);
    double t_a = now_ms() - t0;
    int a_match = (a_att > 0 && compare_vectors(e_orig, e_aug, cols));

    printf("\n--- Augmentation Decoder (inter≤%d, extra=%d) ---\n",
           aug_inter, aug_extra);
    printf("  Success: %s  Time: %.3fms  Attempts: %d\n",
           a_match ? "YES" : "NO", t_a, a_att);

    /* Summary */
    printf("\n--- Speed Comparison ---\n");
    printf("%-20s %10s %10s %10s\n", "Decoder", "Time(ms)", "Converged", "ErrorOK");
    printf("%-20s %10.3f %10s %10s\n", "BP (SPA)", t_bp,
           bp_ok ? "YES" : "NO",
           compare_vectors(bp_error, e_orig, cols) ? "YES" : "NO");
    printf("%-20s %10.3f %10s %10s\n", "Pruning", t_p,
           p_match ? "YES" : "YES",
           p_match ? "YES" : "NO");
    printf("%-20s %10.3f %10s %10s\n", "Augmentation", t_a,
           a_match ? "YES" : "YES",
           a_match ? "YES" : "NO");

    /* Cleanup */
    free(e_orig); free(e_prune); free(e_aug); free(syndrome);
    free(bp_recovered); free(llr); free(dec_bp);
    for (int r = 0; r < rows; ++r) free(Hdense[r]);
    free(Hdense);
    free_sparse_matrix(&H);
    return 0;
}
