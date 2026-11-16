#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#include "sparse_utils.h"
#include "gf16.h"
#include "gf256.h"
#include "gf4.h"
#include "lu_solve.h"
#include "alg3prunegf16.h"
#include "alg3prun256.h"
#include "alg3prunegf4.h"
#include "alg3prunebin.h"

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1000.0 * (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e6;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <matrix_file> <field> <error_rate_pct> [threshold_v] [max_weight] [max_tries]\n", argv[0]);
        fprintf(stderr, "  field: 0=GF(2), 4=GF(4), 16=GF(16), 1=GF(256)\n");
        fprintf(stderr, "  Defaults: threshold_v=2, max_weight=error_count+20, max_tries=20\n");
        return 1;
    }

    const char *matrix_file = argv[1];
    int field = atoi(argv[2]);
    double err_pct = atof(argv[3]);
    int thr = (argc >= 5) ? atoi(argv[4]) : 2;
    int max_weight = (argc >= 6) ? atoi(argv[5]) : (int)(2000 * err_pct / 100.0 + 30);
    int max_tries = (argc >= 7) ? atoi(argv[6]) : 20;

    srand(54321);
    SparseMatrix H;
    if (!load_sparse_matrix(matrix_file, &H)) {
        fprintf(stderr, "[test] Failed to load matrix: %s\n", matrix_file);
        return 1;
    }

    printf("========================================================\n");
    printf("alg3 LU Pruning Test (pure C, no Metal)\n");
    printf("========================================================\n");
    printf("Matrix:  %s\n", matrix_file);
    printf("Size:    %d x %d (c=%d, field=%d)\n", H.rows, H.cols, H.c, field);
    printf("Error:   %.2f%%\n", err_pct);
    printf("Params:  threshold_v=%d, max_weight=%d, max_tries=%d\n", thr, max_weight, max_tries);
    printf("--------------------------------------------------------\n");

    /* Generate random error vector */
    int n_err = (int)(H.cols * err_pct / 100.0 + 0.5);
    if (n_err <= 0) n_err = 1;

    uint8_t *e = (uint8_t *)calloc(H.cols, 1);
    int max_val = 1;
    if (field == 1) max_val = 255;
    else if (field == 16) max_val = 15;
    else if (field == 4) max_val = 3;

    for (int p = 0; p < n_err; ) {
        int pos = rand() % H.cols;
        if (!e[pos]) {
            e[pos] = max_val > 1 ? (uint8_t)((rand() % max_val) + 1) : 1;
            p++;
        }
    }

    uint8_t *syndrome = (uint8_t *)malloc(H.rows);
    uint8_t *e_recv = (uint8_t *)calloc(H.cols, 1);
    if (!syndrome || !e_recv) {
        free(e); free(syndrome); free(e_recv);
        free_sparse_matrix(&H);
        return 1;
    }

    /* Init field and compute syndrome = H * e */
    switch (field) {
        case 16: gf16_init();
                 sparse_matvec_gf16(&H, e, syndrome); break;
        case 1:  gf256_init();
                 sparse_matvec_gf256(&H, e, syndrome); break;
        case 4:  gf4_init();
                 sparse_matvec_gf4(&H, e, syndrome); break;
        case 0:  sparse_matvec(&H, e, syndrome); break;
        default:
            fprintf(stderr, "[test] Unknown field %d\n", field);
            free(e); free(syndrome); free(e_recv);
            free_sparse_matrix(&H);
            return 1;
    }

    double t0 = now_ms();
    int attempts = 0;

    switch (field) {
        case 16:
            attempts = alg3_prune_gf16(&H, syndrome, e_recv, thr, max_weight, max_tries);
            break;
        case 1:
            attempts = alg3_prune256(&H, syndrome, e_recv, thr, max_weight, max_tries);
            break;
        case 4:
            attempts = alg3_prune_gf4(&H, syndrome, e_recv, thr, max_weight, max_tries);
            break;
        case 0:
            attempts = alg3_prunebin(&H, syndrome, e_recv, thr, max_weight, max_tries);
            break;
    }

    double elapsed = now_ms() - t0;
    int ok = (attempts > 0 && compare_vectors(e, e_recv, H.cols));

    if (ok) {
        printf("[PASS] Decoded in %.3f ms after %d attempt(s)\n", elapsed, attempts);
    } else {
        printf("[FAIL] Decoding failed after %d attempts (%.3f ms)\n", attempts, elapsed);
    }

    free(e); free(syndrome); free(e_recv);
    free_sparse_matrix(&H);
    return ok ? 0 : 2;
}
