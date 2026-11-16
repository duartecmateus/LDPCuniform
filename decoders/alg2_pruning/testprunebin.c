#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>

#include "sparse_utils.h"
#include "alg2prunebin.h"

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1000.0 * (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e6;
}

static void generate_sparse_error_binary(uint8_t *e, int size, double error_rate_percent)
{
    int nonzero_count = (int)((double)size * error_rate_percent / 100.0 + 0.5);
    if (nonzero_count <= 0) nonzero_count = 1;

    memset(e, 0, (size_t)size * sizeof(uint8_t));

    int placed = 0;
    while (placed < nonzero_count) {
        int pos = rand() % size;
        if (e[pos] != 0) continue;
        e[pos] = 1;
        ++placed;
    }
}

static int append_results_csv(const char *csv_file,
                              const char *matrix_file,
                              int rows, int cols, int c,
                              double error_rate_percent,
                              int threshold_v,
                              int max_error_weight,
                              int max_tries,
                              int original_error_weight,
                              int syndrome_weight,
                              int attempts,
                              double elapsed_ms,
                              int success)
{
    int file_exists = 0;
    FILE *check = fopen(csv_file, "r");
    if (check) { file_exists = 1; fclose(check); }

    FILE *f = fopen(csv_file, "a");
    if (!f) { fprintf(stderr, "[ERROR] Cannot open CSV '%s'\n", csv_file); return 0; }

    if (!file_exists)
        fprintf(f, "algorithm,matrix_file,m,n,c,error_rate_percent,threshold_v,max_error_weight,max_tries,original_error_weight,syndrome_weight,attempts,time_ms,success\n");

    fprintf(f, "prune_bin,%s,%d,%d,%d,%.6f,%d,%d,%d,%d,%d,%d,%.6f,%d\n",
            matrix_file, rows, cols, c,
            error_rate_percent,
            threshold_v, max_error_weight, max_tries,
            original_error_weight, syndrome_weight,
            attempts, elapsed_ms, success);
    fclose(f);
    return 1;
}

int main(int argc, char **argv)
{
    if (argc != 7 && argc != 8) {
        fprintf(stderr,
                "Usage: %s <matrix_file> <error_rate_percent> <threshold_v> <max_error_weight> <max_tries> <csv_file> [seed]\n",
                argv[0]);
        return 1;
    }

    const char *matrix_file = argv[1];
    double error_rate_percent = atof(argv[2]);
    int threshold_v = atoi(argv[3]);
    int max_error_weight = atoi(argv[4]);
    int max_tries = atoi(argv[5]);
    const char *csv_file = argv[6];
    unsigned int seed = (argc == 8) ? (unsigned int)strtoul(argv[7], NULL, 10) : 54321u;

    if (error_rate_percent <= 0.0 || threshold_v < 0 || max_error_weight <= 0 || max_tries <= 0) {
        fprintf(stderr, "[ERROR] Invalid parameters.\n");
        return 1;
    }

    srand(seed);

    SparseMatrix H;
    if (!load_sparse_matrix(matrix_file, &H)) {
        return 1;
    }

    if (H.field != 0) {
        fprintf(stderr, "[ERROR] Matrix '%s' is not binary (field=%d). Use testprune for GF(256) matrices.\n",
                matrix_file, H.field);
        free_sparse_matrix(&H);
        return 1;
    }

    printf("[testprunebin] Loaded matrix: %s, m=%d, n=%d, c=%d, field=GF(2)\n",
           matrix_file, H.rows, H.cols, H.c);

    uint8_t *e_orig = (uint8_t *)malloc((size_t)H.cols * sizeof(uint8_t));
    uint8_t *e_recv = (uint8_t *)calloc((size_t)H.cols, sizeof(uint8_t));
    uint8_t *syndrome = (uint8_t *)malloc((size_t)H.rows * sizeof(uint8_t));

    if (!e_orig || !e_recv || !syndrome) {
        fprintf(stderr, "[ERROR] Allocation failed.\n");
        free(e_orig); free(e_recv); free(syndrome);
        free_sparse_matrix(&H);
        return 1;
    }

    generate_sparse_error_binary(e_orig, H.cols, error_rate_percent);
    int original_error_weight = count_nonzero(e_orig, H.cols);

    /* Compute syndrome: s = H * e_orig (binary version) */
    sparse_matvec(&H, e_orig, syndrome);
    int syndrome_weight = count_nonzero(syndrome, H.rows);

    printf("[testprunebin] error_weight=%d, syndrome_weight=%d, threshold_v=%d\n",
           original_error_weight, syndrome_weight, threshold_v);

    double start = now_ms();
    int attempts = alg2_prunebin(&H, syndrome, e_recv,
                                  threshold_v, max_error_weight, max_tries);
    double end = now_ms();
    double elapsed_ms = end - start;

    int success = (attempts > 0 && compare_vectors(e_orig, e_recv, H.cols));

    if (!append_results_csv(csv_file, matrix_file,
                            H.rows, H.cols, H.c,
                            error_rate_percent,
                            threshold_v,
                            max_error_weight, max_tries,
                            original_error_weight, syndrome_weight,
                            attempts, elapsed_ms, success)) {
        free(e_orig); free(e_recv); free(syndrome);
        free_sparse_matrix(&H);
        return 1;
    }

    printf("algorithm=prune_bin matrix=%s m=%d n=%d c=%d "
           "error_rate=%.4f threshold_v=%d max_weight=%d max_tries=%d "
           "attempts=%d time_ms=%.6f success=%d\n",
           matrix_file, H.rows, H.cols, H.c,
           error_rate_percent, threshold_v, max_error_weight, max_tries,
           attempts, elapsed_ms, success);

    free(e_orig); free(e_recv); free(syndrome);
    free_sparse_matrix(&H);
    return success ? 0 : 2;
}
