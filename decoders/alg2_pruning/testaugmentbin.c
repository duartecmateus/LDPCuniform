#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sparse_utils.h"
#include "alg2augmentbin.h"

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1000.0 * (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e6;
}

static void generate_sparse_error_binary(uint8_t *e, int size, double error_rate_pct)
{
    int nz = (int)((double)size * error_rate_pct / 100.0 + 0.5);
    if (nz <= 0) nz = 1;
    memset(e, 0, (size_t)size * sizeof(uint8_t));
    int placed = 0;
    while (placed < nz) {
        int pos = rand() % size;
        if (e[pos] != 0) continue;
        e[pos] = 1;
        ++placed;
    }
}

int main(int argc, char **argv)
{
    if (argc != 8 && argc != 9) {
        fprintf(stderr,
                "Usage: %s <matrix_file> <error_rate_pct> <max_intersection> "
                "<extra_added_rows> <max_error_weight> <max_tries> <csv_file> [seed]\n",
                argv[0]);
        return 1;
    }

    const char *matrix_file = argv[1];
    double err_pct = atof(argv[2]);
    int max_inter = atoi(argv[3]);
    int extra_rows = atoi(argv[4]);
    int max_weight = atoi(argv[5]);
    int max_tries = atoi(argv[6]);
    const char *csv_file = argv[7];
    unsigned int seed = (argc == 9) ? (unsigned int)strtoul(argv[8], NULL, 10) : 54321u;

    srand(seed);

    SparseMatrix H;
    if (!load_sparse_matrix(matrix_file, &H)) return 1;
    if (H.field != 0) {
        fprintf(stderr, "[ERROR] Matrix is not binary (field=%d).\n", H.field);
        free_sparse_matrix(&H); return 1;
    }

    printf("[testaugmentbin] Loaded %s, m=%d, n=%d, c=%d\n",
           matrix_file, H.rows, H.cols, H.c);

    uint8_t *e_orig = (uint8_t *)malloc((size_t)H.cols);
    uint8_t *e_recv = (uint8_t *)calloc((size_t)H.cols, 1);
    uint8_t *syndrome = (uint8_t *)malloc((size_t)H.rows);

    generate_sparse_error_binary(e_orig, H.cols, err_pct);
    int orig_w = count_nonzero(e_orig, H.cols);
    sparse_matvec(&H, e_orig, syndrome);
    int synd_w = count_nonzero(syndrome, H.rows);

    printf("[testaugmentbin] errors=%d, syndrome_weight=%d, "
           "max_inter=%d, extra_rows=%d\n",
           orig_w, synd_w, max_inter, extra_rows);

    double start = now_ms();
    int attempts = alg2_augment_bin(&H, syndrome, e_recv,
                                     max_inter, extra_rows,
                                     max_weight, max_tries);
    double end = now_ms();
    double elapsed = end - start;

    int success = (attempts > 0 && compare_vectors(e_orig, e_recv, H.cols));

    /* CSV append */
    {
        int exists = 0;
        FILE *ck = fopen(csv_file, "r");
        if (ck) { exists = 1; fclose(ck); }
        FILE *f = fopen(csv_file, "a");
        if (f) {
            if (!exists)
                fprintf(f, "algorithm,matrix_file,m,n,c,error_rate_pct,max_intersection,"
                           "extra_added_rows,max_weight,max_tries,"
                           "original_weight,syndrome_weight,attempts,time_ms,success\n");
            fprintf(f, "augment_bin,%s,%d,%d,%d,%.6f,%d,%d,%d,%d,%d,%d,%d,%.6f,%d\n",
                    matrix_file, H.rows, H.cols, H.c, err_pct,
                    max_inter, extra_rows, max_weight, max_tries,
                    orig_w, synd_w, attempts, elapsed, success);
            fclose(f);
        }
    }

    printf("algorithm=augment_bin matrix=%s m=%d n=%d c=%d "
           "error_rate=%.4f max_inter=%d extra_rows=%d "
           "max_weight=%d max_tries=%d "
           "attempts=%d time_ms=%.6f success=%d\n",
           matrix_file, H.rows, H.cols, H.c, err_pct,
           max_inter, extra_rows, max_weight, max_tries,
           attempts, elapsed, success);

    free(e_orig); free(e_recv); free(syndrome);
    free_sparse_matrix(&H);
    return success ? 0 : 2;
}
