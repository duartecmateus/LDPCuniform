#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>

#include "gf256.h"
#include "alg2.h"

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1000.0 * (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e6;
}

static int count_nonzero_entries(const gf256_t *v, int size)
{
    int count = 0;
    for (int i = 0; i < size; ++i) {
        if (v[i] != 0) ++count;
    }
    return count;
}

static int compare_vectors(const gf256_t *a, const gf256_t *b, int size)
{
    for (int i = 0; i < size; ++i) {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}

static void generate_sparse_error(gf256_t *e, int size, double error_rate_percent, unsigned int seed)
{
    int nonzero_count = (int)((double)size * error_rate_percent / 100.0 + 0.5);
    if (nonzero_count <= 0) nonzero_count = 1;

    memset(e, 0, (size_t)size * sizeof(gf256_t));

    srand(seed);
    int placed = 0;
    while (placed < nonzero_count) {
        int pos = rand() % size;
        if (e[pos] != 0) continue;
        e[pos] = (gf256_t)((rand() % 255) + 1);
        ++placed;
    }
}

static int parse_column_line(const char *line, int c, int col, int m, gf256_t *A, int n)
{
    const char *p = line;
    int count = 0;

    while (*p != '\0' && *p != '\n' && *p != '\r') {
        while (*p == ' ' || *p == '\t') ++p;
        if (*p == '\0' || *p == '\n' || *p == '\r') break;

        char *endptr = NULL;
        long row = strtol(p, &endptr, 10);
        if (endptr == p || *endptr != ':') return 0;
        p = endptr + 1;

        long value = strtol(p, &endptr, 10);
        if (endptr == p) return 0;
        p = endptr;

        if (row < 0 || row >= m || value <= 0 || value > 255) return 0;
        A[(size_t)row * (size_t)n + (size_t)col] = (gf256_t)value;
        ++count;
    }

    return (count > 0 && count <= c);
}

static int load_sparse_matrix(const char *path, gf256_t **A_out, int *m_out, int *n_out, int *c_out)
{
    FILE *fp = fopen(path, "r");
    if (!fp) return 0;

    int m = 0, n = 0, c = 0;
    if (fscanf(fp, "%d", &m) != 1 ||
        fscanf(fp, "%d", &n) != 1 ||
        fscanf(fp, "%d", &c) != 1 ||
        m <= 0 || n <= 0 || c <= 0) {
        fclose(fp);
        return 0;
    }

    // Skip rest of line 3 AND the entire field-code line (line 4)
    int ch;
    while ((ch = fgetc(fp)) != '\n' && ch != EOF) {}
    while ((ch = fgetc(fp)) != '\n' && ch != EOF) {}

    gf256_t *A = (gf256_t *)calloc((size_t)m * (size_t)n, sizeof(gf256_t));
    if (!A) {
        fclose(fp);
        return 0;
    }

    size_t line_cap = (size_t)c * 24u + 64u;
    char *line = (char *)malloc(line_cap);
    if (!line) {
        free(A);
        fclose(fp);
        return 0;
    }

    for (int col = 0; col < n; ++col) {
        if (!fgets(line, (int)line_cap, fp)) {
            free(line);
            free(A);
            fclose(fp);
            return 0;
        }
        if (!parse_column_line(line, c, col, m, A, n)) {
            free(line);
            free(A);
            fclose(fp);
            return 0;
        }
    }

    free(line);
    fclose(fp);
    *A_out = A;
    *m_out = m;
    *n_out = n;
    *c_out = c;
    return 1;
}

static int csv_exists_and_nonempty(const char *path)
{
    FILE *fp = fopen(path, "r");
    if (!fp) return 0;
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fclose(fp);
    return size > 0;
}

static int append_csv_row(const char *csv_path,
                          const char *matrix_file,
                          int m,
                          int n,
                          int c,
                          double error_rate_percent,
                          int max_syndrome_intersections,
                          int extra_added_rows,
                          int max_error_weight,
                          int max_tries,
                          int original_error_weight,
                          int syndrome_weight,
                          int attempts,
                          double time_ms,
                          int success)
{
    int has_header = csv_exists_and_nonempty(csv_path);
    FILE *fp = fopen(csv_path, "a");
    if (!fp) return 0;

    if (!has_header) {
        fprintf(fp,
                "algorithm,matrix_file,m,n,c,error_rate_percent,max_syndrome_intersections,extra_added_rows,max_error_weight,max_tries,original_error_weight,syndrome_weight,attempts,time_ms,success\n");
    }

    fprintf(fp,
            "parallel,%s,%d,%d,%d,%.6f,%d,%d,%d,%d,%d,%d,%d,%.6f,%d\n",
            matrix_file,
            m,
            n,
            c,
            error_rate_percent,
            max_syndrome_intersections,
            extra_added_rows,
            max_error_weight,
            max_tries,
            original_error_weight,
            syndrome_weight,
            attempts,
            time_ms,
            success);

    fclose(fp);
    return 1;
}

int main(int argc, char **argv)
{
    if (argc < 8 || argc > 9) {
        fprintf(stderr,
                "Usage: %s <matrix_file> <error_rate_percent> <max_syndrome_intersections> <extra_added_rows> <max_error_weight> <max_tries> <csv_file> [seed]\n",
                argv[0]);
        return 1;
    }

    const char *matrix_file = argv[1];
    double error_rate_percent = atof(argv[2]);
    int max_syndrome_intersections = atoi(argv[3]);
    int extra_added_rows = atoi(argv[4]);
    int max_error_weight = atoi(argv[5]);
    int max_tries = atoi(argv[6]);
    const char *csv_file = argv[7];
    unsigned int seed = (argc >= 9) ? (unsigned int)strtoul(argv[8], NULL, 10) : 54321u;

    gf256_init();

    gf256_t *A = NULL;
    int rows = 0, cols = 0, c = 0;
    if (!load_sparse_matrix(matrix_file, &A, &rows, &cols, &c)) {
        fprintf(stderr, "[ERROR] Failed to load matrix from '%s'\n", matrix_file);
        return 1;
    }

    gf256_t *e_orig = (gf256_t *)malloc((size_t)cols * sizeof(gf256_t));
    gf256_t *e_recovered = (gf256_t *)malloc((size_t)cols * sizeof(gf256_t));
    gf256_t *s = (gf256_t *)malloc((size_t)rows * sizeof(gf256_t));
    if (!e_orig || !e_recovered || !s) {
        fprintf(stderr, "[ERROR] Allocation failed\n");
        free(A);
        free(e_orig);
        free(e_recovered);
        free(s);
        return 1;
    }

    generate_sparse_error(e_orig, cols, error_rate_percent, seed);
    memset(e_recovered, 0, (size_t)cols * sizeof(gf256_t));

    int original_error_weight = count_nonzero_entries(e_orig, cols);
    gf256_matvec(A, e_orig, s, rows, cols);
    int syndrome_weight = count_nonzero_entries(s, rows);

    double t0 = now_ms();
    int attempts = alg2_ex(A,
                           s,
                           e_recovered,
                           rows,
                           cols,
                           max_error_weight,
                           max_tries,
                           max_syndrome_intersections,
                           extra_added_rows);
    double t1 = now_ms();
    double elapsed_ms = t1 - t0;

    int success = (attempts > 0) && compare_vectors(e_orig, e_recovered, cols);

    if (!append_csv_row(csv_file,
                        matrix_file,
                        rows,
                        cols,
                        c,
                        error_rate_percent,
                        max_syndrome_intersections,
                        extra_added_rows,
                        max_error_weight,
                        max_tries,
                        original_error_weight,
                        syndrome_weight,
                        attempts,
                        elapsed_ms,
                        success)) {
        fprintf(stderr, "[ERROR] Failed to append results to '%s': %s\n", csv_file, strerror(errno));
        free(A);
        free(e_orig);
        free(e_recovered);
        free(s);
        return 1;
    }

    printf("[RESULT] algorithm=parallel matrix=%s m=%d n=%d c=%d error_rate=%.4f attempts=%d time_ms=%.3f success=%d\n",
           matrix_file, rows, cols, c, error_rate_percent, attempts, elapsed_ms, success);

    free(A);
    free(e_orig);
    free(e_recovered);
    free(s);
    return success ? 0 : 2;
}

// Debug: check if matrix loaded
__attribute__((constructor))
void debug_check() {}
