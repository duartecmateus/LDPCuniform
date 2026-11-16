#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <limits.h>

#include "gf256.h"
#include "gf16.h"
#include "gf4.h"

#ifndef DEFAULT_TRIALS_PER_COLUMN
#define DEFAULT_TRIALS_PER_COLUMN 32
#endif

#ifndef DEFAULT_PROGRESS_STEP
#define DEFAULT_PROGRESS_STEP 100
#endif

static int compare_ints(const void *a, const void *b)
{
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    return (ia > ib) - (ia < ib);
}

static gf256_t random_nonzero_gf256(void)
{
    gf256_t v = 0;
    while (v == 0) {
        v = (gf256_t)(rand() & 0xFF);
    }
    return v;
}

static size_t pair_index(int a, int b, int m)
{
    size_t aa = (size_t)a;
    size_t bb = (size_t)b;
    size_t mm = (size_t)m;
    return aa * mm + bb;
}

static void sample_random_support(int *support, int m, int c)
{
    int *perm = (int *)malloc((size_t)m * sizeof(int));
    if (!perm) {
        fprintf(stderr, "sample_random_support: allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < m; ++i) {
        perm[i] = i;
    }

    for (int i = 0; i < c; ++i) {
        int j = i + rand() % (m - i);
        int tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
        support[i] = perm[i];
    }

    qsort(support, (size_t)c, sizeof(int), compare_ints);
    free(perm);
}

static int intersection_size_sorted(const int *a, const int *b, int c)
{
    int i = 0;
    int j = 0;
    int count = 0;

    while (i < c && j < c) {
        if (a[i] == b[j]) {
            ++count;
            ++i;
            ++j;
        } else if (a[i] < b[j]) {
            ++i;
        } else {
            ++j;
        }
    }

    return count;
}

static long long evaluate_candidate(const int *candidate,
                                    const int *selected_supports,
                                    int selected_count,
                                    int m,
                                    int c,
                                    const int *row_loads,
                                    const uint16_t *pair_loads,
                                    int *candidate_max_intersection,
                                    long long *candidate_sum_intersection,
                                    long long *candidate_pair_score,
                                    long long *candidate_row_score)
{
    int max_intersection = 0;
    long long sum_intersection = 0;
    long long pair_score = 0;
    long long row_score = 0;

    for (int i = 0; i < c; ++i) {
        int r = candidate[i];
        long long load = row_loads[r];
        row_score += load * load;
    }

    for (int i = 0; i < c; ++i) {
        for (int j = i + 1; j < c; ++j) {
            pair_score += pair_loads[pair_index(candidate[i], candidate[j], m)];
        }
    }

    for (int col = 0; col < selected_count; ++col) {
        const int *prev = selected_supports + (size_t)col * (size_t)c;
        int inter = intersection_size_sorted(candidate, prev, c);
        if (inter > max_intersection) {
            max_intersection = inter;
        }
        sum_intersection += inter;
    }

    if (candidate_max_intersection) *candidate_max_intersection = max_intersection;
    if (candidate_sum_intersection) *candidate_sum_intersection = sum_intersection;
    if (candidate_pair_score) *candidate_pair_score = pair_score;
    if (candidate_row_score) *candidate_row_score = row_score;

    return ((long long)max_intersection << 50)
         + (sum_intersection << 24)
         + (pair_score << 8)
         + row_score;
}

static void update_bookkeeping(const int *support,
                               int c,
                               int *row_loads,
                               uint16_t *pair_loads,
                               int m)
{
    for (int i = 0; i < c; ++i) {
        row_loads[support[i]]++;
    }

    for (int i = 0; i < c; ++i) {
        for (int j = i + 1; j < c; ++j) {
            size_t idx = pair_index(support[i], support[j], m);
            if (pair_loads[idx] != UINT16_MAX) {
                pair_loads[idx]++;
            }
        }
    }
}

static int choose_supports_randomized(int m,
                                      int n,
                                      int c,
                                      int threshold,
                                      int trials_per_column,
                                      int *selected_supports)
{
    int *row_loads = (int *)calloc((size_t)m, sizeof(int));
    uint16_t *pair_loads = (uint16_t *)calloc((size_t)m * (size_t)m, sizeof(uint16_t));
    int *candidate = (int *)malloc((size_t)c * sizeof(int));
    int *best_candidate = (int *)malloc((size_t)c * sizeof(int));

    if (!row_loads || !pair_loads || !candidate || !best_candidate) {
        fprintf(stderr, "choose_supports_randomized: allocation failed\n");
        free(row_loads);
        free(pair_loads);
        free(candidate);
        free(best_candidate);
        return 0;
    }

    for (int col = 0; col < n; ++col) {
        long long best_score = LLONG_MAX;
        int found_feasible = 0;
        int best_max_intersection = INT_MAX;
        long long best_sum_intersection = LLONG_MAX;
        long long best_pair_score = LLONG_MAX;
        long long best_row_score = LLONG_MAX;

        int local_trials = trials_per_column;
        if (col == 0) {
            local_trials = 1;
        }

        for (int t = 0; t < local_trials; ++t) {
            int max_intersection = 0;
            long long sum_intersection = 0;
            long long pair_score = 0;
            long long row_score = 0;
            long long score;

            sample_random_support(candidate, m, c);
            score = evaluate_candidate(candidate,
                                       selected_supports,
                                       col,
                                       m,
                                       c,
                                       row_loads,
                                       pair_loads,
                                       &max_intersection,
                                       &sum_intersection,
                                       &pair_score,
                                       &row_score);

            if (max_intersection <= threshold) {
                if (!found_feasible || score < best_score) {
                    found_feasible = 1;
                    best_score = score;
                    best_max_intersection = max_intersection;
                    best_sum_intersection = sum_intersection;
                    best_pair_score = pair_score;
                    best_row_score = row_score;
                    memcpy(best_candidate, candidate, (size_t)c * sizeof(int));
                }
            } else if (!found_feasible && score < best_score) {
                best_score = score;
                best_max_intersection = max_intersection;
                best_sum_intersection = sum_intersection;
                best_pair_score = pair_score;
                best_row_score = row_score;
                memcpy(best_candidate, candidate, (size_t)c * sizeof(int));
            }
        }

        if (!found_feasible && best_max_intersection > threshold) {
            fprintf(stderr,
                    "Warning: no candidate satisfied threshold at column %d; using best candidate with max intersection %d\n",
                    col,
                    best_max_intersection);
        }

        memcpy(selected_supports + (size_t)col * (size_t)c,
               best_candidate,
               (size_t)c * sizeof(int));
        update_bookkeeping(best_candidate, c, row_loads, pair_loads, m);

        if ((col + 1) % DEFAULT_PROGRESS_STEP == 0 || col + 1 == n) {
            printf("Generated %d / %d columns (max_inter=%d, sum_inter=%lld, pair_score=%lld, row_score=%lld)\n",
                   col + 1,
                   n,
                   best_max_intersection,
                   best_sum_intersection,
                   best_pair_score,
                   best_row_score);
        }
    }

    free(row_loads);
    free(pair_loads);
    free(candidate);
    free(best_candidate);
    return 1;
}

static int write_matrix_text(const char *filename,
                             const int *selected_supports,
                             int m,
                             int n,
                             int c,
                             int field)
{
    FILE *f = fopen(filename, "w");
    if (!f) {
        return 0;
    }

    fprintf(f, "%d\n", m);
    fprintf(f, "%d\n", n);
    fprintf(f, "%d\n", c);
    fprintf(f, "%d\n", field);

    for (int col = 0; col < n; ++col) {
        const int *support = selected_supports + (size_t)col * (size_t)c;
        for (int k = 0; k < c; ++k) {
            int row = support[k];
            unsigned int value;
            if (field == 0) {
                value = 1;
            } else if (field == 2) {
                value = (unsigned)(rand() % 15 + 1);
            } else if (field == 3) {
                value = (unsigned)(rand() % 3 + 1);
            } else {
                value = (unsigned)random_nonzero_gf256();
            }
            fprintf(f, "%d:%u", row, value);
            if (k + 1 < c) {
                fputc(' ', f);
            }
        }
        if (col + 1 < n) {
            fputc('\n', f);
        }
    }

    fclose(f);
    return 1;
}

int main(int argc, char **argv)
{
    int m = 3000;
    int n = 4000;
    int c = 30;
    int threshold = 3;
    int trials_per_column = DEFAULT_TRIALS_PER_COLUMN;
    int field = 1; /* 0 = GF(2), 1 = GF(256) */
    const char *output_file = "H_unif_matrix.txt";

    if (argc >= 5) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        c = atoi(argv[3]);
        threshold = atoi(argv[4]);
    }
    if (argc >= 6) {
        output_file = argv[5];
    }
    if (argc >= 7) {
        trials_per_column = atoi(argv[6]);
    }
    if (argc >= 8) {
        field = atoi(argv[7]);
    }

    if (field != 0 && field != 1 && field != 2 && field != 3) {
        fprintf(stderr, "Invalid field. Use 0 for GF(2), 1 for GF(256), 2 for GF(16), or 3 for GF(4).\n");
        return 1;
    }

    if (m <= 0 || n <= 0 || c <= 0 || c > m) {
        fprintf(stderr, "Invalid parameters. Require m > 0, n > 0, 0 < c <= m.\n");
        return 1;
    }
    if (trials_per_column <= 0) {
        fprintf(stderr, "Invalid trials_per_column. Must be > 0.\n");
        return 1;
    }

    int *selected_supports = (int *)malloc((size_t)n * (size_t)c * sizeof(int));
    if (!selected_supports) {
        fprintf(stderr, "Allocation failed.\n");
        free(selected_supports);
        return 1;
    }

    srand((unsigned)time(NULL));

    const char *field_name;
    if (field == 0) field_name = "GF(2)";
    else if (field == 2) field_name = "GF(16)";
    else if (field == 3) field_name = "GF(4)";
    else field_name = "GF(256)";
    printf("Generating %s matrix with m=%d, n=%d, c=%d, threshold=%d, trials=%d\n",
           field_name, m, n, c, threshold, trials_per_column);

    if (field == 1) {
        gf256_init();
    } else if (field == 2) {
        gf16_init();
    } else if (field == 3) {
        gf4_init();
    }

    if (!choose_supports_randomized(m, n, c, threshold, trials_per_column, selected_supports)) {
        free(selected_supports);
        return 1;
    }

    if (!write_matrix_text(output_file, selected_supports, m, n, c, field)) {
        fprintf(stderr, "Failed to write output file: %s\n", output_file);
        free(selected_supports);
        return 1;
    }

    printf("Generated matrix stored in %s\n", output_file);

    free(selected_supports);
    return 0;
}
