// algorithm2 (helper library) - exported alg2 plus configurable alg2_ex

#include "alg2.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__APPLE__) && defined(ENABLE_ALG2_METAL)
#include "metal_bridge.h"
#endif

/*
 * For every column c, count how many rows in supp(column c) also belong to
 * supp(v).  Columns with count 0 are the old "non-intersecting" columns.
 */
static int compute_column_syndrome_intersections(const gf256_t *matrix,
                                                 const gf256_t *v,
                                                 int rows,
                                                 int cols,
                                                 int *intersection_count)
{
#if defined(__APPLE__) && defined(ENABLE_ALG2_METAL)
    if (cols >= 1024 && metal_bridge_init()) {
        int nonint_count = -1;
        if (alg2_metal_compute_column_intersections((const uint8_t *)matrix,
                                                    (const uint8_t *)v,
                                                    rows,
                                                    cols,
                                                    intersection_count,
                                                    &nonint_count)) {
            return nonint_count;
        }
        fprintf(stderr, "[alg2] Metal intersection-count kernel failed; falling back to CPU.\n");
    }
#endif

    int nonint_count = 0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:nonint_count) if (cols >= 512) schedule(static)
#endif
    for (int c = 0; c < cols; ++c) {
        int count = 0;
        for (int r = 0; r < rows; ++r) {
            if (v[r] != 0 && matrix[(size_t)r * (size_t)cols + (size_t)c] != 0) {
                ++count;
            }
        }
        intersection_count[c] = count;
        if (count == 0) {
            ++nonint_count;
        }
    }

    return nonint_count;
}

typedef struct {
    int index;
    int intersections;
} ColumnCandidate;

static int compare_column_candidates(const void *a, const void *b)
{
    const ColumnCandidate *ca = (const ColumnCandidate *)a;
    const ColumnCandidate *cb = (const ColumnCandidate *)b;

    if (ca->intersections != cb->intersections) {
        return ca->intersections - cb->intersections;
    }
    return ca->index - cb->index;
}

/*
 * Build the pool of candidate columns for the added zero constraints.
 * A column is added if its support intersects the syndrome support at most
 * max_syndrome_intersections times.
 */
static int build_potential_indices(const int *intersection_count,
                                   int cols,
                                   int max_syndrome_intersections,
                                   int *potential)
{
    ColumnCandidate *candidates = NULL;
    int count = 0;

    if (max_syndrome_intersections < 0) {
        max_syndrome_intersections = 0;
    }

    candidates = (ColumnCandidate *)malloc((size_t)cols * sizeof(ColumnCandidate));
    if (!candidates) {
        return -1;
    }

    for (int c = 0; c < cols; ++c) {
        if (intersection_count[c] <= max_syndrome_intersections) {
            candidates[count].index = c;
            candidates[count].intersections = intersection_count[c];
            ++count;
        }
    }

    qsort(candidates, (size_t)count, sizeof(ColumnCandidate), compare_column_candidates);

    for (int i = 0; i < count; ++i) {
        potential[i] = candidates[i].index;
    }

    free(candidates);
    return count;
}

/*
 * Construct the augmented, generally overdetermined, system.
 *
 * The original matrix A has size rows x cols. We append added_rows unit
 * constraints x[selected_indices[k]] = 0. If added_rows > cols - rows, the
 * resulting system has more rows than columns.
 */
static void construct_singular_matrix(const gf256_t *A,
                                      gf256_t *output,
                                      int rows,
                                      int cols,
                                      const int *selected_indices,
                                      int added_rows)
{
#ifdef _OPENMP
#pragma omp parallel for if ((size_t)rows * (size_t)cols >= 65536) schedule(static)
#endif
    for (int r = 0; r < rows; ++r) {
        memcpy(output + (size_t)r * (size_t)cols,
               A + (size_t)r * (size_t)cols,
               (size_t)cols * sizeof(gf256_t));
    }

    if (added_rows > 0) {
        memset(output + (size_t)rows * (size_t)cols,
               0,
               (size_t)added_rows * (size_t)cols * sizeof(gf256_t));
    }

    for (int k = 0; k < added_rows; ++k) {
        int out_row = rows + k;
        int basis_col = selected_indices[k];
        output[(size_t)out_row * (size_t)cols + (size_t)basis_col] = 1;
    }
}

static int hamming_weight(const gf256_t *v, int n)
{
    int w = 0;
    for (int i = 0; i < n; ++i) {
        if (v[i] != 0) {
            ++w;
        }
    }
    return w;
}

static int sample_without_replacement(int *selected,
                                      int selected_count,
                                      const int *potential,
                                      int potential_count)
{
    int *work = NULL;

    if (selected_count == 0) {
        return 1;
    }
    if (selected_count > potential_count) {
        return 0;
    }

    work = (int *)malloc((size_t)potential_count * sizeof(int));
    if (!work) {
        return 0;
    }

    for (int i = 0; i < potential_count; ++i) {
        work[i] = potential[i];
    }

    for (int i = 0; i < selected_count; ++i) {
        int j = i + rand() % (potential_count - i);
        int tmp = work[i];
        work[i] = work[j];
        work[j] = tmp;
        selected[i] = work[i];
    }

    free(work);
    return 1;
}

static void swap_rect_rows(gf256_t *A, gf256_t *b, int cols, int row1, int row2)
{
    if (row1 == row2) {
        return;
    }

    gf256_t *r1 = A + (size_t)row1 * (size_t)cols;
    gf256_t *r2 = A + (size_t)row2 * (size_t)cols;

    for (int c = 0; c < cols; ++c) {
        gf256_t tmp = r1[c];
        r1[c] = r2[c];
        r2[c] = tmp;
    }

    gf256_t tb = b[row1];
    b[row1] = b[row2];
    b[row2] = tb;
}

/* CPU rectangular Gauss-Jordan solver. Returns 1 only for consistent full-column-rank systems. */
static int gf256_solve_overdetermined_unique_cpu(gf256_t *M,
                                                 gf256_t *rhs,
                                                 int total_rows,
                                                 int cols,
                                                 gf256_t *x)
{
    int *pivot_row_for_col = NULL;
    int rank = 0;

    if (total_rows < cols) {
        return 0;
    }

    gf256_init();

    pivot_row_for_col = (int *)malloc((size_t)cols * sizeof(int));
    if (!pivot_row_for_col) {
        return 0;
    }
    for (int c = 0; c < cols; ++c) {
        pivot_row_for_col[c] = -1;
    }

    for (int col = 0; col < cols && rank < total_rows; ++col) {
        int pivot = -1;
        for (int r = rank; r < total_rows; ++r) {
            if (M[(size_t)r * (size_t)cols + (size_t)col] != 0) {
                pivot = r;
                break;
            }
        }

        if (pivot < 0) {
            continue;
        }

        swap_rect_rows(M, rhs, cols, rank, pivot);

        gf256_t *pivot_row = M + (size_t)rank * (size_t)cols;
        gf256_t pv = pivot_row[col];
        if (pv == 0) {
            free(pivot_row_for_col);
            return 0;
        }

        if (pv != 1) {
            gf256_t inv = gf256_inv(pv);
            for (int j = col; j < cols; ++j) {
                pivot_row[j] = gf256_mul(pivot_row[j], inv);
            }
            rhs[rank] = gf256_mul(rhs[rank], inv);
        }

#ifdef _OPENMP
#pragma omp parallel for if (total_rows >= 512) schedule(static)
#endif
        for (int r = 0; r < total_rows; ++r) {
            if (r == rank) {
                continue;
            }

            gf256_t *row = M + (size_t)r * (size_t)cols;
            gf256_t factor = row[col];
            if (factor == 0) {
                continue;
            }

            row[col] = 0;
            for (int j = col + 1; j < cols; ++j) {
                row[j] = gf256_sub(row[j], gf256_mul(factor, pivot_row[j]));
            }
            rhs[r] = gf256_sub(rhs[r], gf256_mul(factor, rhs[rank]));
        }

        pivot_row_for_col[col] = rank;
        ++rank;
    }

    /* Check consistency: zero row in M with nonzero rhs means no solution. */
    for (int r = 0; r < total_rows; ++r) {
        int nonzero = 0;
        for (int c = 0; c < cols; ++c) {
            if (M[(size_t)r * (size_t)cols + (size_t)c] != 0) {
                nonzero = 1;
                break;
            }
        }
        if (!nonzero && rhs[r] != 0) {
            free(pivot_row_for_col);
            return 0;
        }
    }

    if (rank < cols) {
        free(pivot_row_for_col);
        return 0;
    }

    for (int c = 0; c < cols; ++c) {
        int pr = pivot_row_for_col[c];
        if (pr < 0) {
            free(pivot_row_for_col);
            return 0;
        }
        x[c] = rhs[pr];
    }

    free(pivot_row_for_col);
    return 1;
}

static int gf256_solve_overdetermined_unique(gf256_t *M,
                                             gf256_t *rhs,
                                             int total_rows,
                                             int cols,
                                             gf256_t *x)
{
#if defined(__APPLE__) && defined(ENABLE_ALG2_METAL)
    if (cols >= 1024 && total_rows >= cols && metal_bridge_init()) {
        int ok = alg2_metal_solve_overdetermined_unique((uint8_t *)M,
                                                        (uint8_t *)rhs,
                                                        total_rows,
                                                        cols,
                                                        (uint8_t *)x);
        if (ok) {
            return 1;
        }
        return 0;
    }
#endif
    return gf256_solve_overdetermined_unique_cpu(M, rhs, total_rows, cols, x);
}

int alg2_ex(const gf256_t *A,
            const gf256_t *v,
            gf256_t *x,
            int rows,
            int cols,
            int max_weight,
            int max_tries,
            int max_syndrome_intersections,
            int extra_added_rows)
{
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned)time(NULL));
        seeded = 1;
    }

    if (!A || !v || !x || rows <= 0 || cols <= 0 || rows > cols ||
        max_weight < 0 || max_tries <= 0) {
        fprintf(stderr, "Invalid arguments to alg2_ex.\n");
        return 0;
    }

    if (max_syndrome_intersections < 0) {
        max_syndrome_intersections = 0;
    }
    if (extra_added_rows < 0) {
        extra_added_rows = 0;
    }

    const int minimum_added_rows = cols - rows;
    const int added_rows = minimum_added_rows + extra_added_rows;
    const int total_rows = rows + added_rows;

    int *intersection_count = (int *)malloc((size_t)cols * sizeof(int));
    int *potential = (int *)malloc((size_t)cols * sizeof(int));
    int *selected = (int *)malloc((size_t)added_rows * sizeof(int));
    gf256_t *M = (gf256_t *)malloc((size_t)total_rows * (size_t)cols * sizeof(gf256_t));
    gf256_t *rhs = (gf256_t *)malloc((size_t)total_rows * sizeof(gf256_t));
    gf256_t *candidate_x = (gf256_t *)malloc((size_t)cols * sizeof(gf256_t));

    if (!intersection_count || !potential || !selected || !M || !rhs || !candidate_x) {
        fprintf(stderr, "Allocation failed in alg2_ex.\n");
        free(intersection_count);
        free(potential);
        free(selected);
        free(M);
        free(rhs);
        free(candidate_x);
        return 0;
    }

    int nonint_count = compute_column_syndrome_intersections(A, v, rows, cols, intersection_count);

    int potential_count = build_potential_indices(intersection_count,
                                                  cols,
                                                  max_syndrome_intersections,
                                                  potential);

    if (potential_count < 0) {
        fprintf(stderr, "Could not build potential index set.\n");
        free(intersection_count);
        free(potential);
        free(selected);
        free(M);
        free(rhs);
        free(candidate_x);
        return 0;
    }

    printf("[alg2] non-intersecting columns: %d\n", nonint_count);
    printf("[alg2] potential columns with intersection <= %d: %d\n",
           max_syndrome_intersections, potential_count);
    printf("[alg2] original rows: %d, cols: %d\n", rows, cols);
    printf("[alg2] added rows: %d = minimum %d + extra %d\n",
           added_rows, minimum_added_rows, extra_added_rows);
    printf("[alg2] constructed system: %d x %d\n", total_rows, cols);

    if (added_rows > potential_count) {
        fprintf(stderr,
                "[alg2] Not enough candidate columns: need %d added rows, have %d candidates.\n",
                added_rows,
                potential_count);
        free(intersection_count);
        free(potential);
        free(selected);
        free(M);
        free(rhs);
        free(candidate_x);
        return 0;
    }

    for (int attempt = 0; attempt < max_tries; ++attempt) {
        if (!sample_without_replacement(selected, added_rows, potential, potential_count)) {
            continue;
        }

        construct_singular_matrix(A, M, rows, cols, selected, added_rows);

        memcpy(rhs, v, (size_t)rows * sizeof(gf256_t));
        if (added_rows > 0) {
            memset(rhs + rows, 0, (size_t)added_rows * sizeof(gf256_t));
        }

        if (!gf256_solve_overdetermined_unique(M, rhs, total_rows, cols, candidate_x)) {
            continue;
        }

        if (hamming_weight(candidate_x, cols) <= max_weight) {
            memcpy(x, candidate_x, (size_t)cols * sizeof(gf256_t));
            free(intersection_count);
            free(potential);
            free(selected);
            free(M);
            free(rhs);
            free(candidate_x);
            return attempt + 1;
        }
    }

    free(intersection_count);
    free(potential);
    free(selected);
    free(M);
    free(rhs);
    free(candidate_x);
    return 0;
}

/* Backward-compatible wrapper. */
int alg2(const gf256_t *A, const gf256_t *v, gf256_t *x,
         int rows, int cols, int max_weight, int max_tries)
{
    return alg2_ex(A,
                   v,
                   x,
                   rows,
                   cols,
                   max_weight,
                   max_tries,
                   0,
                   0);
}

