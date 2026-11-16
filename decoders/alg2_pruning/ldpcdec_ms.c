#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "ldpcdec.h"

/* ============================================================
 *                  BINARY LDPC DECODER (Min-Sum + SPA)
 * ============================================================ */
 
static double sgn(double x) { return x >= 0.0 ? 1.0 : -1.0; }

static double clip_atanh_arg(double x) {
    const double eps = 1e-12;
    if (x >= 1.0) return 1.0 - eps;
    if (x <= -1.0) return -1.0 + eps;
    return x;
}

static int **alloc_matrix_int(int M, int N) {
    int **A = (int**)malloc((size_t)M * sizeof(int*));
    if (!A) return NULL;
    for (int i = 0; i < M; i++) {
        A[i] = (int*)calloc((size_t)N, sizeof(int));
        if (!A[i]) {
            for (int k = 0; k < i; k++) free(A[k]);
            free(A);
            return NULL;
        }
    }
    return A;
}

static void free_matrix_int(int **A, int M) {
    if (!A) return;
    for (int i = 0; i < M; i++) free(A[i]);
    free(A);
}

static int check_syndrome(int M, int N, int **H, const int *x) {
    for (int m = 0; m < M; m++) {
        int s = 0;
        for (int n = 0; n < N; n++) s ^= (H[m][n] & x[n]);
        if (s) return 0;
    }
    return 1;
}

int ldpc_decode_min_sum(int M, int N, int **H,
                               const double *llr_in,
                               int *decoded,
                               int *iters_used) {
    int *cn_deg = (int*)calloc((size_t)M, sizeof(int));
    int *vn_deg = (int*)calloc((size_t)N, sizeof(int));
    int **cn_adj = NULL;
    int **vn_adj = NULL;
    int *cn_pos = NULL, *vn_pos = NULL;
    double **q_nm = NULL, **r_mn = NULL;
    int success = 0;

    if (!cn_deg || !vn_deg) goto cleanup;

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            if (H[m][n]) {
                cn_deg[m]++;
                vn_deg[n]++;
            }
        }
    }

    cn_adj = alloc_matrix_int(M, N);
    vn_adj = alloc_matrix_int(N, M);
    cn_pos = (int*)calloc((size_t)M, sizeof(int));
    vn_pos = (int*)calloc((size_t)N, sizeof(int));
    if (!cn_adj || !vn_adj || !cn_pos || !vn_pos) goto cleanup;

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            if (H[m][n]) {
                cn_adj[m][cn_pos[m]++] = n;
                vn_adj[n][vn_pos[n]++] = m;
            }
        }
    }

    q_nm = (double**)malloc((size_t)N * sizeof(double*));
    r_mn = (double**)malloc((size_t)M * sizeof(double*));
    if (!q_nm || !r_mn) goto cleanup;

    for (int n = 0; n < N; n++) {
        q_nm[n] = (double*)calloc((size_t)M, sizeof(double));
        if (!q_nm[n]) goto cleanup;
    }
    for (int m = 0; m < M; m++) {
        r_mn[m] = (double*)calloc((size_t)N, sizeof(double));
        if (!r_mn[m]) goto cleanup;
    }

    for (int n = 0; n < N; n++) {
        for (int i = 0; i < vn_deg[n]; i++) {
            int m = vn_adj[n][i];
            q_nm[n][m] = llr_in[n];
        }
    }

    for (int it = 0; it < MAX_ITERS; it++) {
        for (int m = 0; m < M; m++) {
            for (int i = 0; i < cn_deg[m]; i++) {
                int n = cn_adj[m][i];
                double prod = 1.0;
                double minabs = 1e30;

                for (int j = 0; j < cn_deg[m]; j++) {
                    if (j == i) continue;
                    int n2 = cn_adj[m][j];
                    double val = q_nm[n2][m];
                    prod *= sgn(val);
                    double a = fabs(val);
                    if (a < minabs) minabs = a;
                }

                if (cn_deg[m] <= 1) minabs = 0.0;
                r_mn[m][n] = prod * minabs;
            }
        }

        for (int n = 0; n < N; n++) {
            double sum = llr_in[n];
            for (int i = 0; i < vn_deg[n]; i++) {
                int m = vn_adj[n][i];
                sum += r_mn[m][n];
            }
            decoded[n] = (sum < 0.0) ? 1 : 0;
        }

        if (check_syndrome(M, N, H, decoded)) {
            success = 1;
            if (iters_used) *iters_used = it + 1;
            goto cleanup;
        }

        for (int n = 0; n < N; n++) {
            for (int i = 0; i < vn_deg[n]; i++) {
                int m = vn_adj[n][i];
                double sum = llr_in[n];
                for (int j = 0; j < vn_deg[n]; j++) {
                    int m2 = vn_adj[n][j];
                    if (m2 == m) continue;
                    sum += r_mn[m2][n];
                }
                q_nm[n][m] = sum;
            }
        }
    }

    if (iters_used) *iters_used = MAX_ITERS;

cleanup:
    if (q_nm) {
        for (int n = 0; n < N; n++) free(q_nm[n]);
        free(q_nm);
    }
    if (r_mn) {
        for (int m = 0; m < M; m++) free(r_mn[m]);
        free(r_mn);
    }
    free_matrix_int(cn_adj, M);
    free_matrix_int(vn_adj, N);
    free(cn_pos);
    free(vn_pos);
    free(cn_deg);
    free(vn_deg);
    return success;
}

int ldpc_decode_spa(int M, int N, int **H,
                        const double *llr_in,
                        int *decoded,
                        int *iters_used) {
    int *cn_deg = (int*)calloc((size_t)M, sizeof(int));
    int *vn_deg = (int*)calloc((size_t)N, sizeof(int));
    int **cn_adj = NULL;
    int **vn_adj = NULL;
    int *cn_pos = NULL, *vn_pos = NULL;
    double **q_nm = NULL, **r_mn = NULL;
    int success = 0;

    if (!cn_deg || !vn_deg) goto cleanup;

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            if (H[m][n]) {
                cn_deg[m]++;
                vn_deg[n]++;
            }
        }
    }

    cn_adj = alloc_matrix_int(M, N);
    vn_adj = alloc_matrix_int(N, M);
    cn_pos = (int*)calloc((size_t)M, sizeof(int));
    vn_pos = (int*)calloc((size_t)N, sizeof(int));
    if (!cn_adj || !vn_adj || !cn_pos || !vn_pos) goto cleanup;

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            if (H[m][n]) {
                cn_adj[m][cn_pos[m]++] = n;
                vn_adj[n][vn_pos[n]++] = m;
            }
        }
    }

    q_nm = (double**)malloc((size_t)N * sizeof(double*));
    r_mn = (double**)malloc((size_t)M * sizeof(double*));
    if (!q_nm || !r_mn) goto cleanup;

    for (int n = 0; n < N; n++) {
        q_nm[n] = (double*)calloc((size_t)M, sizeof(double));
        if (!q_nm[n]) goto cleanup;
    }
    for (int m = 0; m < M; m++) {
        r_mn[m] = (double*)calloc((size_t)N, sizeof(double));
        if (!r_mn[m]) goto cleanup;
    }

    for (int n = 0; n < N; n++) {
        for (int i = 0; i < vn_deg[n]; i++) {
            int m = vn_adj[n][i];
            q_nm[n][m] = llr_in[n];
        }
    }

    for (int it = 0; it < MAX_ITERS; it++) {
        for (int m = 0; m < M; m++) {
            for (int i = 0; i < cn_deg[m]; i++) {
                int n = cn_adj[m][i];
                double prod = 1.0;

                for (int j = 0; j < cn_deg[m]; j++) {
                    if (j == i) continue;
                    int n2 = cn_adj[m][j];
                    double val = q_nm[n2][m];
                    prod *= tanh(0.5 * val);
                }

                prod = clip_atanh_arg(prod);
                r_mn[m][n] = 2.0 * atanh(prod);
            }
        }

        for (int n = 0; n < N; n++) {
            double sum = llr_in[n];
            for (int i = 0; i < vn_deg[n]; i++) {
                int m = vn_adj[n][i];
                sum += r_mn[m][n];
            }
            decoded[n] = (sum < 0.0) ? 1 : 0;
        }

        if (check_syndrome(M, N, H, decoded)) {
            success = 1;
            if (iters_used) *iters_used = it + 1;
            goto cleanup;
        }

        for (int n = 0; n < N; n++) {
            for (int i = 0; i < vn_deg[n]; i++) {
                int m = vn_adj[n][i];
                double sum = llr_in[n];
                for (int j = 0; j < vn_deg[n]; j++) {
                    int m2 = vn_adj[n][j];
                    if (m2 == m) continue;
                    sum += r_mn[m2][n];
                }
                q_nm[n][m] = sum;
            }
        }
    }

    if (iters_used) *iters_used = MAX_ITERS;

cleanup:
    if (q_nm) {
        for (int n = 0; n < N; n++) free(q_nm[n]);
        free(q_nm);
    }
    if (r_mn) {
        for (int m = 0; m < M; m++) free(r_mn[m]);
        free(r_mn);
    }
    free_matrix_int(cn_adj, M);
    free_matrix_int(vn_adj, N);
    free(cn_pos);
    free(vn_pos);
    free(cn_deg);
    free(vn_deg);
    return success;
}