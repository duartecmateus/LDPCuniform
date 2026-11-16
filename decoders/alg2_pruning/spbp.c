/* Sparse BP decoder implementation.
   Builds adjacency directly from SparseMatrix (column-oriented).
   Avoids dense matrix representation entirely.
*/

#include "spbp.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1000.0 * (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e6;
}

/* Clip tanh argument to avoid numerical overflow */
static inline double clip_atanh_arg(double x)
{
    if (x >= 1.0 - 1e-15) return 1.0 - 1e-15;
    if (x <= -1.0 + 1e-15) return -1.0 + 1e-15;
    return x;
}

/* Build check-node and variable-node adjacency lists from sparse column matrix.
   cn_adj[m][k] = variable node index for k-th edge of check m
   vn_adj[n][k] = check node index for k-th edge of variable n
   Max cols: rows × n (worst-case density), but we use sparse degree arrays.

   Returns 1 on success, 0 on alloc failure.
   Caller must free cn_adj/vn_adj arrays and deg arrays.
*/
static int build_adjacency(const SparseMatrix *H,
                           int ***cn_adj_out, int **cn_deg_out,
                           int ***vn_adj_out)
{
    int m = H->rows, n = H->cols;

    /* Count check-node degrees */
    int *cn_deg = (int *)calloc(m, sizeof(int));
    if (!cn_deg) return 0;

    for (int j = 0; j < n; j++) {
        for (int k = 0; k < H->col[j].c; k++) {
            int r = H->col[j].rows[k];
            cn_deg[r]++;
        }
    }

    /* Allocate adjacency arrays */
    int **cn_adj = (int **)malloc(m * sizeof(int *));
    if (!cn_adj) { free(cn_deg); return 0; }

    for (int r = 0; r < m; r++) {
        cn_adj[r] = (int *)malloc(cn_deg[r] * sizeof(int));
        if (!cn_adj[r]) {
            for (int t = 0; t < r; t++) free(cn_adj[t]);
            free(cn_adj); free(cn_deg);
            return 0;
        }
    }

    /* Allocate variable-node adjacency from column weights (all c) */
    int **vn_adj = (int **)malloc(n * sizeof(int *));
    if (!vn_adj) {
        for (int r = 0; r < m; r++) free(cn_adj[r]);
        free(cn_adj); free(cn_deg);
        return 0;
    }
    for (int j = 0; j < n; j++) {
        int cw = H->col[j].c;
        vn_adj[j] = (int *)malloc(cw * sizeof(int));
        if (!vn_adj[j]) {
            for (int t = 0; t < j; t++) free(vn_adj[t]);
            free(vn_adj);
            for (int r = 0; r < m; r++) free(cn_adj[r]);
            free(cn_adj); free(cn_deg);
            return 0;
        }
    }

    /* Fill adjacency — reset cn_deg to use as position counter */
    memset(cn_deg, 0, m * sizeof(int));
    for (int j = 0; j < n; j++) {
        for (int k = 0; k < H->col[j].c; k++) {
            int r = H->col[j].rows[k];
            cn_adj[r][cn_deg[r]++] = j;
            vn_adj[j][k] = r;
        }
    }

    *cn_adj_out = cn_adj;
    *cn_deg_out = cn_deg;
    *vn_adj_out = vn_adj;
    return 1;
}

static int check_syndrome_sparse(const SparseMatrix *H, const int *decoded)
{
    int m = H->rows;
    for (int r = 0; r < m; r++) {
        int sum = 0;
        for (int k = 0; k < H->col[0].c; k++) {
            /* Can't use col-wise iteration efficiently for syndrome check.
               We'll use the check-node adjacency. */
        }
    }
    /* Recompute using column-oriented access */
    uint8_t *tmp = (uint8_t *)calloc(m, 1);
    if (!tmp) return 0;
    for (int j = 0; j < H->cols; j++) {
        if (!decoded[j]) continue;
        for (int k = 0; k < H->col[j].c; k++) {
            int r = H->col[j].rows[k];
            tmp[r] ^= 1;
        }
    }
    int ok = 1;
    for (int r = 0; r < m; r++) {
        if (tmp[r]) { ok = 0; break; }
    }
    free(tmp);
    return ok;
}

BPResult spbp_decode_spa(const SparseMatrix *H, const double *llr_in,
                          int max_iters)
{
    BPResult res = {0, NULL, 0, 0.0};
    int m = H->rows, n = H->cols;

    int *cn_deg = NULL;
    int **cn_adj = NULL, **vn_adj = NULL;
    if (!build_adjacency(H, &cn_adj, &cn_deg, &vn_adj))
        return res;

    /* Allocate messages: vn->cn (q_nm) and cn->vn (r_mn) */
    /* Total messages = total edges */
    int total_edges = 0;
    for (int r = 0; r < m; r++) total_edges += cn_deg[r];

    /* We'll store messages as flat arrays indexed by edge position.
       For simplicity, use double arrays indexed by (vn, pos_in_vn). */
    double **q_nm = (double **)malloc(n * sizeof(double *));
    double **r_mn = (double **)malloc(m * sizeof(double *));
    if (!q_nm || !r_mn) goto cleanup;

    for (int j = 0; j < n; j++) {
        int cw = H->col[j].c;
        q_nm[j] = (double *)calloc(cw, sizeof(double));
        if (!q_nm[j]) goto cleanup;
        for (int k = 0; k < cw; k++)
            q_nm[j][k] = llr_in[j];
    }
    for (int r = 0; r < m; r++) {
        r_mn[r] = (double *)calloc(cn_deg[r], sizeof(double));
        if (!r_mn[r]) goto cleanup;
    }

    /* Decoded array */
    res.decoded = (int *)calloc(n, sizeof(int));
    if (!res.decoded) goto cleanup;

    double t0 = now_ms();

    /* SPA iteration loop */
    for (int it = 0; it < max_iters; it++) {
        /* Check node update: compute r_mn from q_nm */
        for (int r = 0; r < m; r++) {
            int deg = cn_deg[r];
            for (int i = 0; i < deg; i++) {
                int nv = cn_adj[r][i];
                double prod = 1.0;
                for (int j = 0; j < deg; j++) {
                    if (j == i) continue;
                    int nv2 = cn_adj[r][j];
                    /* Find position of r in nv2's adjacency */
                    int pos = -1;
                    for (int k = 0; k < H->col[nv2].c; k++) {
                        if (vn_adj[nv2][k] == r) { pos = k; break; }
                    }
                    double val = q_nm[nv2][pos];
                    prod *= tanh(0.5 * val);
                }
                prod = clip_atanh_arg(prod);
                r_mn[r][i] = 2.0 * atanh(prod);
            }
        }

        /* Variable node update: compute q_nm and decoded */
        for (int j = 0; j < n; j++) {
            double sum = llr_in[j];
            for (int k = 0; k < H->col[j].c; k++) {
                int r = vn_adj[j][k];
                /* Find position of j in r's adjacency */
                int pos = -1;
                for (int i = 0; i < cn_deg[r]; i++) {
                    if (cn_adj[r][i] == j) { pos = i; break; }
                }
                sum += r_mn[r][pos];
            }
            res.decoded[j] = (sum < 0.0) ? 1 : 0;

            /* Update q_nm */
            for (int k = 0; k < H->col[j].c; k++) {
                int r = vn_adj[j][k];
                double s = llr_in[j];
                for (int t = 0; t < H->col[j].c; t++) {
                    if (t == k) continue;
                    int r2 = vn_adj[j][t];
                    int pos = -1;
                    for (int i = 0; i < cn_deg[r2]; i++) {
                        if (cn_adj[r2][i] == j) { pos = i; break; }
                    }
                    s += r_mn[r2][pos];
                }
                q_nm[j][k] = s;
            }
        }

        /* Check syndrome */
        if (check_syndrome_sparse(H, res.decoded)) {
            res.converged = 1;
            res.iters_used = it + 1;
            goto cleanup;
        }
    }

    res.iters_used = max_iters;
    res.converged = 0;

cleanup:
    res.time_ms = now_ms() - t0;

    if (q_nm) {
        for (int j = 0; j < n; j++) free(q_nm[j]);
        free(q_nm);
    }
    if (r_mn) {
        for (int r = 0; r < m; r++) free(r_mn[r]);
        free(r_mn);
    }
    if (cn_adj) {
        for (int r = 0; r < m; r++) free(cn_adj[r]);
        free(cn_adj);
    }
    if (vn_adj) {
        for (int j = 0; j < n; j++) free(vn_adj[j]);
        free(vn_adj);
    }
    free(cn_deg);

    return res;
}

BPResult spbp_decode_ms(const SparseMatrix *H, const double *llr_in,
                         int max_iters)
{
    BPResult res = {0, NULL, 0, 0.0};
    int m = H->rows, n = H->cols;

    int *cn_deg = NULL;
    int **cn_adj = NULL, **vn_adj = NULL;
    if (!build_adjacency(H, &cn_adj, &cn_deg, &vn_adj))
        return res;

    double **q_nm = (double **)malloc(n * sizeof(double *));
    double **r_mn = (double **)malloc(m * sizeof(double *));
    if (!q_nm || !r_mn) goto cleanup;

    for (int j = 0; j < n; j++) {
        int cw = H->col[j].c;
        q_nm[j] = (double *)calloc(cw, sizeof(double));
        if (!q_nm[j]) goto cleanup;
        for (int k = 0; k < cw; k++)
            q_nm[j][k] = llr_in[j];
    }
    for (int r = 0; r < m; r++) {
        r_mn[r] = (double *)calloc(cn_deg[r], sizeof(double));
        if (!r_mn[r]) goto cleanup;
    }

    res.decoded = (int *)calloc(n, sizeof(int));
    if (!res.decoded) goto cleanup;

    double t0 = now_ms();

    for (int it = 0; it < max_iters; it++) {
        /* Check node update (Min-Sum) */
        for (int r = 0; r < m; r++) {
            int deg = cn_deg[r];
            /* Find min1, min2 magnitudes and overall sign */
            double min1 = 1e100, min2 = 1e100;
            int min1_sign = 1; /* product of signs ignoring min1 position */
            int total_sign = 1;

            for (int i = 0; i < deg; i++) {
                int nv = cn_adj[r][i];
                int pos = -1;
                for (int k = 0; k < H->col[nv].c; k++) {
                    if (vn_adj[nv][k] == r) { pos = k; break; }
                }
                double mag = fabs(q_nm[nv][pos]);
                int sgn = (q_nm[nv][pos] >= 0) ? 1 : -1;
                total_sign *= sgn;

                if (mag < min1) { min2 = min1; min1 = mag; }
                else if (mag < min2) { min2 = mag; }
            }

            for (int i = 0; i < deg; i++) {
                int nv = cn_adj[r][i];
                int pos = -1;
                for (int k = 0; k < H->col[nv].c; k++) {
                    if (vn_adj[nv][k] == r) { pos = k; break; }
                }
                double mag = fabs(q_nm[nv][pos]);
                int sgn = (q_nm[nv][pos] >= 0) ? 1 : -1;
                int out_sgn = total_sign * sgn;
                double out_mag = (mag == min1) ? min2 : min1;
                r_mn[r][i] = out_sgn * out_mag * 0.35f;  /* NMS: BSC-optimized factor */
            }
        }

        /* Variable node update */
        for (int j = 0; j < n; j++) {
            double sum = llr_in[j];
            for (int k = 0; k < H->col[j].c; k++) {
                int r = vn_adj[j][k];
                int pos = -1;
                for (int i = 0; i < cn_deg[r]; i++) {
                    if (cn_adj[r][i] == j) { pos = i; break; }
                }
                sum += r_mn[r][pos];
            }
            res.decoded[j] = (sum < 0.0) ? 1 : 0;

            for (int k = 0; k < H->col[j].c; k++) {
                int r = vn_adj[j][k];
                double s = llr_in[j];
                for (int t = 0; t < H->col[j].c; t++) {
                    if (t == k) continue;
                    int r2 = vn_adj[j][t];
                    int pos = -1;
                    for (int i = 0; i < cn_deg[r2]; i++) {
                        if (cn_adj[r2][i] == j) { pos = i; break; }
                    }
                    s += r_mn[r2][pos];
                }
                q_nm[j][k] = s;
            }
        }

        if (check_syndrome_sparse(H, res.decoded)) {
            res.converged = 1;
            res.iters_used = it + 1;
            goto cleanup;
        }
    }

    res.iters_used = max_iters;
    res.converged = 0;

cleanup:
    res.time_ms = now_ms() - t0;

    if (q_nm) {
        for (int j = 0; j < n; j++) free(q_nm[j]);
        free(q_nm);
    }
    if (r_mn) {
        for (int r = 0; r < m; r++) free(r_mn[r]);
        free(r_mn);
    }
    if (cn_adj) {
        for (int r = 0; r < m; r++) free(cn_adj[r]);
        free(cn_adj);
    }
    if (vn_adj) {
        for (int j = 0; j < n; j++) free(vn_adj[j]);
        free(vn_adj);
    }
    free(cn_deg);

    return res;
}

void spbp_free(BPResult *res)
{
    free(res->decoded);
    res->decoded = NULL;
}
