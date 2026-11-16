#include "gf4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// =====================================================
// ---------------- GF(4) tables + setup ---------------
// =====================================================
/* GF(4) with primitive polynomial x^2 + x + 1 = 0x7 */
#define GF4_PRIM_POLY 0x7
#define GF4_SIZE       4

static uint8_t gflog_tbl[GF4_SIZE];
static uint8_t gfexp_tbl[GF4_SIZE];
static int gf4_inited = 0;

void gf4_init(void) {
    if (gf4_inited) return;

    memset(gflog_tbl, 0, sizeof(gflog_tbl));
    memset(gfexp_tbl, 0, sizeof(gfexp_tbl));

    unsigned int b = 1;
    for (unsigned int log = 0; log < GF4_SIZE - 1; log++) {
        gflog_tbl[b] = (uint8_t)log;
        gfexp_tbl[log] = (uint8_t)b;
        b <<= 1;
        if (b & GF4_SIZE)
            b ^= GF4_PRIM_POLY;
    }
    gfexp_tbl[3] = gfexp_tbl[0];
    gf4_inited = 1;
}

// =====================================================
// ---------------- GF(4) ops --------------------------
// =====================================================
gf4_t gf4_add(gf4_t a, gf4_t b) { return (gf4_t)(a ^ b); }
gf4_t gf4_sub(gf4_t a, gf4_t b) { return (gf4_t)(a ^ b); }

gf4_t gf4_mul(gf4_t a, gf4_t b) {
    if (!gf4_inited) gf4_init();
    if (a == 0 || b == 0) return 0;
    unsigned int la = gflog_tbl[a];
    unsigned int lb = gflog_tbl[b];
    unsigned int s = la + lb;
    if (s >= 3) s -= 3;
    return gfexp_tbl[s];
}

gf4_t gf4_inv(gf4_t a) {
    if (!gf4_inited) gf4_init();
    if (a == 0) {
        fprintf(stderr, "gf4_inv(0) not defined\n");
        exit(1);
    }
    unsigned int la = gflog_tbl[a];
    unsigned int idx = 3u - la;
    if (idx >= 3u) idx -= 3u;
    return gfexp_tbl[idx];
}

gf4_t gf4_div(gf4_t a, gf4_t b) {
    if (!gf4_inited) gf4_init();
    if (b == 0) {
        fprintf(stderr, "gf4_divide by zero\n");
        exit(1);
    }
    if (a == 0) return 0;
    unsigned int la = gflog_tbl[a];
    unsigned int lb = gflog_tbl[b];
    int d = (int)la - (int)lb;
    if (d < 0) d += 3;
    return gfexp_tbl[(unsigned)d];
}

void gf4_matvec(const gf4_t *A, const gf4_t *x,
                 gf4_t *y, int rows, int cols)
{
    for (int i = 0; i < rows; ++i) {
        gf4_t acc = 0;
        for (int j = 0; j < cols; ++j) {
            acc = gf4_add(acc, gf4_mul(A[i * cols + j], x[j]));
        }
        y[i] = acc;
    }
}

// =====================================================
// ----------- Gaussian Elimination in GF(4) ----------
// =====================================================
int gf4_gaussian_elimination(gf4_t *A, gf4_t *b, int n) {
    if (!gf4_inited) gf4_init();

    for (int col = 0; col < n; col++) {
        int pivot = -1;
        for (int row = col; row < n; row++) {
            if (A[row * n + col] != 0) {
                pivot = row;
                break;
            }
        }
        if (pivot == -1)
            return 0;

        if (pivot != col) {
            for (int j = 0; j < n; j++) {
                gf4_t tmp = A[col * n + j];
                A[col * n + j] = A[pivot * n + j];
                A[pivot * n + j] = tmp;
            }
            gf4_t tmpb = b[col];
            b[col] = b[pivot];
            b[pivot] = tmpb;
        }

        gf4_t inv_pivot = gf4_inv(A[col * n + col]);
        for (int j = col; j < n; j++)
            A[col * n + j] = gf4_mul(A[col * n + j], inv_pivot);
        b[col] = gf4_mul(b[col], inv_pivot);

        for (int row = 0; row < n; row++) {
            if (row == col) continue;
            gf4_t factor = A[row * n + col];
            if (factor != 0) {
                for (int j = col; j < n; j++)
                    A[row * n + j] = gf4_sub(A[row * n + j],
                                              gf4_mul(factor, A[col * n + j]));
                b[row] = gf4_sub(b[row], gf4_mul(factor, b[col]));
            }
        }
    }
    return 1;
}
