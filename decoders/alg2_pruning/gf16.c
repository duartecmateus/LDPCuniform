#include "gf16.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// =====================================================
// ---------------- GF(16) tables + setup ---------------
// =====================================================
/* GF(16) with primitive polynomial x^4 + x + 1 = 0x13 */
#define GF16_PRIM_POLY 0x13
#define GF16_SIZE      16

static uint8_t gflog_tbl[GF16_SIZE];
static uint8_t gfexp_tbl[GF16_SIZE];
static int gf16_inited = 0;

void gf16_init(void) {
    if (gf16_inited) return;

    memset(gflog_tbl, 0, sizeof(gflog_tbl));
    memset(gfexp_tbl, 0, sizeof(gfexp_tbl));

    unsigned int b = 1;
    for (unsigned int log = 0; log < GF16_SIZE - 1; log++) {
        gflog_tbl[b] = (uint8_t)log;
        gfexp_tbl[log] = (uint8_t)b;
        b <<= 1;
        if (b & GF16_SIZE)
            b ^= GF16_PRIM_POLY;
    }
    gfexp_tbl[15] = gfexp_tbl[0];
    gf16_inited = 1;
}

// =====================================================
// ---------------- GF(16) ops --------------------------
// =====================================================
gf16_t gf16_add(gf16_t a, gf16_t b) { return (gf16_t)(a ^ b); }
gf16_t gf16_sub(gf16_t a, gf16_t b) { return (gf16_t)(a ^ b); }

gf16_t gf16_mul(gf16_t a, gf16_t b) {
    if (!gf16_inited) gf16_init();
    if (a == 0 || b == 0) return 0;
    unsigned int la = gflog_tbl[a];
    unsigned int lb = gflog_tbl[b];
    unsigned int s = la + lb;
    if (s >= 15) s -= 15;
    return gfexp_tbl[s];
}

gf16_t gf16_inv(gf16_t a) {
    if (!gf16_inited) gf16_init();
    if (a == 0) {
        fprintf(stderr, "gf16_inv(0) not defined\n");
        exit(1);
    }
    unsigned int la = gflog_tbl[a];
    unsigned int idx = 15u - la;
    if (idx >= 15u) idx -= 15u;
    return gfexp_tbl[idx];
}

gf16_t gf16_div(gf16_t a, gf16_t b) {
    if (!gf16_inited) gf16_init();
    if (b == 0) {
        fprintf(stderr, "gf16_divide by zero\n");
        exit(1);
    }
    if (a == 0) return 0;
    unsigned int la = gflog_tbl[a];
    unsigned int lb = gflog_tbl[b];
    int d = (int)la - (int)lb;
    if (d < 0) d += 15;
    return gfexp_tbl[(unsigned)d];
}

void gf16_matvec(const gf16_t *A, const gf16_t *x,
                 gf16_t *y, int rows, int cols)
{
    for (int i = 0; i < rows; ++i) {
        gf16_t acc = 0;
        for (int j = 0; j < cols; ++j) {
            acc = gf16_add(acc, gf16_mul(A[i * cols + j], x[j]));
        }
        y[i] = acc;
    }
}

// =====================================================
// ----------- Gaussian Elimination in GF(16) ----------
// =====================================================
int gf16_gaussian_elimination(gf16_t *A, gf16_t *b, int n) {
    if (!gf16_inited) gf16_init();

    for (int col = 0; col < n; col++) {
        // find pivot
        int pivot = -1;
        for (int row = col; row < n; row++) {
            if (A[row * n + col] != 0) {
                pivot = row;
                break;
            }
        }
        if (pivot == -1)
            return 0; // singular

        // swap rows
        if (pivot != col) {
            for (int j = 0; j < n; j++) {
                gf16_t tmp = A[col * n + j];
                A[col * n + j] = A[pivot * n + j];
                A[pivot * n + j] = tmp;
            }
            gf16_t tmpb = b[col];
            b[col] = b[pivot];
            b[pivot] = tmpb;
        }

        // normalize pivot row
        gf16_t inv_pivot = gf16_inv(A[col * n + col]);
        for (int j = col; j < n; j++)
            A[col * n + j] = gf16_mul(A[col * n + j], inv_pivot);
        b[col] = gf16_mul(b[col], inv_pivot);

        // eliminate other rows
        for (int row = 0; row < n; row++) {
            if (row == col) continue;
            gf16_t factor = A[row * n + col];
            if (factor != 0) {
                for (int j = col; j < n; j++)
                    A[row * n + j] = gf16_sub(A[row * n + j],
                                              gf16_mul(factor, A[col * n + j]));
                b[row] = gf16_sub(b[row], gf16_mul(factor, b[col]));
            }
        }
    }
    return 1;
}
