
#include "gf256.h"

#include <stdio.h>
#include <stdlib.h>

// =====================================================
// ---------------- GF(256) tables + setup ---------------
// =====================================================
static uint8_t gflog_tbl[256];
static uint8_t gfexp_tbl[256];
static int gf256_inited = 0;

static int setup_tables_cpu(int w) {
    unsigned int b;
    unsigned int log;
    unsigned int x_to_w;
    unsigned int prim_poly;

    unsigned int prim_poly_4  = 023;      // octal
    unsigned int prim_poly_8  = 0435;     // octal (== 0x11D == 285)
    unsigned int prim_poly_16 = 0210013;  // octal

    switch (w) {
        case 4:  prim_poly = prim_poly_4;  break;
        case 8:  prim_poly = prim_poly_8;  break;
        case 16: prim_poly = prim_poly_16; break;
        default: return -1;
    }

    x_to_w = 1u << w;

    for (unsigned i = 0; i < x_to_w; ++i) gflog_tbl[i] = 0;
    for (unsigned i = 0; i < x_to_w; ++i) gfexp_tbl[i] = 0;

    b = 1;
    for (log = 0; log < x_to_w - 1; log++) {
        gflog_tbl[b] = (uint8_t)log;
        gfexp_tbl[log] = (uint8_t)b;
        b <<= 1;
        if (b & x_to_w) b ^= prim_poly;
    }
    gfexp_tbl[255] = gfexp_tbl[0];
    return 0;
}

void gf256_init(void) {
    if (gf256_inited) return;
    if (setup_tables_cpu(8) != 0) {
        fprintf(stderr, "gf256_init failed\n");
        exit(1);
    }
    gf256_inited = 1;
}

// =====================================================
// ---------------- GF(256) ops --------------------------
// =====================================================
gf256_t gf256_add(gf256_t a, gf256_t b) { return (gf256_t)(a ^ b); }
gf256_t gf256_sub(gf256_t a, gf256_t b) { return (gf256_t)(a ^ b); }

gf256_t gf256_mul(gf256_t a, gf256_t b) {
    if (!gf256_inited) gf256_init();
    if (a == 0 || b == 0) return 0;
    unsigned int la = gflog_tbl[a];
    unsigned int lb = gflog_tbl[b];
    unsigned int s = la + lb;
    if (s >= 255) s -= 255;
    return gfexp_tbl[s];
}

gf256_t gf256_inv(gf256_t a) {
    if (!gf256_inited) gf256_init();
    if (a == 0) {
        fprintf(stderr, "gf256_inv(0) not defined\n");
        exit(1);
    }
    unsigned int la = gflog_tbl[a];
    unsigned int idx = 255u - la;
    if (idx >= 255u) idx -= 255u;
    return gfexp_tbl[idx];
}

gf256_t gf256_div(gf256_t a, gf256_t b) {
    if (!gf256_inited) gf256_init();
    if (b == 0) {
        fprintf(stderr, "gf256_divide by zero\n");
        exit(1);
    }
    if (a == 0) return 0;
    unsigned int la = gflog_tbl[a];
    unsigned int lb = gflog_tbl[b];
    int d = (int)la - (int)lb;
    if (d < 0) d += 255;
    return gfexp_tbl[(unsigned)d];
}


void gf256_matvec(const gf256_t *A, const gf256_t *x,
                   gf256_t *y, int rows, int cols)
{
    for (int i = 0; i < rows; ++i) {
        gf256_t acc = 0;
        for (int j = 0; j < cols; ++j) {
            acc = gf256_add(acc, gf256_mul(A[i * cols + j], x[j]));
        }
        y[i] = acc;
    }
}

// =====================================================
// ----------- Gaussian Elimination in GF(256) ----------
// =====================================================
int gf256_gaussian_elimination(gf256_t *A, gf256_t *b, int n) {
    if (!gf256_inited) gf256_init();

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
                gf256_t tmp = A[col * n + j];
                A[col * n + j] = A[pivot * n + j];
                A[pivot * n + j] = tmp;
            }
            gf256_t tmpb = b[col];
            b[col] = b[pivot];
            b[pivot] = tmpb;
        }

        // normalize pivot row
        gf256_t inv_pivot = gf256_inv(A[col * n + col]);
        for (int j = col; j < n; j++)
            A[col * n + j] = gf256_mul(A[col * n + j], inv_pivot);
        b[col] = gf256_mul(b[col], inv_pivot);

        // eliminate other rows
        for (int row = 0; row < n; row++) {
            if (row == col) continue;
            gf256_t factor = A[row * n + col];
            if (factor != 0) {
                for (int j = col; j < n; j++)
                    A[row * n + j] = gf256_sub(A[row * n + j],
                                                gf256_mul(factor, A[col * n + j]));
                b[row] = gf256_sub(b[row], gf256_mul(factor, b[col]));
            }
        }
    }
    return 1;
}
