#if defined(__APPLE__) && defined(ENABLE_GF256_METAL)
int gf256_metal_elimination_update(unsigned char *A, unsigned char *b,
                                   const unsigned char *pivot_row,
                                   unsigned char pivot_b,
                                   const unsigned char *gflog,
                                   const unsigned char *gfexp,
                                   int n, int col);
#endif

#include "gf256.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

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

    unsigned int prim_poly_4  = 023;
    unsigned int prim_poly_8  = 0435;
    unsigned int prim_poly_16 = 0210013;

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
        const gf256_t *row = A + (size_t)i * (size_t)cols;
        for (int j = 0; j < cols; ++j) {
            acc ^= gf256_mul(row[j], x[j]);
        }
        y[i] = acc;
    }
}

static void gf256_swap_rows(gf256_t *A, gf256_t *b, int n, int row1, int row2)
{
    if (row1 == row2) return;

    gf256_t *r1 = A + (size_t)row1 * (size_t)n;
    gf256_t *r2 = A + (size_t)row2 * (size_t)n;

    for (int j = 0; j < n; ++j) {
        gf256_t tmp = r1[j];
        r1[j] = r2[j];
        r2[j] = tmp;
    }

    gf256_t tmpb = b[row1];
    b[row1] = b[row2];
    b[row2] = tmpb;
}

void gf256_normalize_pivot_row(gf256_t *A, gf256_t *b, int n, int col)
{
    gf256_t *pivot_row = A + (size_t)col * (size_t)n;
    gf256_t pivot = pivot_row[col];

    if (pivot == 0) return;
    if (pivot == 1) return;

    gf256_t inv = gf256_inv(pivot);
    for (int j = col; j < n; ++j) {
        pivot_row[j] = gf256_mul(pivot_row[j], inv);
    }
    b[col] = gf256_mul(b[col], inv);
}

static void gf256_eliminate_column(gf256_t *A, gf256_t *b, int n, int col)
{
    gf256_t *pivot_row = A + (size_t)col * (size_t)n;
    gf256_t pivot_b = b[col];
    const int remaining = n - col;

#if defined(__APPLE__) && defined(ENABLE_GF256_METAL)
    if (n >= 1024) {
        if (gf256_metal_elimination_update((unsigned char *)A, (unsigned char *)b,
                                           (const unsigned char *)pivot_row,
                                           (unsigned char)pivot_b,
                                           gflog_tbl, gfexp_tbl, n, col)) {
            return;
        }
    }
#endif

    // This loop is the main future Metal offload point:
    // each row update is independent once the pivot row is fixed.
#ifdef _OPENMP
#pragma omp parallel for if (n >= 256) schedule(static)
#endif
    for (int row = 0; row < n; ++row) {
        if (row == col) continue;

        gf256_t *target_row = A + (size_t)row * (size_t)n;
        gf256_t factor = target_row[col];
        if (factor == 0) continue;

        target_row[col] = 0;
        for (int offset = 1; offset < remaining; ++offset) {
            int j = col + offset;
            target_row[j] ^= gf256_mul(factor, pivot_row[j]);
        }
        b[row] ^= gf256_mul(factor, pivot_b);
    }
}

// =====================================================
// ----------- Gaussian Elimination in GF(256) ----------
// =====================================================
int gf256_gaussian_elimination(gf256_t *A, gf256_t *b, int n) {
    if (!gf256_inited) gf256_init();
    if (!A || !b || n <= 0) return 0;

    for (int col = 0; col < n; ++col) {
        int pivot = -1;
        for (int row = col; row < n; ++row) {
            if (A[(size_t)row * (size_t)n + (size_t)col] != 0) {
                pivot = row;
                break;
            }
        }
        if (pivot == -1) {
            return 0;
        }

        gf256_swap_rows(A, b, n, col, pivot);
        gf256_normalize_pivot_row(A, b, n, col);
        gf256_eliminate_column(A, b, n, col);
    }

    return 1;
}
