/*
 * lu_solve.c — Pure C LU solver for GF(16), GF(256), GF(4), binary.
 *
 * Forward elimination on ALL columns + back-substitution (v3 algorithm).
 * No Metal, no GPU — compiles anywhere with a C99 compiler.
 */

#include "lu_solve.h"
#include <string.h>
#include <stdlib.h>

typedef unsigned int uint;

/* ================================================================== */
/*  GF(16) tables                                                      */
/* ================================================================== */
static uint8_t gf16_mul_tbl[65536];
static uint8_t gf16_inv_tbl[256];
static int     gf16_ready = 0;

static void gf16_build_tables(void) {
    if (gf16_ready) return;
    uint8_t log[16] = {0}, exp[16];
    exp[0] = 1;
    for (int i = 1; i < 15; i++) {
        exp[i] = (uint8_t)((exp[i-1] << 1) ^ ((exp[i-1] & 8) ? 0x13 : 0));
        log[exp[i] & 0xF] = (uint8_t)i;
    }
    for (int i = 0; i < 256; i++) {
        int a_lo = i & 0xF, a_hi = (i >> 4);
        for (int j = 0; j < 256; j++) {
            int b_lo = j & 0xF, b_hi = (j >> 4);
            int p = 0;
            if (a_lo && b_lo) { int s = log[a_lo] + log[b_lo]; if (s >= 15) s -= 15; p ^= exp[s]; }
            if (a_hi && b_lo) { int s = log[a_hi] + log[b_lo]; if (s >= 15) s -= 15; p ^= exp[s] << 4; }
            if (a_lo && b_hi) { int s = log[a_lo] + log[b_hi]; if (s >= 15) s -= 15; p ^= exp[s]; }
            if (a_hi && b_hi) { int s = log[a_hi] + log[b_hi]; if (s >= 15) s -= 15; p ^= exp[s] << 4; }
            gf16_mul_tbl[i * 256 + j] = (uint8_t)p;
        }
    }
    for (int i = 0; i < 256; i++) {
        if (i == 0) { gf16_inv_tbl[i] = 0; continue; }
        uint8_t inv = 1;
        for (int t = 0; t < 254; t++) {
            if (gf16_mul_tbl[((uint)inv << 8) | i] == 1) break;
            inv = gf16_mul_tbl[((uint)inv << 8) | i];
        }
        gf16_inv_tbl[i] = inv;
    }
    gf16_ready = 1;
}

/* ================================================================== */
/*  GF(256) tables                                                     */
/* ================================================================== */
static uint8_t gf256_mul_tbl[65536];
static uint8_t gf256_inv_tbl[256];
static int     gf256_ready = 0;

static void gf256_build_tables(void) {
    if (gf256_ready) return;
    uint8_t log[256] = {0}, exp[512] = {0};
    exp[0] = 1;
    for (int i = 1; i < 255; i++) {
        exp[i] = (uint8_t)((exp[i-1] << 1) ^ ((exp[i-1] & 0x80) ? 0x11D : 0));
        log[exp[i]] = (uint8_t)i;
    }
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            if (i == 0 || j == 0) { gf256_mul_tbl[i * 256 + j] = 0; continue; }
            int s = log[i] + log[j];
            if (s >= 255) s -= 255;
            gf256_mul_tbl[i * 256 + j] = exp[s];
        }
    }
    for (int i = 0; i < 256; i++) {
        if (i == 0) { gf256_inv_tbl[i] = 0; continue; }
        gf256_inv_tbl[i] = exp[255 - log[i]];
    }
    gf256_ready = 1;
}

/* ================================================================== */
/*  GF(4) tables                                                       */
/* ================================================================== */
static uint8_t gf4_mul_tbl[65536];
static uint8_t gf4_inv_tbl[256];
static int     gf4_ready = 0;

static void gf4_build_tables(void) {
    if (gf4_ready) return;
    uint8_t log[4] = {0}, exp[4];
    exp[0] = 1; exp[1] = 2; exp[2] = 3; exp[3] = 1;
    log[1] = 0; log[2] = 1; log[3] = 2;
    for (int i = 0; i < 256; i++) {
        int a_lo = i & 3, a_hi = (i >> 2) & 3, a_2 = (i >> 4) & 3, a_3 = (i >> 6) & 3;
        for (int j = 0; j < 256; j++) {
            int b_lo = j & 3, b_hi = (j >> 2) & 3, b_2 = (j >> 4) & 3, b_3 = (j >> 6) & 3;
            int p = 0;
            if (a_lo && b_lo) { int s = log[a_lo] + log[b_lo]; if (s >= 3) s -= 3; p ^= exp[s]; }
            p ^= (a_hi && b_hi) ? (exp[(log[a_hi] + log[b_hi]) % 3] << 2) : 0;
            p ^= (a_2 && b_2) ? (exp[(log[a_2] + log[b_2]) % 3] << 4) : 0;
            p ^= (a_3 && b_3) ? (exp[(log[a_3] + log[b_3]) % 3] << 6) : 0;
            gf4_mul_tbl[i * 256 + j] = (uint8_t)p;
        }
    }
    for (int i = 0; i < 256; i++) {
        if (i == 0) { gf4_inv_tbl[i] = 0; continue; }
        uint8_t inv = 1;
        for (int t = 0; t < 254; t++) {
            uint8_t prod = gf4_mul_tbl[((uint)inv << 8) | i];
            if (prod == 1) break;
            inv = gf4_mul_tbl[((uint)inv << 8) | i];
        }
        gf4_inv_tbl[i] = inv;
    }
    gf4_ready = 1;
}

/* ================================================================== */
/*  Generic solver                                                      */
/* ================================================================== */
/* field: 0=GF16, 1=GF256, 2=GF4, 3=binary */
static int solve_generic(uint8_t *M, uint8_t *rhs, uint8_t *x,
                          int rows, int cols, int field)
{
    if (rows < cols || cols <= 0) return 0;

    const uint8_t *mul = NULL, *inv = NULL;
    int is_bin = (field == 3);

    switch (field) {
        case 0: if (!gf16_ready) gf16_build_tables();
                mul = gf16_mul_tbl; inv = gf16_inv_tbl; break;
        case 1: if (!gf256_ready) gf256_build_tables();
                mul = gf256_mul_tbl; inv = gf256_inv_tbl; break;
        case 2: if (!gf4_ready) gf4_build_tables();
                mul = gf4_mul_tbl; inv = gf4_inv_tbl; break;
        case 3: break; /* binary: no tables needed */
        default: return 0;
    }

    int rank = 0;

    /* Forward elimination — per-column, ALL columns */
    for (int col = 0; col < cols && rank < rows; ++col) {
        /* Find pivot */
        int pivot = rows;
        for (int r = rank; r < rows; ++r) {
            if (M[r * cols + col] != 0) { pivot = r; break; }
        }
        if (pivot >= rows) continue; /* skip dependent column */

        /* Swap pivot to rank */
        if (pivot != rank) {
            uint8_t *r1 = M + (size_t)rank * (size_t)cols;
            uint8_t *r2 = M + (size_t)pivot * (size_t)cols;
            for (int j = 0; j < cols; ++j) {
                uint8_t t = r1[j]; r1[j] = r2[j]; r2[j] = t;
            }
            uint8_t t = rhs[rank]; rhs[rank] = rhs[pivot]; rhs[pivot] = t;
        }

        /* Normalize pivot row (binary: skip, since 1*1=1) */
        if (!is_bin) {
            uint8_t pv = M[(size_t)rank * (size_t)cols + (size_t)col];
            uint8_t iv = (pv == 0) ? 0 : inv[pv];
            uint8_t *pr = M + (size_t)rank * (size_t)cols;
            for (int j = col; j < cols; ++j)
                pr[j] = mul[((unsigned int)pr[j] << 8) | iv];
            rhs[rank] = mul[((unsigned int)rhs[rank] << 8) | iv];
        }

        /* Forward-eliminate ALL columns from rows below */
        if (is_bin) {
            for (int r = rank + 1; r < rows; ++r) {
                if (M[r * cols + col] == 0) continue;
                uint8_t *rr = M + (size_t)r * (size_t)cols;
                uint8_t *pr = M + (size_t)rank * (size_t)cols;
                for (int j = col; j < cols; ++j)
                    rr[j] ^= pr[j];  /* binary: factor=1, just XOR */
                rhs[r] ^= rhs[rank];
            }
        } else {
            for (int r = rank + 1; r < rows; ++r) {
                uint8_t factor = M[r * cols + col];
                if (factor == 0) continue;
                unsigned int fbase = (unsigned int)factor << 8;
                uint8_t *rr = M + (size_t)r * (size_t)cols;
                uint8_t *pr = M + (size_t)rank * (size_t)cols;
                for (int j = col; j < cols; ++j)
                    rr[j] ^= mul[fbase | (uint)pr[j]];
                rhs[r] ^= mul[fbase | rhs[rank]];
            }
        }

        ++rank;
    }

    /* Consistency check: extra rows must have zero RHS */
    for (int r = rank; r < rows; ++r)
        if (rhs[r] != 0) return 0;
    if (rank < cols) return 0;

    /* Back-substitution */
    memset(x, 0, (size_t)cols * sizeof(uint8_t));
    if (is_bin) {
        for (int r = cols - 1; r >= 0; --r) {
            uint8_t s = rhs[r];
            for (int c = r + 1; c < cols; ++c)
                s ^= M[r * cols + c] & x[c];
            x[r] = s;
        }
    } else {
        for (int r = cols - 1; r >= 0; --r) {
            uint8_t s = rhs[r];
            for (int c = r + 1; c < cols; ++c)
                s ^= mul[((unsigned int)M[r * cols + c] << 8) | x[c]];
            x[r] = s;
        }
    }
    return cols;
}

/* ================================================================== */
/*  Public API                                                          */
/* ================================================================== */
int solve_lu_gf16(uint8_t *M, uint8_t *rhs, uint8_t *x, int rows, int cols) {
    return solve_generic(M, rhs, x, rows, cols, 0);
}
int solve_lu_gf256(uint8_t *M, uint8_t *rhs, uint8_t *x, int rows, int cols) {
    return solve_generic(M, rhs, x, rows, cols, 1);
}
int solve_lu_gf4(uint8_t *M, uint8_t *rhs, uint8_t *x, int rows, int cols) {
    return solve_generic(M, rhs, x, rows, cols, 2);
}
int solve_lu_binary(uint8_t *M, uint8_t *rhs, uint8_t *x, int rows, int cols) {
    return solve_generic(M, rhs, x, rows, cols, 3);
}
