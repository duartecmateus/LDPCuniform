/* bench_solver.c — standalone solver benchmark (all 4 fields) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "lu_solve.h"

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1000.0 * (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e6;
}

int main(void)
{
    int R = 1600, C = 659;
    uint8_t *M = (uint8_t *)malloc((size_t)R * C);
    uint8_t *r = (uint8_t *)malloc(R);
    uint8_t *x = (uint8_t *)calloc(C, 1);

    srand(54321);
    for (int i = 0; i < R * C; i++) M[i] = (uint8_t)(rand() % 16);
    for (int i = 0; i < R; i++) r[i] = (uint8_t)(rand() % 16);

    double t0, t1;
    int rk;

    t0 = now_ms(); rk = solve_lu_gf16(M, r, x, R, C); t1 = now_ms();
    printf("GF(16)  %dx%d  rank=%-4d  %.0fms\n", R, C, rk, t1 - t0);

    t0 = now_ms(); rk = solve_lu_gf256(M, r, x, R, C); t1 = now_ms();
    printf("GF(256) %dx%d  rank=%-4d  %.0fms\n", R, C, rk, t1 - t0);

    t0 = now_ms(); rk = solve_lu_gf4(M, r, x, R, C); t1 = now_ms();
    printf("GF(4)   %dx%d  rank=%-4d  %.0fms\n", R, C, rk, t1 - t0);

    t0 = now_ms(); rk = solve_lu_binary(M, r, x, R, C); t1 = now_ms();
    printf("Binary  %dx%d  rank=%-4d  %.0fms\n", R, C, rk, t1 - t0);

    free(M); free(r); free(x);
    return 0;
}
