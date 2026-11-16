#ifndef LU_SOLVE_H
#define LU_SOLVE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Solve an overdetermined GF(16) linear system M x = rhs
 * using forward elimination + back-substitution (v3 algorithm).
 *
 * M:     rows x cols matrix (row-major), modified in place
 * rhs:   length rows, modified in place
 * x:     length cols, receives the solution
 * rows:  number of rows (must be >= cols)
 * cols:  number of columns
 *
 * Returns cols on success, 0 if rank-deficient or inconsistent.
 */
int solve_lu_gf16(uint8_t *M, uint8_t *rhs, uint8_t *x, int rows, int cols);

/*
 * Same for GF(256) — elements 0..255, primitive polynomial 0x11D.
 */
int solve_lu_gf256(uint8_t *M, uint8_t *rhs, uint8_t *x, int rows, int cols);

/*
 * Same for GF(4) — elements 0..3.
 */
int solve_lu_gf4(uint8_t *M, uint8_t *rhs, uint8_t *x, int rows, int cols);

/*
 * Same for binary (GF(2)) — elements 0/1.
 */
int solve_lu_binary(uint8_t *M, uint8_t *rhs, uint8_t *x, int rows, int cols);

#ifdef __cplusplus
}
#endif

#endif /* LU_SOLVE_H */
