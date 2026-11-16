#ifndef METAL_BRIDGE_H
#define METAL_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize / release global Metal state. */
int metal_bridge_init(void);
void metal_bridge_cleanup(void);

/*
 * alg2 helper: for each column c, compute
 *
 *     intersection_count[c] = |supp(column c) intersect supp(syndrome)|.
 *
 * nonint_count receives the number of columns with intersection_count[c] == 0.
 * Returns 1 on successful Metal execution, 0 on failure.
 */
int alg2_metal_compute_column_intersections(const uint8_t *matrix,
                                            const uint8_t *syndrome,
                                            int rows,
                                            int cols,
                                            int *intersection_count,
                                            int *nonint_count);

/*
 * alg2 helper: solve an overdetermined GF(256) system
 *
 *     M x = rhs,
 *
 * where M is total_rows x cols, row-major.  Returns 1 only if the system is
 * consistent and has full column rank, hence a unique solution.  Returns 0 if
 * the system is inconsistent, underdetermined, or if Metal execution fails.
 *
 * M and rhs may be modified, matching the CPU elimination behavior.
 */
int alg2_metal_solve_overdetermined_unique(uint8_t *M,
                                           uint8_t *rhs,
                                           int total_rows,
                                           int cols,
                                           uint8_t *x);

/*
 * Same signature, for GF(16) arithmetic.  Uses the same Metal kernels with
 * GF(16) multiplication / inverse tables (elements 0..15, prim poly 0x13).
 */
int alg2_metal_solve_overdetermined_unique_gf16(uint8_t *M,
                                                uint8_t *rhs,
                                                int total_rows,
                                                int cols,
                                                uint8_t *x);

/*
 * GF(4) version (elements 0..3, prim poly 0x13 → GF(4) via x^2+x+1).
 */
int alg2_metal_solve_overdetermined_unique_gf4(uint8_t *M,
                                               uint8_t *rhs,
                                               int total_rows,
                                               int cols,
                                               uint8_t *x);

/*
 * GF(2) / binary version — multiplication is AND (trivial), inverse is
 * identity.  Uses the same Metal elimination kernels with appropriate tables.
 */
int alg2_metal_solve_overdetermined_unique_gf2(uint8_t *M,
                                               uint8_t *rhs,
                                               int total_rows,
                                               int cols,
                                               uint8_t *x);

/*
 * Direct binary elimination using dedicated XOR-only Metal kernels.
 * No table lookups — pure word-wise XOR on GPU.  Skips normalization
 * and pivot-inverse computation (unnecessary for GF2).
 */
int alg2_metal_solve_overdetermined_unique_bin(uint8_t *M,
                                               uint8_t *rhs,
                                               int total_rows,
                                               int cols,
                                               uint8_t *x);

#ifdef __cplusplus
}
#endif

#endif /* METAL_BRIDGE_H */
