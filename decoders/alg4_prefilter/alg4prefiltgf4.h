#ifndef ALG4PREFILTGF4_H
#define ALG4PREFILTGF4_H

#include <stdint.h>
#include "sparse_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * alg4prefilt_gf4 — Pruning decoder for GF(4) with consistency pre-filter.
 *
 * Adds a pre-filter step before LU solve: for each candidate column,
 * estimates the error value from each incident non-zero syndrome row.
 * The column is kept only if ALL such estimates agree on the same value.
 *
 * H             — sparse parity-check matrix
 * syndrome      — syndrome vector (size H->rows)
 * e_out         — recovered error vector (size H->cols)
 * threshold_v   — keep column if intersection with syndrome support > threshold_v
 * max_weight    — maximum allowed Hamming weight of e_out
 * max_tries     — maximum random sampling attempts
 *
 * Returns the number of attempts on success (>0), or 0 on failure.
 */
int alg4prefilt_gf4(const SparseMatrix *H,
                    const uint8_t *syndrome,
                    uint8_t *e_out,
                    int threshold_v,
                    int max_weight,
                    int max_tries);

/* GF(4) sparse matrix-vector multiply: syndrome = H * e */
void sparse_matvec_gf4(const SparseMatrix *H, const uint8_t *e, uint8_t *syndrome);

#ifdef __cplusplus
}
#endif

#endif /* ALG4PREFILTGF4_H */
