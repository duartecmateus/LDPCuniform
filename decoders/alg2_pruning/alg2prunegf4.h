#ifndef ALG2PRUNEGF4_H
#define ALG2PRUNEGF4_H

#include <stdint.h>
#include "sparse_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Pruning decoder for GF(4).
 *
 * H             — sparse parity-check matrix
 * syndrome      — syndrome vector (size H->rows)
 * e_out         — recovered error vector (size H->cols)
 * threshold_v   — keep column if its intersection with syndrome support > threshold_v
 * max_weight    — maximum allowed Hamming weight of e_out
 * max_tries     — maximum random sampling attempts
 *
 * Returns the number of attempts on success (>0), or 0 on failure.
 */
int alg2_prune_gf4(const SparseMatrix *H,
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

#endif /* ALG2PRUNEGF4_H */
