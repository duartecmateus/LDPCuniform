#ifndef ALG3PRUNEGF4_H
#define ALG3PRUNEGF4_H

#include <stdint.h>
#include "sparse_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Pruning decoder for GF(4).
 */
int alg3_prune_gf4(const SparseMatrix *H,
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

#endif /* ALG3PRUNEGF4_H */
