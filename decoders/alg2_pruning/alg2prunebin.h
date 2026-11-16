#ifndef ALG2PRUNEBIN_H
#define ALG2PRUNEBIN_H

#include <stdint.h>
#include "sparse_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Pruning decoder for GF(2) binary.
 *
 * Same semantics as alg2_prune, but all arithmetic is XOR.
 */
int alg2_prunebin(const SparseMatrix *H,
                  const uint8_t *syndrome,
                  uint8_t *e_out,
                  int threshold_v,
                  int max_weight,
                  int max_tries);

#ifdef __cplusplus
}
#endif

#endif /* ALG2PRUNEBIN_H */
