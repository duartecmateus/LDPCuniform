#ifndef ALG3PRUNEBIN_H
#define ALG3PRUNEBIN_H

#include <stdint.h>
#include "sparse_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Pruning decoder for GF(2) binary. All arithmetic is XOR.
 */
int alg3_prunebin(const SparseMatrix *H,
                  const uint8_t *syndrome,
                  uint8_t *e_out,
                  int threshold_v,
                  int max_weight,
                  int max_tries);

#ifdef __cplusplus
}
#endif

#endif /* ALG3PRUNEBIN_H */
