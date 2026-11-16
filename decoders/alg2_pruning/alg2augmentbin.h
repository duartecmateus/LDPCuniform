#ifndef ALG2AUGMENTBIN_H
#define ALG2AUGMENTBIN_H

#include <stdint.h>
#include "sparse_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Augmentation decoder for GF(2) binary.
 *
 * Adds zero-constraint rows for columns with low syndrome intersection,
 * building an overdetermined system and solving via XOR Gaussian elimination.
 *
 * H             — sparse parity-check matrix
 * syndrome      — syndrome vector (size H->rows)
 * e_out         — recovered error vector (size H->cols)
 * max_intersection — add zero rows for columns with intersection <= this value
 * extra_added_rows — extra constraints beyond cols-rows (for overdetermined system)
 * max_weight    — maximum allowed Hamming weight of recovered error
 * max_tries     — maximum random sampling attempts
 *
 * Returns number of attempts on success (>0), or 0 on failure.
 */
int alg2_augment_bin(const SparseMatrix *H,
                     const uint8_t *syndrome,
                     uint8_t *e_out,
                     int max_intersection,
                     int extra_added_rows,
                     int max_weight,
                     int max_tries);

#ifdef __cplusplus
}
#endif

#endif /* ALG2AUGMENTBIN_H */
