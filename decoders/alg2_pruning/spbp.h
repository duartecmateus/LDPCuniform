/* Sparse BP decoder for large LDPC matrices.
   Builds adjacency lists directly from SparseMatrix, avoiding
   the dense representation that consumes ~400MB for 6400x16000.

   Implements both SPA (Sum-Product) and Min-Sum (MS) BP.
*/

#ifndef SPBP_H
#define SPBP_H

#include "sparse_utils.h"

/* BP result */
typedef struct {
    int converged;       /* 1 if H * decoded = 0 */
    int *decoded;        /* decoded codeword (allocated, N entries) */
    int iters_used;      /* iterations actually used */
    double time_ms;      /* wall clock time */
} BPResult;

/* Run SPA (Sum-Product Algorithm) belief propagation.
   llr_in: N LLRs (double array, caller-owned).
   max_iters: max iterations (50 is typical).
   Returns BPResult struct; decoded is newly allocated. */
BPResult spbp_decode_spa(const SparseMatrix *H, const double *llr_in,
                          int max_iters);

/* Run Min-Sum belief propagation.
   Returns BPResult struct; decoded is newly allocated. */
BPResult spbp_decode_ms(const SparseMatrix *H, const double *llr_in,
                         int max_iters);

/* Free a BPResult */
void spbp_free(BPResult *res);

#endif
