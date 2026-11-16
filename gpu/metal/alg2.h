#ifndef ALG2_H
#define ALG2_H

#include "gf256.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Backward-compatible algorithm entrypoint. */
int alg2(const gf256_t *A,
         const gf256_t *v,
         gf256_t *x,
         int rows,
         int cols,
         int max_weight,
         int max_tries);

/* Configurable version.
 *
 * max_syndrome_intersections:
 *   A column is allowed as an added-row candidate if the support of that
 *   column intersects the support of the syndrome in at most this many rows.
 *   Setting this to 0 recovers the old non-intersecting-column rule.
 *
 * extra_added_rows:
 *   Additional zero constraints beyond the minimum cols - rows needed to form
 *   a square system.  The constructed system therefore has
 *
 *       rows + (cols - rows) + extra_added_rows = cols + extra_added_rows
 *
 *   rows and cols columns.  If extra_added_rows > 0, the system is
 *   overdetermined.  The solver accepts only consistent full-column-rank
 *   systems, so the result is either a unique solution or no solution.
 */
int alg2_ex(const gf256_t *A,
            const gf256_t *v,
            gf256_t *x,
            int rows,
            int cols,
            int max_weight,
            int max_tries,
            int max_syndrome_intersections,
            int extra_added_rows);

#ifdef __cplusplus
}
#endif

#endif /* ALG2_H */
int alg2_gpu_prune(const gf256_t *A, const gf256_t *v, gf256_t *x,
                   int rows, int cols,
                   int max_weight, int max_tries,
                   int max_syndrome_intersections);
