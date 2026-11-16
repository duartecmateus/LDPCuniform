#ifndef GF16_H
#define GF16_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint8_t gf16_t;

/* Must be called once before using other GF(16) functions */
void gf16_init(void);

/* Core GF(16) ops (addition/subtraction are XOR) */
gf16_t gf16_add(gf16_t a, gf16_t b);
gf16_t gf16_sub(gf16_t a, gf16_t b);
gf16_t gf16_mul(gf16_t a, gf16_t b);
gf16_t gf16_inv(gf16_t a);
gf16_t gf16_div(gf16_t a, gf16_t b);

/* Gaussian elimination over GF(16).
   Solves A*x=b for square A (n x n), modifying A and b in place.
   Returns 1 on success, 0 if A is singular. */
int gf16_gaussian_elimination(gf16_t *A, gf16_t *b, int n);

/* Matrix-vector multiply: y = A*x, where A is rows x cols */
void gf16_matvec(const gf16_t *A, const gf16_t *x,
                 gf16_t *y, int rows, int cols);

#ifdef __cplusplus
}
#endif

#endif /* GF16_H */
