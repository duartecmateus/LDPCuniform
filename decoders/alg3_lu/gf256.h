#ifndef GF256_H
#define GF256_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint8_t gf256_t;

/* Must be called once before using other GF(256) functions */
void gf256_init(void);

/* Core GF(256) ops (addition/subtraction are XOR) */
gf256_t gf256_add(gf256_t a, gf256_t b);
gf256_t gf256_sub(gf256_t a, gf256_t b);
gf256_t gf256_mul(gf256_t a, gf256_t b);
gf256_t gf256_inv(gf256_t a);
gf256_t gf256_div(gf256_t a, gf256_t b);

/* Gaussian elimination over GF(256).
   Solves A*x=b for square A (n x n), modifying A and b in place.
   Returns 1 on success, 0 if A is singular. */
int gf256_gaussian_elimination(gf256_t *A, gf256_t *b, int n);

/* Matrix-vector multiply: y = A*x, where A is rows x cols */
void gf256_matvec(const gf256_t *A, const gf256_t *x,
                   gf256_t *y, int rows, int cols);

#ifdef __cplusplus
}
#endif

#endif /* GF256_H */
