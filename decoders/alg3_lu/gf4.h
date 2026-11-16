#ifndef GF4_H
#define GF4_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint8_t gf4_t;

/* Must be called once before using other GF(4) functions */
void gf4_init(void);

/* Core GF(4) ops (addition/subtraction are XOR) */
gf4_t gf4_add(gf4_t a, gf4_t b);
gf4_t gf4_sub(gf4_t a, gf4_t b);
gf4_t gf4_mul(gf4_t a, gf4_t b);
gf4_t gf4_inv(gf4_t a);
gf4_t gf4_div(gf4_t a, gf4_t b);

/* Gaussian elimination over GF(4). Returns 1 on success, 0 if singular. */
int gf4_gaussian_elimination(gf4_t *A, gf4_t *b, int n);

/* Matrix-vector multiply: y = A*x, where A is rows x cols */
void gf4_matvec(const gf4_t *A, const gf4_t *x,
                 gf4_t *y, int rows, int cols);

#ifdef __cplusplus
}
#endif

#endif /* GF4_H */
