#ifndef GF16_GAUSS_H 
#define GF16_GAUSS_H

#include <stddef.h> 
#include <stdbool.h>

#include "gf16.h"

// Solves A x = b in-place over GF(2^4). // A: n x n matrix, row-major, modified in-place to reduced form. // b: length n vector, modified in-place. // num_workers: number of threads to use during elimination (>=1). // x_out: length n vector to store the solution. // Returns 0 on success, -1 if the matrix is singular. 

int gf16_gaussian_elimination(gf16_t* A, gf16_t* b, int n, int num_workers, gf16_t* x_out);

// If you want to call the phases separately:
 //int gf16_forward_elimination(gf16_t* A, gf16_t* b, int n, int num_workers); 
 //int gf16_back_substitution(const gf16_t* A, const gf16_t* b, int n, gf16_t* x_out);

#endif // GF16_GAUSS_H