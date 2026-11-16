#ifndef SPARSE_UTILS_H
#define SPARSE_UTILS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Sparse column: c non-zero entries with row indices and field values */
typedef struct {
    int c;
    int *rows;        /* row indices, size c */
    uint8_t *values;  /* field values, size c (1 for binary, 1..255 for GF256) */
} SparseCol;

/* Loaded matrix: sparse columns + header info */
typedef struct {
    int rows;         /* m: number of parity checks */
    int cols;         /* n: code length (number of columns) */
    int c;            /* column weight */
    int field;        /* 0 = GF(2) binary, 1 = GF(256) */
    SparseCol *col;   /* array of cols sparse columns */
} SparseMatrix;

/* Load a matrix from the genU file format.
   Returns 1 on success, 0 on failure (prints error to stderr). */
int load_sparse_matrix(const char *filename, SparseMatrix *mat);

/* Free all memory inside mat (but not mat itself) */
void free_sparse_matrix(SparseMatrix *mat);

/* Compute syndrome = H * e (dense mat-vec using sparse representation).
   H is the matrix, e has size cols, syndrome has size rows.
   Binary version (GF(2) — XOR only). */
void sparse_matvec(const SparseMatrix *H, const uint8_t *e, uint8_t *syndrome);

/* Count non-zero entries in a uint8_t vector */
int count_nonzero(const uint8_t *v, int size);

/* Compare two uint8_t vectors; returns 1 if equal, 0 otherwise */
int compare_vectors(const uint8_t *a, const uint8_t *b, int size);

#ifdef __cplusplus
}
#endif

#endif /* SPARSE_UTILS_H */
