#include "sparse_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

int load_sparse_matrix(const char *filename, SparseMatrix *mat)
{
    if (!mat) return 0;

    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "[ERROR] Could not open '%s': %s\n", filename, strerror(errno));
        return 0;
    }

    int rows, cols, c, field;
    if (fscanf(f, "%d", &rows) != 1 ||
        fscanf(f, "%d", &cols) != 1 ||
        fscanf(f, "%d", &c) != 1 ||
        fscanf(f, "%d", &field) != 1) {
        fprintf(stderr, "[ERROR] Failed to read header from '%s'\n", filename);
        fclose(f);
        return 0;
    }

    /* consume the rest of the 4th line */
    int ch;
    while ((ch = fgetc(f)) != '\n' && ch != EOF) {}

    if (rows <= 0 || cols <= 0 || (c < 0) || (c > rows && c != 0)) {
        fprintf(stderr, "[ERROR] Invalid header in '%s': m=%d n=%d c=%d\n",
                filename, rows, cols, c);
        fclose(f);
        return 0;
    }
    if (field != 0 && field != 1 && field != 2 && field != 3) {
        fprintf(stderr, "[ERROR] Invalid field=%d in '%s' (expected 0=binary, 1=GF256, 2=GF16, 3=GF4)\n",
                field, filename);
        fclose(f);
        return 0;
    }

    SparseCol *col = (SparseCol *)calloc((size_t)cols, sizeof(SparseCol));
    if (!col) {
        fprintf(stderr, "[ERROR] Allocation failed\n");
        fclose(f);
        return 0;
    }

    char *line = NULL;
    size_t line_len = 0;

    for (int j = 0; j < cols; ++j) {
        if (getline(&line, &line_len, f) < 0) {
            fprintf(stderr, "[ERROR] Unexpected EOF at column %d in '%s'\n", j, filename);
            free(line);
            for (int k = 0; k < j; ++k) {
                free(col[k].rows);
                free(col[k].values);
            }
            free(col);
            fclose(f);
            return 0;
        }

        col[j].c = c;
        if (c > 0) {
            col[j].rows = (int *)malloc((size_t)c * sizeof(int));
            col[j].values = (uint8_t *)malloc((size_t)c * sizeof(uint8_t));
        } else {
            /* Variable column weight: allocate temp buffer, count entries first */
            col[j].rows = NULL;
            col[j].values = NULL;
        }

        if ((c > 0 && (!col[j].rows || !col[j].values)) ||
            (c == 0 && !line)) {
            fprintf(stderr, "[ERROR] Allocation failed at column %d\n", j);
            free(line);
            for (int k = 0; k <= j; ++k) {
                free(col[k].rows);
                free(col[k].values);
            }
            free(col);
            fclose(f);
            return 0;
        }

        int entries = 0;
        char *saveptr;
        char *token = strtok_r(line, " \t\r\n", &saveptr);

        if (c == 0) {
            /* Variable column weight: count first, then allocate */
            int count = 0;
            char *line_copy = strdup(line);
            if (!line_copy) {
                fprintf(stderr, "[ERROR] strdup failed at column %d\n", j);
                free(line);
                for (int k = 0; k < j; ++k) {
                    free(col[k].rows); free(col[k].values);
                }
                free(col); fclose(f); return 0;
            }
            char *ct = strtok_r(line_copy, " \t\r\n", &saveptr);
            while (ct) { ++count; ct = strtok_r(NULL, " \t\r\n", &saveptr); }
            free(line_copy);

            col[j].c = count;
            col[j].rows = (int *)malloc((size_t)count * sizeof(int));
            col[j].values = (uint8_t *)malloc((size_t)count * sizeof(uint8_t));
            if (!col[j].rows || !col[j].values) {
                fprintf(stderr, "[ERROR] Allocation failed at column %d\n", j);
                free(line);
                for (int k = 0; k <= j; ++k) {
                    free(col[k].rows); free(col[k].values);
                }
                free(col); fclose(f); return 0;
            }

            /* Re-parse with allocated arrays */
            saveptr = NULL;
            token = strtok_r(line, " \t\r\n", &saveptr);
            while (token && entries < count) {
                int row; unsigned val;
                if (sscanf(token, "%d:%u", &row, &val) != 2) { /* error */ }
                if (row < 0 || row >= rows || (field == 0 && val != 1) ||
                    (field == 1 && (val == 0 || val > 255))) { /* error */ }
                col[j].rows[entries] = row;
                col[j].values[entries] = (uint8_t)val;
                ++entries;
                token = strtok_r(NULL, " \t\r\n", &saveptr);
            }
        } else {
            /* Fixed column weight c */
            while (token && entries < c) {
                int row; unsigned val;
                if (sscanf(token, "%d:%u", &row, &val) != 2) { /* ... */ }
                if (row < 0 || row >= rows || (field == 0 && val != 1) ||
                    (field == 1 && (val == 0 || val > 255))) { /* ... */ }
                col[j].rows[entries] = row;
                col[j].values[entries] = (uint8_t)val;
                ++entries;
                token = strtok_r(NULL, " \t\r\n", &saveptr);
            }
        }

        if (c > 0 && entries != c) {
            fprintf(stderr, "[ERROR] Column %d has %d entries, expected %d\n",
                    j, entries, c);
            free(line);
            for (int k = 0; k <= j; ++k) {
                free(col[k].rows);
                free(col[k].values);
            }
            free(col);
            fclose(f);
            return 0;
        }
    }

    free(line);
    fclose(f);

    mat->rows = rows;
    mat->cols = cols;
    mat->c = c;
    mat->field = field;
    mat->col = col;
    return 1;
}

void free_sparse_matrix(SparseMatrix *mat)
{
    if (!mat || !mat->col) return;
    for (int j = 0; j < mat->cols; ++j) {
        free(mat->col[j].rows);
        free(mat->col[j].values);
    }
    free(mat->col);
    mat->col = NULL;
}

/* GF(2) version of matvec, used by the binary pruning test.
   XOR only — no gf256 dependency. */
void sparse_matvec(const SparseMatrix *H, const uint8_t *e, uint8_t *syndrome)
{
    memset(syndrome, 0, (size_t)H->rows * sizeof(uint8_t));

    for (int j = 0; j < H->cols; ++j) {
        if (e[j] == 0) continue;
        const SparseCol *sc = &H->col[j];
        for (int k = 0; k < sc->c; ++k) {
            int r = sc->rows[k];
            syndrome[r] ^= 1;
        }
    }
}

int count_nonzero(const uint8_t *v, int size)
{
    int count = 0;
    for (int i = 0; i < size; ++i) {
        if (v[i] != 0) ++count;
    }
    return count;
}

int compare_vectors(const uint8_t *a, const uint8_t *b, int size)
{
    for (int i = 0; i < size; ++i) {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}
