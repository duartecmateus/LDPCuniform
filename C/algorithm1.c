#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <pthread.h>

#include "gf16.h"
#include "gf16_gauss.h"


static void matvec(const gf16_t* A, const gf16_t* x, int m, int n, gf16_t* b) { 
	for (int i = 0; i < m; ++i) { gf16_t s = 0; for (int k = 0; k < n; ++k) { s = add(s, mult(A[i * n + k], x[k])); } b[i] = s; } }
	
// Thread task for scanning a block of columns and collecting those whose support is disjoint with v's support.
typedef struct {
	const gf16_t* H; // r x c, row-major
	const bool* v_supp; // length r: true where v[i] != 0
	int r;
	int c;
	int start_col; // inclusive
	int end_col; // exclusive
	int* out_indices;     // pre-allocated capacity = (end_col - start_col)
	int out_count;        // number of indices written
} ColumnTask;

static void* column_worker(void* argp) {
	ColumnTask* arg = (ColumnTask*)argp;
	int idx = 0;
	for (int j = arg->start_col; j < arg->end_col; ++j) {
    	bool disjoint = true;
    	const gf16_t* col_base = arg->H + j; // offset increments by c per row
    	for (int i = 0; i < arg->r; ++i) {
        	if (arg->v_supp[i]) {
            	if (col_base[i * arg->c] != 0) {
                	disjoint = false;
                	break;
            	}
        	}
    	}
    	if (disjoint) {
        	arg->out_indices[idx++] = j;
    	}
	}
	arg->out_count = idx;
	return NULL;
}

// Parallel version of: find_non_intersecting_columns(H, v)
static int find_non_intersecting_columns_pthreads(const gf16_t* H, int r, int c, const gf16_t* v,int num_workers, int** out_indices,int* out_count) {
	if (num_workers < 1) num_workers = 1;
	if (num_workers > c) num_workers = c;
	bool* v_supp = (bool*)malloc(sizeof(bool) * r);
	if (!v_supp) return -1;
	for (int i = 0; i < r; ++i) v_supp[i] = (v[i] != 0);

	pthread_t* tids = (pthread_t*)malloc(sizeof(pthread_t) * num_workers);
	ColumnTask* tasks = (ColumnTask*)malloc(sizeof(ColumnTask) * num_workers);
	if (!tids || !tasks) {
  	  free(v_supp);
  	  free(tids);
  	  free(tasks);
   	 return -1;
	}

	int base = c / num_workers;
	int rem = c % num_workers;
	int start = 0;

	for (int t = 0; t < num_workers; ++t) {
    	int count = base + (t < rem ? 1 : 0);
    	int end = start + count;
    	tasks[t].H = H;
    	tasks[t].v_supp = v_supp;
    	tasks[t].r = r;
    	tasks[t].c = c;
    	tasks[t].start_col = start;
    	tasks[t].end_col = end;
    	tasks[t].out_indices = (int*)malloc(sizeof(int) * (count > 0 ? count : 1));
    	tasks[t].out_count = 0;

    	if (count > 0) {
     	   pthread_create(&tids[t], NULL, column_worker, &tasks[t]);
    	} else {
    		// No columns to process in this thread
      	  tasks[t].out_count = 0;
   	 	}
   	 	start = end;
	}

	int total = 0;
	for (int t = 0; t < num_workers; ++t) {
    // Only join if thread actually had work
    	int count = tasks[t].end_col - tasks[t].start_col;
   		if (count > 0) pthread_join(tids[t], NULL);
    	total += tasks[t].out_count;
	 }

	 int* indices = (int*)malloc(sizeof(int) * (total > 0 ? total : 1));
	 if (!indices) {
 		for (int t = 0; t < num_workers; ++t) free(tasks[t].out_indices);
 	   	free(tasks);
    	free(tids);
    	free(v_supp);
    	return -1;
	}
	int pos = 0;
	for (int t = 0; t < num_workers; ++t) {
   		if (tasks[t].out_count > 0) {
        	memcpy(indices + pos, tasks[t].out_indices, sizeof(int) * tasks[t].out_count);
        	pos += tasks[t].out_count;
    	}
    	free(tasks[t].out_indices);
	}

	*out_indices = indices;
	*out_count = total;

	free(tasks);
	free(tids);
	free(v_supp);
	return 0;
}

// Constructs a square matrix by appending rows e_{li[k]} until rows == cols or li is exhausted.
// Returns 0 on success, 1 if unable to reach square (rows != cols after append), or -1 on alloc error.
static int construct_square_matrix(const gf16_t* H, int r, int c,const int* li, int li_len,gf16_t** H_out,int* r_out, int* added_out) {
	if (r >= c) {
		gf16_t* A = (gf16_t*)malloc(sizeof(gf16_t) * r * c);
		if (!A) return -1;
		memcpy(A, H, sizeof(gf16_t) * r * c);
		*H_out = A;
		*r_out = r;
		*added_out = 0;
		return (r == c) ? 0 : 1; // already tall; if equals, it's square
	}


	int need = c - r;
	int add = (li_len < need) ? li_len : need;
	int r2 = r + add;

	gf16_t* A = (gf16_t*)malloc(sizeof(gf16_t) * r2 * c);
	if (!A) return -1;

	// Copy original matrix
	memcpy(A, H, sizeof(gf16_t) * r * c);

	// Append standard-basis rows
	for (int k = 0; k < add; ++k) {
    	gf16_t* row = A + (r + k) * c;
    	memset(row, 0, sizeof(gf16_t) * c);
    	int j = li[k];
    	if (j >= 0 && j < c) {
        	row[j] = (gf16_t)1;
    	} else {
        	// Out-of-range index; keep row zero
    	}
	}

	*H_out = A;
	*r_out = r2;
	*added_out = add;

	return (r2 == c) ? 0 : 1;
}

static gf16_t* extend_v(const gf16_t* v, int r, int new_r) {
	gf16_t* vt = (gf16_t*)malloc(sizeof(gf16_t) * new_r);
	if (!vt) return NULL;
	int i = 0;
	for (; i < r && i < new_r; ++i) vt[i] = v[i];
	for (; i < new_r; ++i) vt[i] = (gf16_t)0;
	return vt;
}

// alg1(H, v, num_workers) => calls gaussian elimination over GF(16).
// Returns 0 on success. x_out is allocated and must be freed by the caller.
int alg1(const gf16_t* H, int r, int c, const gf16_t* v, int num_workers, gf16_t** x_out) {
	int* non_inter_cols = NULL;
	int non_inter_count = 0;


	if (find_non_intersecting_columns_pthreads(H, r, c, v, num_workers, &non_inter_cols, &non_inter_count) != 0) {
   	 	fprintf(stderr, "Error: failed to compute non-intersecting columns.\n");
    	return -1;
	}

	gf16_t* A = NULL;
	int r2 = 0;
	int added = 0;
	int csq = construct_square_matrix(H, r, c, non_inter_cols, non_inter_count, &A, &r2, &added);
	free(non_inter_cols);

	if (csq < 0) {
    	fprintf(stderr, "Error: allocation while constructing square matrix.\n");
    	return -1;
	}
	if (csq > 0) {
		fprintf(stderr, "Error: could not reach a square matrix (rows=%d, cols=%d). Not enough columns to append. Needs to use algorithm 2\n", r2, c);
   	 	free(A);
    	return -1;
	}

	gf16_t* b = extend_v(v, r, r2);
	if (!b) {
   		fprintf(stderr, "Error: allocation for extended vector.\n");
   	  	free(A);
    	return -1;
	}

	gf16_init(); // ensure GF(16) tables are initialized

	gf16_t* sol = (gf16_t*)malloc(sizeof(gf16_t) * c);
	if (!sol) {
 	    fprintf(stderr, "Error: allocation for solution vector.\n");
   	  	free(A);
      	free(b);
    	return -1;
	}

	// gf16_gaussian_elimination modifies A and b in-place.
	int rc = gf16_gaussian_elimination(A, b, c, num_workers, sol);

	free(A);
	free(b);

	if (rc != 0) {
    	fprintf(stderr, "Gaussian elimination failed (singular matrix?).\n");
    	free(sol);
    	return -1;
	}

	*x_out = sol;
	return 0;
}

// Helpers for demo
static void print_vector_gf16(const gf16_t* v, int n, const char* label) {
	if (label) printf("%s", label);
	printf("[");
	for (int i = 0; i < n; ++i) {
		printf("%u", (unsigned)v[i]);
		if (i + 1 < n) printf(", ");
	}
	printf("]\n");
}

static void print_matrix_gf16(const gf16_t* A, int r, int c, const char* label) {
	if (label) printf("%s\n", label);
	for (int i = 0; i < r; ++i) {
		printf("[");
		for (int j = 0; j < c; ++j) {
			printf("%u", (unsigned)A[i * c + j]);
			if (j + 1 < c) printf(", ");
		}
		printf("]\n");
	}
}

int main(void) {
// Example usage matching the Python sample (values interpreted in GF(16))
// H = [[1,2,0,3],
// [2,2,0,1],
// [2,1,1,0]]
// v = [3,1,0]
int r = 8, c = 10;
gf16_t H[] =  {
0x3,0x0,0x4,0x0,0x0,0x0,0x0,0xE,0x4,0x0,
0xE,0x0,0x0,0x4,0x0,0x6,0x0,0x0,0xA,0x0,
0x2,0x0,0x0,0x0,0x5,0x0,0xC,0x0,0x0,0x3,
0x0,0x1,0x0,0x2,0x0,0x0,0x0,0x5,0xC,0x0,
0x0,0xA,0x0,0x0,0xA,0x0,0x0,0xB,0x0,0x9,
0x0,0xA,0x0,0x0,0x0,0xD,0x1,0x0,0x0,0xC,
0x0,0x0,0x5,0x5,0x0,0x0,0xC,0x0,0x0,0x0,
0x0,0x0,0x3,0x0,0x3,0x4,0x0,0x0,0x0,0x0
};
		
gf16_t e[] = { 0, 0,3, 3 ,0, 0 ,0 ,0,0,0};

gf16_t v[r];

matvec(H, e, r, c, v);
	
print_vector_gf16(v,r,"");


// Find non-intersecting columns (demo print)
int* cols = NULL;
int cols_count = 0;
if (find_non_intersecting_columns_pthreads(H, r, c, v, 4, &cols, &cols_count) != 0) {
    fprintf(stderr, "Error computing non-intersecting columns.\n");
    return 1;
}
printf("Non-intersecting columns: [");
for (int i = 0; i < cols_count; ++i) {
    printf("%d", cols[i]);
    if (i + 1 < cols_count) printf(", ");
}
printf("]\n");
free(cols);


gf16_t* x = NULL;
int rc = alg1(H, r, c, v, 4, &x);
if (rc == 0) {
    print_vector_gf16(x, c, "Solution x = ");
    free(x);
} else {
    fprintf(stderr, "alg1 failed.\n");
    return 2;
}

return 0;
}