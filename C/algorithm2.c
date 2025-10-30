#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <pthread.h>

#include <time.h>
#include <unistd.h>



#include "gf16.h"
#include "gf16_gauss.h"

static void seed_rng_once(void) {
unsigned seed = (unsigned)time(NULL) ^ (unsigned)getpid();
srand(seed);
}

static int hammingweight(const gf16_t* v, int n) {
	int i=0,r=0;
	while(i<n) {
		if(v[i]!=0) r++;
		i++;
	}
	return r;
}

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
static int construct_square_matrix(const gf16_t* H, int r, int c, const int* li, int li_len, gf16_t** H_out, int* r_out, int* added_out){
	if (r >= c) {
		gf16_t* A = (gf16_t*)malloc(sizeof(gf16_t) * (size_t)r * (size_t)c);
		if (!A) return -1;
		memcpy(A, H, sizeof(gf16_t) * (size_t)r * (size_t)c);
		*H_out = A;
		*r_out = r;
		*added_out = 0;
		return (r == c) ? 0 : 1;
	}

// r < c: we will make it square by adding rows
	const int need = c - r;

// Mark valid unique indices from li
	uint8_t* mark = (uint8_t*)calloc((size_t)c, 1);
	if (!mark) return -1;

	int valid_li_count = 0;
	if (li && li_len > 0) {
   		for (int k = 0; k < li_len; ++k) {
      	  int j = li[k];
       	  if (j >= 0 && j < c && !mark[j]) {
            mark[j] = 1;
            ++valid_li_count;
          }
   	 	}
	}

	const int add_from_li = (valid_li_count < need) ? valid_li_count : need;
	const int final_rows = r + need; // will be exactly c
	gf16_t* A = (gf16_t*)malloc(sizeof(gf16_t) * (size_t)final_rows * (size_t)c);
	if (!A) { free(mark); return -1; }

	// Copy original matrix
	memcpy(A, H, sizeof(gf16_t) * (size_t)r * (size_t)c);

	// Append rows from li (preserving order, skipping duplicates/out-of-range)
	int appended = 0;
	if (add_from_li > 0) {
  		uint8_t* used_li = (uint8_t*)calloc((size_t)c, 1);
   	 	if (!used_li) { free(A); free(mark); return -1; }

    	for (int k = 0; k < li_len && appended < add_from_li; ++k) {
       	 	int j = li[k];
       	 	if (j >= 0 && j < c && !used_li[j]) {
            	used_li[j] = 1;
            	gf16_t* row = A + (size_t)(r + appended) * (size_t)c;
            	memset(row, 0, sizeof(gf16_t) * (size_t)c);
            	row[j] = (gf16_t)1;
            	++appended;
        	}
    	}
   	 	free(used_li);
	}

	// If not enough yet, append random canonical-basis rows from complement of li
	int remain = need - appended;
	if (remain > 0) {
		printf("Adding %d random row(s)\n",remain);
    	const int comp_count = c - valid_li_count;
    	int* comp = (int*)malloc(sizeof(int) * (size_t)comp_count);
    	if (!comp) { free(A); free(mark); return -1; }

    	int idx = 0;
    	for (int j = 0; j < c; ++j) {
       	 	if (!mark[j]) comp[idx++] = j;
    	}

    	// Fisher-Yates shuffle of complement
    	for (int i = comp_count - 1; i > 0; --i) {
       	 	int t = rand() % (i + 1);
        	int tmp = comp[i]; comp[i] = comp[t]; comp[t] = tmp;
   	 	}

    	// Append first 'remain' indices from shuffled complement
   	 	for (int i = 0; i < remain; ++i) {
     	  	int j = comp[i];
      	  	gf16_t* row = A + (size_t)(r + appended + i) * (size_t)c;
       	 	memset(row, 0, sizeof(gf16_t) * (size_t)c);
        	row[j] = (gf16_t)1;
			printf("%d\n",j);
    	}
    	appended += remain;
    	free(comp);
	}

	free(mark);

	*H_out = A;
	*r_out = r + appended;   // should be c
	*added_out = appended;

	return (*r_out == c) ? 0 : 1;
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
int alg2(const gf16_t* H, int r, int c, const gf16_t* v, int num_workers, gf16_t** x_out, int max_tries, int max_error) {
	int* non_inter_cols = NULL;
	int non_inter_count = 0;
	gf16_t* sol = NULL;
	gf16_t* b = NULL;
	
 	// ensure GF(16) tables are initialized
	gf16_init();

	if (find_non_intersecting_columns_pthreads(H, r, c, v, num_workers, &non_inter_cols, &non_inter_count) != 0) {
   	 	fprintf(stderr, "Error: failed to compute non-intersecting columns.\n");
    	return -1;
	}
    int num_tries=0;
	while(num_tries<max_tries){
		gf16_t* A = NULL;
		int r2 = 0;
		int added = 0;
		int csq = construct_square_matrix(H, r, c, non_inter_cols, non_inter_count, &A, &r2, &added);
		

		if (csq < 0) {
    		fprintf(stderr, "Error: allocation while constructing square matrix.\n");
    		return -1;
		}
		if (csq > 0) {
			fprintf(stderr, "Error: could not reach a square matrix (rows=%d, cols=%d). Not enough columns to append. Needs to use algorithm 2\n", r2, c);
   	 		free(A);
    		return -1;
		}

		
		b = extend_v(v, r, r2);
		if (!b) {
   				fprintf(stderr, "Error: allocation for extended vector.\n");
   	  	  	    free(A);
    			return -1;
		}
		sol = (gf16_t*)malloc(sizeof(gf16_t) * c);
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
    		printf("Gaussian elimination failed (singular matrix?), next try.\n");
			free(sol);
 		} 
		else if(hammingweight(sol,c)>max_error) {
			printf("Error to large, next try.\n");
			free(sol);
		}
		else {
			*x_out = sol;
			//free(b);
			free(non_inter_cols);
			return 0;
		}
		num_tries++;
	}
	//free(b);
	free(non_inter_cols);
	free(sol);
	return -1;
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
	
seed_rng_once();
// Example usage matching the Python sample (values interpreted in GF(16))
// H = [[1,2,0,3],
// [2,2,0,1],
// [2,1,1,0]]
// v = [3,1,0]


int r = 9, c = 11;
gf16_t H[] =  {
	4, 0, 0, 15, 0, 0, 9, 0, 0, 5, 0, 
	6, 0, 0, 0,  8, 0, 0, 2, 0, 0, 6, 
	2, 0, 0, 0, 0, 14, 0, 0, 5, 0, 0, 
	0, 1, 0, 3, 0, 0, 0, 0, 2, 0, 11,
	0, 7, 0, 0, 6, 0, 6, 0, 0, 0, 0, 
	0, 12, 0, 0, 0, 4, 0, 1, 0, 14, 0,
	0, 0, 3, 9, 0, 0, 0, 11, 0, 0, 0,
	0, 0, 14, 0, 10, 0, 0, 0, 4, 6, 0, 
	0, 0, 15, 0, 0, 6, 7, 0, 0, 0, 5
};
		
gf16_t e[] = { 3, 0,0, 0 ,0, 0 ,0 ,0 ,0,0,5};

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
int rc = alg2(H, r, c, v, 4, &x, 20, 2);
if (rc == 0) {
    print_vector_gf16(x, c, "Solution x = ");
    free(x);
} else {
    fprintf(stderr, "alg2 failed.\n");
    return 2;
}

return 0;
}