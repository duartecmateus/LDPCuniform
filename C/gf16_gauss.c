#define _XOPEN_SOURCE 700 
#include <pthread.h> 
#include <stdlib.h> 
#include <string.h> 
#include <stdio.h>

#include "gf16.h" 
#include "gf16_gauss.h"

typedef struct { 
	gf16_t* A; //matrix represented as concatenated lines
	gf16_t* b; 
	int n; 
	int i; // pivot row/col index 
	int row_start; // inclusive 
	int row_end; // exclusive 
} elim_task_t;


void* elim_worker(void* arg) { 
	elim_task_t* t = (elim_task_t*)arg; 
	const int n = t->n; 
	const int i = t->i; 
	gf16_t* A = t->A; 
	gf16_t* b = t->b;
	
	// Pivot row (normalized before threads are launched)
	gf16_t* prow = &A[i * n];
	const gf16_t pb = b[i];

	for (int j = t->row_start; j < t->row_end; ++j) {
	    gf16_t factor = A[j * n + i];
	    if (factor == 0) continue;
	    // Row_j = Row_j - factor * Row_i
	    // Start at k=i (columns < i are already 0 below the pivot)
	    for (int k = i; k < n; ++k) {
	        gf16_t val = A[j * n + k];
	        gf16_t pk = prow[k];
	        A[j * n + k] = add(val, mult(factor, pk));
	    }
	    b[j] = add(b[j], mult(factor, pb));
	}
	return NULL;
}



void swap_rows(gf16_t* A, gf16_t* b, int n, int r1, int r2) { 
	if (r1 == r2) return; 
	for (int k = 0; k < n; ++k) { 
		gf16_t tmp = A[r1 * n + k];
		A[r1 * n + k] = A[r2 * n + k];
		A[r2 * n + k] = tmp; 
	} 
	gf16_t tb = b[r1]; 
	b[r1] = b[r2]; 
	b[r2] = tb; 
}

int gf16_forward_elimination(gf16_t* A, gf16_t* b, int n, int num_workers) { 
	if (num_workers < 1) num_workers = 1; 
	gf16_init();
	for (int i = 0; i < n; ++i) {
	    // Find a pivot row with A[r,i] != 0
	    int pivot_row = -1;
	    for (int r = i; r < n; ++r) {
	        if (A[r * n + i] != 0) { pivot_row = r; break; }
	    }
	    if (pivot_row == -1) {
	        // Singular matrix
	        return -1;
	    }

	    // Swap to put pivot on row i
	    swap_rows(A, b, n, i, pivot_row);

	    // Normalize pivot row so that A[i,i] = 1
	    gf16_t piv = A[i * n + i];
	    gf16_t invp = inv(piv); // piv is non-zero here
	    for (int k = i; k < n; ++k) {
	        A[i * n + k] = mult(A[i * n + k], invp);
	    }
	    b[i] = mult(b[i], invp);

	    // Eliminate rows below i in parallel
	    int start = i + 1;
	    int total = n - start;
	    if (total <= 0) continue;

	    int tcount = num_workers;
	    if (tcount > total) tcount = total;

	    pthread_t* tids = (pthread_t*)malloc(sizeof(pthread_t) * tcount);
	    elim_task_t* tasks = (elim_task_t*)malloc(sizeof(elim_task_t) * tcount);

	    int chunk = (total + tcount - 1) / tcount;
	    for (int t = 0; t < tcount; ++t) {
	        int rs = start + t * chunk;
	        int re = rs + chunk;
	        if (re > n) re = n;
	        tasks[t].A = A;
	        tasks[t].b = b;
	        tasks[t].n = n;
	        tasks[t].i = i;
	        tasks[t].row_start = rs;
	        tasks[t].row_end = re;
	        pthread_create(&tids[t], NULL, elim_worker, &tasks[t]);
	    }
	    for (int t = 0; t < tcount; ++t) {
	        pthread_join(tids[t], NULL);
	    }
	    free(tasks);
	    free(tids);
	}

	return 0;
}

typedef struct { const gf16_t* A; const gf16_t* x; // x_out (already computed for j > i) 
	            int n; int i; // row index 
				int j_start; // inclusive 
				int j_end; // exclusive 
				gf16_t partial; // thread’s partial sum 
} dot_task_t;

static void* dot_worker(void* arg) { 
	dot_task_t* t = (dot_task_t*)arg; 
	const gf16_t* row = &t->A[t->i * t->n]; 
	gf16_t s = 0; 
	for (int j = t->j_start; j < t->j_end; ++j) { 
		s = add(s, mult(row[j], t->x[j])); 
	} 
	t->partial = s; 
	return NULL;
}

int gf16_back_substitution_mt(const gf16_t* A, const gf16_t* b, int n, int num_workers, gf16_t* x_out) { 
	if (num_workers < 1) num_workers = 1;

	for (int i = n - 1; i >= 0; --i) {
    	int j0 = i + 1;
    	int total = n - j0;

    	if (total <= 0) {
        	// Nothing to sum on this row
        	x_out[i] = b[i];
        	continue;
    	}

    	if (num_workers == 1 || total < 2) {
        // Fast sequential path
        	gf16_t s = 0;
        	for (int j = j0; j < n; ++j) {
            	s = add(s, mult(A[i * n + j], x_out[j]));
        	}
        	x_out[i] = add(b[i], s);
        	continue;
    	}

    	int tcount = num_workers;
    	if (tcount > total) tcount = total;

    	pthread_t* tids = (pthread_t*)malloc(sizeof(pthread_t) * tcount);
    	dot_task_t* tasks = (dot_task_t*)malloc(sizeof(dot_task_t) * tcount);

    	int chunk = (total + tcount - 1) / tcount;

    	for (int t = 0; t < tcount; ++t) {
        	int js = j0 + t * chunk;
        	int je = js + chunk;
        	if (je > n) je = n;

        	tasks[t].A = A;
        	tasks[t].x = x_out;
        	tasks[t].n = n;
        	tasks[t].i = i;
        	tasks[t].j_start = js;
        	tasks[t].j_end = je;
        	tasks[t].partial = 0;

        	pthread_create(&tids[t], NULL, dot_worker, &tasks[t]);
    	}

    	for (int t = 0; t < tcount; ++t) {
        	pthread_join(tids[t], NULL);
    	}

    	gf16_t s = 0;
    	for (int t = 0; t < tcount; ++t) {
        	s = add(s, tasks[t].partial);
    	}

    	x_out[i] = add(b[i], s);

    	free(tasks);
    	free(tids);
	}
return 0;
}

int gf16_gaussian_elimination(gf16_t* A, gf16_t* b, int n, int num_workers, gf16_t* x_out) { 
	int rc = gf16_forward_elimination(A, b, n, num_workers); 
	if (rc != 0) return rc; 
	return gf16_back_substitution_mt(A, b, n, num_workers, x_out);; 
}

