# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: Duarte Mateus
"""

import galois
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def forward_elimination(A, b, GF,num_workers):
    n = len(A)
    for i in range(n):
        # Find the pivot
        max_row = max(range(i, n), key=lambda r: A[r, i])
        if A[max_row, i] == 0:
            raise ValueError("Matrix is singular!")

        # Swap the rows
        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]

        # Make the diagonal contain all 1s
        inv = GF(1)/A[i, i]
        A[i] = np.array([x * inv for x in A[i]],dtype=GF)
        b[i] = b[i] * inv

        # Eliminate column entries below the pivot
        def eliminate_row(j):
            factor =A[j, i]
            A[j] = np.array([A[j, k] - factor * A[i, k] for k in range(n)],dtype=GF)
            b[j] = b[j] - factor * b[i]

        # Eliminate rows below the pivot in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            executor.map(eliminate_row, range(i + 1, n))

    return A, b

def back_substitution(A, b, GF,num_workers):
    n = len(A)
    x = GF.Zeros(n)
    for i in reversed(range(n)):
        s=GF(0);
        for j in range(i + 1, n): 
            s+=A[i, j] * x[j] 
        x[i] = b[i] - s
    return x

# def back_substitution(A, b, GF, max_workers):
#     n=A.shape[0]
#     x = GF.Zeros(n)
#     def chunk_sum(i, start, end):
#          if start >= end:
#              return GF(0)
#          return np.array(np.array(A[i, start:end],dtype=GF) * np.array(x[start:end],dtype=GF),dtype=GF).sum()
    
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         for i in range(n - 1, -1, -1):
#             tail_start = i + 1
#             tail_len = n - tail_start
#             if tail_len <= 0:
#                 s = GF(0)
#             elif max_workers == 1:
#                 s =np.array(np.array(A[i, tail_start:n],dtype=GF) * np.array(x[tail_start:n],dtype=GF),dtype=GF).sum()
#             else:
#                 chunks = min(max_workers, tail_len)
#                 step = (tail_len + chunks - 1) // chunks
#                 futures = []
#                 for start in range(tail_start, n, step):
#                     end = min(n, start + step)
#                     futures.append(executor.submit(chunk_sum, i, start, end))
#                 s = GF(0)
#                 for f in futures:
#                     s += f.result()

#             x[i] = GF(b[i] - s) # unit diagonal assumed

#     return x

def gaussian_elimination(A, b, GF,num_workers):
    A, b = forward_elimination(A, b, GF,num_workers)
    return back_substitution(A, b, GF,num_workers)


def matrix_add_worker(A_slice, B_slice):
    return A_slice + B_slice

def parallel_matrix_add(A, B, num_workers):
    chunks_A = np.array_split(A, num_workers)
    chunks_B = np.array_split(B, num_workers)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(matrix_add_worker, chunk_A, chunk_B) for chunk_A, chunk_B in zip(chunks_A, chunks_B)]
        results = [future.result() for future in futures]
    
    result = np.vstack(results)
    return result


def minor(matrix, row, col):
    """Return the minor of the matrix excluding the specified row and column."""
    minor_matrix = np.delete(matrix, row, axis=0)
    minor_matrix = np.delete(minor_matrix, col, axis=1)
    return minor_matrix

def determinant_worker(GF,matrix, col, num_workers):
    """Calculate the cofactor expansion along the first row."""
    #sign = GF(1)
    sub_matrix = minor(matrix, 0, col)
    sub_det = parallel_determinant(GF,sub_matrix, num_workers)
    return matrix[0, col] * sub_det

def parallel_determinant(GF,matrix, num_workers):
    n = matrix.shape[0]
    
    if n == 1:
        return matrix[0, 0]
    elif n == 2:
        return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = np.array([executor.submit(determinant_worker, GF, matrix, col, num_workers) for col in range(n)],dtype=GF)
        results = np.array([future.result() for future in futures],dtype=GF)
    
    return results.sum()

def matrix_multiply_worker(A_slice, B):
    return np.dot(A_slice, B)

def parallel_matrix_multiply(GF,A, B, num_workers):
    # Split A into chunks
    n=A.shape[0]
    m=B.shape[1]
    chunks = np.array_split(A, num_workers)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Map the worker function to the chunks
        futures = [executor.submit(matrix_multiply_worker, chunk, B) for chunk in chunks]
        results = [future.result() for future in futures]
    
    # Concatenate the results from each worker
    result = GF(np.vstack(results))
    result=np.array([[GF(result[i,j]) for j in range(m)] for i in range(n)],dtype=GF)
    return result

def parallel_inverse(GF,A, num_workers):
    n, m = A.shape
    if n != m:
        raise ValueError("Matrix must be square!")
    
    I = GF.Identity(n)
 
    X = GF.Zeros((n, n))# will hold the columns of A^{-1}
    
    for j in range(n):
        b = I[:, j].copy()
        x = GF(gaussian_elimination(A.copy(), b, GF,num_workers) ) # keep original A intact
        X[:, j] = x

    X=np.array([[GF(X[i,j]) for j in range(n)] for i in range(n)],dtype=GF)

    return X

if __name__ == "__main__":
    # Define the Galois Field GF(2^m)
    m = 7
    GF = galois.GF(2**m)
    
    H =np.array([[ GF(1), GF(0), GF(7), GF(3)],
            [GF(2), GF(2), GF(0), GF(1)],
            [GF(0), GF(1), GF(1), GF(0)],[GF(1), GF(0), GF(0), GF(0)]
            ],dtype=GF)
  
    # Example matrix A and vector b
    A = np.array([[GF(1), GF(67), GF(0)], [GF(0), GF(61), GF(0)], [GF(0), GF(0), GF(1)]], dtype=GF)
    b = np.array([GF(1), GF(1), GF(0)], dtype=GF)

    X=parallel_inverse(GF, H, 3)
    print(parallel_matrix_multiply(H, X, 3))
    
 