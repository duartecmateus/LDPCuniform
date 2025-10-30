import numpy as np
from concurrent.futures import ThreadPoolExecutor
import galois
from parallelAlg import gaussian_elimination

# Define the field GF(2)
GF = galois.GF(2**2)

def check_support_no_intersection(column, v):
    """Check if the support of the column does not intersect with the support of v."""
    column_support = set(np.where(column != GF(0))[0])
    v_support = set(np.where(v != GF(0))[0])
    return column_support.isdisjoint(v_support)

def find_non_intersecting_columns(H, v):
    num_columns = H.shape[1]
    non_intersecting_columns = []

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(check_support_no_intersection, H[:, j], v): j for j in range(num_columns)}
        
        for future in futures:
            column_index = futures[future]
            if future.result():
                non_intersecting_columns.append(column_index)
    
    return non_intersecting_columns

def add_row(H, n, li_element, index):
    """Function to add a row at a specific index."""
    new_row = np.array([GF(0) for _ in range(n)],dtype=GF)
    new_row[li_element] = GF(1)
    return new_row

def construct_square_matrix(H, li):
    r, n = H.shape
    print("Initial number of rows:", r)
    print("Number of columns:", n)
    print("Length of list li:", len(li))
    
    with ThreadPoolExecutor() as executor:
        # Create a list of future objects for the row creation tasks
        futures = [executor.submit(add_row, H, n, li[i], i) for i in range(len(li))]

        # As each future completes, add the resulting row to H
        for future in futures:
            new_row = future.result()
            H = np.vstack([H, new_row])
            r += 1
            if r >= n:
                break
    return H

def extendv(v,n):
    vt=np.array(GF.Zeros(n),dtype=GF)
    m=v.shape[0]
    for i in range(m):
        vt[i]=v[i]
    for i in range(m,n):
        vt[i]=GF(0)
    return vt
    

def alg1(H,v,num_workers):
    result = find_non_intersecting_columns(H, v)
    H=construct_square_matrix(H, result)
    print(H)
    n=H.shape[0]
    vt=extendv(v,n)
    print(n)
    print(len(vt))
    print(vt)
    return gaussian_elimination(H, vt, GF, num_workers)
    

# Example usage
H =np.array([[ GF(1), GF(2), GF(0), GF(3)],
        [GF(2), GF(2), GF(0), GF(1)],
        [GF(2), GF(1), GF(1), GF(0)]],dtype=GF)

v = np.array([GF(3), GF(1), GF(0)],dtype=GF)
result = find_non_intersecting_columns(H, v)
print(result)
v[2]

print(alg1(H,v,4))

