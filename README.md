# LDPC decoder

This repository contains a prototype of the LDPC decoder presented in [1]. We provide:

- Algorithm 1: Decoder under ideal conditions — Python and C
- Algorithm 2: Las Vegas decoder — C
- Algorithm 3: LDPC generator — Mathematica (with C-compatible output)

If you use this code, please cite [1].

In [1], we used Csanky’s algorithm [2], adapted to finite fields [3]. However, for this application it is simpler and more efficient to use the NC algorithm for solving linear equations proposed in [4], since we only need to solve a linear system rather than invert a full matrix.

We migrated the Python prototype to C for two reasons:

1. The Python GF library does not integrate well with NumPy arrays, making it cumbersome to work with matrices of GF-typed objects.
2. Python’s threading model does not provide true parallelism for CPU-bound code, which undermines the purpose of these algorithms.

The current C prototype targets $GF(2^4)$ but can be straightforwardly adapted to other Galois fields (ongoing work). To run the code, copy the C folder and run make. The Mathematica package will also be migrated to C; however, this requires implementing additional libraries not readily available in C and will be done for specific, practically useful finite fields. We also plan to adapt these algorithms to CUDA; however, because CUDA does not implement the PRAM model assumed by an NC algorithm, the migration is not straightforward.

The C implementations generate the multiplication and inverse tables for the Galois field at startup. These tables can be saved to disk and reloaded later, although for $GF(2^4)$ the load time and generation time are essentially the same.



[1] D. Mateus and R. Chaves, A Fast Parallel Decoder for LDPC Codes Suitable for Burst Errors, In Proceedings of the Information Theory Workshop, IEEE (2025).

[2] L. Csanky, Fast parallel matrix inversion algorithms, SIAM Journal of Computing, 5 (1976), 618-623.

[3] A. Schönhage, Fast Parallel Computation of Characteristic Polynomials by Leverrier's Power Sum Method Adapted to Fields of Finite Characteristic, In Proceedings of the 3rd Symposium on Symbolic and Algebraic Computation , ACM,  (1976), 124–128. 

[4] A. Borodin, J. von zur Gathen, and J.E. Hopcroft, Fast parallel matrix and GCD computations, Information and Control 52 (1982), 241-256.
