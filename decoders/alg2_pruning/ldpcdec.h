#ifndef LDPCDEC_H
#define LDPCDEC_H

#define GF_BITS_PER_SYMBOL 8
#define WEIGHT_BUCKETS 3
#define MAX_LINE 65536
#define MAX_ITERS 50
#define LLR_MAG 3.0

int ldpc_decode_min_sum(int M, int N, int **H,
                        const double *llr_in,
                        int *decoded,
                        int *iters_used);

int ldpc_decode_spa(int M, int N, int **H,
                    const double *llr_in,
                    int *decoded,
                    int *iters_used);

#endif
