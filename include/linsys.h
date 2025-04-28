#include "zeronp.h"
// #include "mkl_lapacke.h"

zeronp_int ZERONP(chol)(
    zeronp_int n,
    zeronp_float *H);

zeronp_int ZERONP(solve_lin_sys)(
    zeronp_int n,
    zeronp_int nrhs,
    zeronp_float *L,
    zeronp_float *b);
void ZERONP(cond)(
    zeronp_int n,
    zeronp_float *a,
    zeronp_float *cond);
void ZERONP(solve_general_lin_sys)(
    zeronp_int n,
    zeronp_float *a,
    zeronp_float *b);
zeronp_int ZERONP(solve_sys_lin_sys)(
    zeronp_int n,
    zeronp_float *a,
    zeronp_float *b,
    zeronp_int max_neg_eig);
zeronp_int ZERONP(least_square)(
    zeronp_int m,
    zeronp_int n,
    zeronp_float *A,
    zeronp_float *b);