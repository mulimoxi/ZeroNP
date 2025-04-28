#include "linsys.h"
#include "def_zeronp_lapack.h"

zeronp_int ZERONP(chol)(
    zeronp_int n,
    zeronp_float *H)
{
    zeronp_int info;
    char uploup = LAPACK_UPLOW_UP;

    dpotrf(&uploup, &n, H, &n, &info);
    // info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', n, H, n);
    return info; // 0 if successful
}

zeronp_int ZERONP(solve_lin_sys)(
    zeronp_int n,
    zeronp_int nrhs,
    zeronp_float *L,
    zeronp_float *b)
{ // This subroutine solve PSD linear system
    zeronp_int info;
    char uploup = LAPACK_UPLOW_UP;
    dpotrs(&uploup, &n, &nrhs, L, &n, b, &n, &info);

    // info = LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'U', n, nrhs, L, n, b, n);
    return info; // 0 if successful
}

void ZERONP(cond)(
    zeronp_int n,
    zeronp_float *a,
    zeronp_float *cond)
{
    zeronp_float norm = 0;
    for (zeronp_int i = 0; i < n; i++)
    {
        for (zeronp_int j = 0; i < n; i++)
        {
            norm += ABS(a[i * n + j]);
        }
    }
    // LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', n, a, n);
    // LAPACKE_dpocon(LAPACK_COL_MAJOR, 'U', n, a, n, norm, cond);

    zeronp_int info;
    char uploup = LAPACK_UPLOW_UP;
    zeronp_float *work = (zeronp_float *)zeronp_malloc(3 * n * sizeof(zeronp_float));
    zeronp_int *iwork = (zeronp_int *)zeronp_malloc(n * sizeof(zeronp_int));
    dpotrf(&uploup, &n, a, &n, &info);
    dpocon(&uploup, &n, a, &n, &norm, cond, work, iwork, &info);
    zeronp_free(work);
    zeronp_free(iwork);
}
// solve linear system of square matrix
void ZERONP(solve_general_lin_sys)(
    zeronp_int n,
    zeronp_float *a,
    zeronp_float *b)
{
    zeronp_int *ipiv = (zeronp_int *)zeronp_malloc(n * sizeof(zeronp_int));
    // LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, a, n, ipiv);
    // LAPACKE_dgetrs(LAPACK_COL_MAJOR, 'N', n, 1, a, n, ipiv, b, n);

    zeronp_int info;
    char trans = LAPACK_NOTRANS;
    zeronp_int nrhs = 1;

    dgetrf(&n, &n, a, &n, ipiv, &info);
    dgetrs(&trans, &n, &nrhs, a, &n, ipiv, b, &n, &info);

    zeronp_free(ipiv);
    return;
}

zeronp_int ZERONP(solve_sys_lin_sys)(
    zeronp_int n,
    zeronp_float *a,
    zeronp_float *b,
    zeronp_int max_neg_eig)
{
    // This subroutine solve general symmetric linear system
    zeronp_int info, i, neg_eig;
    neg_eig = 0;
    zeronp_int *ipiv = (zeronp_int *)zeronp_malloc(n * sizeof(zeronp_int));
    // LAPACKE_dsytrf(LAPACK_COL_MAJOR, 'L', n, a, n, ipiv);

    char uplolow = LAPACK_UPLOW_LOW;
    zeronp_int lwork = 8 * n;
    zeronp_float *work = (zeronp_float *)zeronp_malloc(MAX(1, lwork) * sizeof(zeronp_float));
    dsytrf(&uplolow, &n, a, &n, ipiv, work, &lwork, &info);
    zeronp_free(work);

    for (i = 0; i < n; i++)
    {
        if (a[i + i * n] < 0)
        {
            neg_eig++;
        }
    }
    if (neg_eig > max_neg_eig)
    {
        return -1;
    }
    // info = LAPACKE_dsytrs(LAPACK_COL_MAJOR, 'L', n, 1, a, n, ipiv, b, n);
    dsytrs(&uplolow, &n, &n, a, &n, ipiv, b, &n, &info);

    zeronp_free(ipiv);
    return info;
}

zeronp_int ZERONP(least_square)(
    zeronp_int m,
    zeronp_int n,
    zeronp_float *A,
    zeronp_float *b)
{
    // Solve the least square problem using SVD decompoisiton
    zeronp_int info, rank;
    zeronp_float *s = (zeronp_float *)zeronp_malloc(n * sizeof(zeronp_float));
    // info = LAPACKE_dgelss(LAPACK_COL_MAJOR, m, n, 1, A, m, b, MAX(m, n), s, 1e-8, &rank);

    zeronp_int nrhs = 1;
    zeronp_int ldb = MAX(m, n);
    zeronp_float rcond = 1e-8;
    zeronp_int min_lwork = 3 * MIN(m, n) + MAX(MAX(2 * MIN(m, n), MAX(m, n)), nrhs);
    zeronp_int lwork = 8 * MAX(n, min_lwork);
    zeronp_float *work = (zeronp_float *)zeronp_malloc(MAX(1, lwork) * sizeof(zeronp_float));
    dgelss(&m, &n, &nrhs, A, &m, b, &ldb, s, &rcond, &rank, work, &lwork, &info);
    zeronp_free(work);
    zeronp_free(s);

    return info;
}
