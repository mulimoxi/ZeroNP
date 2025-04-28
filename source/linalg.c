#include "linalg.h"
// #include "mkl.h"
// #include<osqp.h>
/* y = mean(x) */
zeronp_float ZERONP(vec_mean)(
    zeronp_float *x,
    zeronp_int len)
{
      if (x == ZERONP_NULL || len <= 0)
      {
            return 0;
      }
      if (len <= 0 || x == ZERONP_NULL)
      {
            printf("invalid ZERONP(vec_mean) parameter");
            return -1;
      }
      zeronp_float y = 0;

      for (int i = 0; i < len; i++)
      {
            y += x[i];
      }

      return y / len;
}

/* x = b*a */
void ZERONP(set_as_scaled_array)(
    zeronp_float *x,
    const zeronp_float *a,
    const zeronp_float b,
    zeronp_int len)
{
      if (x == ZERONP_NULL || a == ZERONP_NULL || len <= 0)
      {
            return;
      }

      zeronp_int i;
      for (i = 0; i < len; ++i)
      {
            x[i] = b * a[i];
      }
}

/* x = sqrt(v) */
void ZERONP(set_as_sqrt)(
    zeronp_float *x,
    const zeronp_float *v,
    zeronp_int len)
{
      if (x == ZERONP_NULL || v == ZERONP_NULL || len <= 0)
      {
            return;
      }

      zeronp_int i;
      for (i = 0; i < len; ++i)
      {
            x[i] = SQRTF(v[i]);
      }
}

/* x = v.^2 */
void ZERONP(set_as_sq)(
    zeronp_float *x,
    const zeronp_float *v,
    zeronp_int len)
{
      if (x == ZERONP_NULL || v == ZERONP_NULL || len <= 0)
      {
            return;
      }

      zeronp_int i;
      for (i = 0; i < len; ++i)
      {
            x[i] = v[i] * v[i];
      }
}

/* a *= b */
void ZERONP(scale_array)(
    zeronp_float *a,
    const zeronp_float b,
    zeronp_int len)
{
      if (a == ZERONP_NULL || len <= 0)
      {
            return;
      }

      zeronp_int i;
      for (i = 0; i < len; ++i)
      {
            a[i] *= b;
      }
}

/* x'*y */
zeronp_float ZERONP(dot)(
    const zeronp_float *x,
    const zeronp_float *y,
    zeronp_int len)
{
      if (x == ZERONP_NULL || y == ZERONP_NULL || len <= 0)
      {
            return 0;
      }

      zeronp_int i;
      zeronp_float ip = 0.0;
      for (i = 0; i < len; ++i)
      {
            ip += x[i] * y[i];
      }
      return ip;
}

/* ||v||_2^2 */
zeronp_float ZERONP(norm_sq)(
    const zeronp_float *v,
    zeronp_int len)
{
      if (v == ZERONP_NULL || len <= 0)
      {
            return 0;
      }

      zeronp_int i;
      zeronp_float nmsq = 0.0;
      for (i = 0; i < len; ++i)
      {
            nmsq += v[i] * v[i];
      }
      return nmsq;
}

/* ||v||_2 */
zeronp_float ZERONP(norm)(
    const zeronp_float *v,
    zeronp_int len)
{
      if (v == ZERONP_NULL || len <= 0)
      {
            return 0;
      }

      return SQRTF(ZERONP(norm_sq)(v, len));
}

/* ||x||_1 */
zeronp_float ZERONP(norm_1)(
    const zeronp_float *x,
    const zeronp_int len)
{
      if (x == ZERONP_NULL || len <= 0)
      {
            return 0;
      }

      zeronp_float result = 0;
      for (int i = 0; i < len; i++)
      {
            result += ABS(x[i]);
      }
      return result;
}

/*the absolute value of the largest component of x*/
zeronp_float ZERONP(cone_norm_1)(
    const zeronp_float *x,
    const zeronp_int len)
{
      if (x == ZERONP_NULL || len <= 0)
      {
            return 0;
      }

      zeronp_int i;
      zeronp_float tmp;
      zeronp_float max = 0.0;
      for (i = 0; i < len; ++i)
      {
            tmp = x[i];
            if (tmp > max)
            {
                  max = tmp;
            }
      }
      return ABS(max);
}

/* max(|v|) */
zeronp_float ZERONP(norm_inf)(
    const zeronp_float *a,
    zeronp_int len)
{
      if (a == ZERONP_NULL || len <= 0)
      {
            return 0;
      }

      zeronp_int i;
      zeronp_float tmp;
      zeronp_float max = 0.0;
      for (i = 0; i < len; ++i)
      {
            tmp = ABS(a[i]);
            if (tmp > max)
            {
                  max = tmp;
            }
      }
      return max;
}

/* a .+= b */
void ZERONP(add_array)(
    zeronp_float *a,
    const zeronp_float b,
    zeronp_int len)
{
      if (a == ZERONP_NULL || len <= 0)
      {
            return;
      }

      zeronp_int i;
      for (i = 0; i < len; ++i)
      {
            a[i] += b;
      }
}

/* saxpy a += sc*b */
void ZERONP(add_scaled_array)(
    zeronp_float *a,
    const zeronp_float *b,
    zeronp_int len,
    const zeronp_float sc)
{
      if (a == ZERONP_NULL || b == ZERONP_NULL || len <= 0)
      {
            return;
      }

      zeronp_int i;
      for (i = 0; i < len; ++i)
      {
            a[i] += sc * b[i];
      }
}

/* ||a-b||_2^2 */
zeronp_float ZERONP(norm_diff)(
    const zeronp_float *a,
    const zeronp_float *b,
    zeronp_int len)
{
      if (a == ZERONP_NULL || b == ZERONP_NULL || len <= 0)
      {
            return 0;
      }

      zeronp_int i;
      zeronp_float tmp;
      zeronp_float nm_diff = 0.0;
      for (i = 0; i < len; ++i)
      {
            tmp = (a[i] - b[i]);
            nm_diff += tmp * tmp;
      }
      return SQRTF(nm_diff);
}

/* max(|a-b|) */
zeronp_float ZERONP(norm_inf_diff)(
    const zeronp_float *a,
    const zeronp_float *b,
    zeronp_int len)
{
      if (a == ZERONP_NULL || b == ZERONP_NULL || len <= 0)
      {
            return 0;
      }

      zeronp_int i;
      zeronp_float tmp;
      zeronp_float max = 0.0;
      for (i = 0; i < len; ++i)
      {
            tmp = ABS(a[i] - b[i]);
            if (tmp > max)
            {
                  max = tmp;
            }
      }
      return max;
}

/* Ax where A \in R^m*n and x \in R^n */
/* ax = Ax */
void ZERONP(Ax)(
    zeronp_float *ax,
    const zeronp_float *A,
    const zeronp_float *x,
    zeronp_int m,
    zeronp_int n)
{
      zeronp_int i;
      ZERONP(set_as_scaled_array)
      (ax, A, x[0], m);
      for (i = 1; i < n; i++)
      {
            ZERONP(add_scaled_array)
            (ax, &(A[i * m]), m, x[i]);
      }

      return;
}

/* Rank 1 update of matrix :h  =  h + alpha * x x^T*/
void ZERONP(rank1update)(
    zeronp_int n,
    zeronp_float *h,
    zeronp_float alpha,
    zeronp_float *x)
{
      zeronp_int i, j;

      for (i = 0; i < n; i++)
      { // col index
            for (j = 0; j < n; j++)
            { // row index
                  // element h(j,i)
                  h[i * n + j] += alpha * x[i] * x[j];
            }
      }
}
// /* Cholesky Decomposition: chol(h+ mu* diag(dx))*/
// void ZERONP(chol)
// (
//     zeronp_int n,
//     ZERONPMatrix* h,
//     zeronp_float* dx,
//     zeronp_float** result,
//     zeronp_float mu
// )
// {

// }

zeronp_float ZERONP(max)(zeronp_float *a, zeronp_int len)
{
      zeronp_int i;
      zeronp_float m = -INFINITY;
      for (i = 0; i < len; i++)
      {
            if (m < a[i])
            {
                  m = a[i];
            }
      }
      return m;
}
zeronp_float ZERONP(min)(zeronp_float *a, zeronp_int len)
{
      zeronp_int i;
      zeronp_float m = INFINITY;
      for (i = 0; i < len; i++)
      {
            if (m > a[i])
            {
                  m = a[i];
            }
      }
      return m;
}

// col major A in R^m*n, AT in R^n*m
void ZERONP(transpose)(
    const zeronp_int m,
    const zeronp_int n,
    const zeronp_float *A,
    zeronp_float *AT)
{
      zeronp_int i;
      zeronp_int j;

      for (i = 0; i < n; i++)
      {
            for (j = 0; j < m; j++)
            {
                  AT[i + j * n] = A[j + m * i];
            }
      }
}

/* C = AB where A \in R^m*n and B \in R^n*p */
void ZERONP(AB)(
    zeronp_float *c,
    const zeronp_float *a,
    const zeronp_float *b,
    zeronp_int m,
    zeronp_int n,
    zeronp_int p)
{
      zeronp_int i;
      for (i = 0; i < p; i++)
      {
            ZERONP(Ax)
            (&c[i * m], a, &b[i * n], m, n);
      }
}

zeronp_int countA_sys(
    zeronp_int m,
    zeronp_int n,
    zeronp_float *A)
{
      zeronp_int i, j, count = 0;
      for (j = 0; j < n; j++)
      {
            for (i = 0; i < j + 1; i++)
            {
                  if (A[i + j * m] != 0.)
                  {
                        count = count + 1;
                  }
            }
      }
      return count;
}
zeronp_int countA(
    zeronp_int m,
    zeronp_int n,
    zeronp_float *A)
{
      zeronp_int i, j, count = 0;
      for (j = 0; j < n; j++)
      {
            for (i = 0; i < m; i++)
            {
                  if (A[i + j * m] != 0.)
                  {
                        count = count + 1;
                  }
            }
            count = count + 1;
      }
      return count;
}

void calculate_csc_sys(
    c_int m,
    c_int n,
    c_float *A,
    c_float *A_x,
    c_int *A_i,
    c_int *A_p)
{
      c_int i, j, count = 0;
      A_p[0] = count;
      for (j = 0; j < n; j++)
      {
            for (i = 0; i < j + 1; i++)
            {
                  if (A[i + j * m] != 0)
                  {
                        A_x[count] = A[i + j * m];
                        A_i[count] = i;
                        count = count + 1;
                  }
            }
            A_p[j + 1] = count;
      }
}
void calculate_csc(
    c_int m,
    c_int n,
    c_float *A,
    c_float *A_x,
    c_int *A_i,
    c_int *A_p)
{
      c_int i, j, count = 0;
      A_p[0] = count;
      for (j = 0; j < n; j++)
      {
            for (i = 0; i < m; i++)
            {
                  if (A[i + j * m] != 0)
                  {
                        A_x[count] = A[i + j * m];
                        A_i[count] = i;
                        count = count + 1;
                  }
            }
            A_x[count] = 1;
            A_i[count] = j + m;
            count = count + 1;
            A_p[j + 1] = count;
      }
}

void max_kelement(
    zeronp_float *array,
    zeronp_int len,
    zeronp_int k,
    zeronp_int *output)
{
      zeronp_int i;
      // calculate the index of max k element in array,
      if (k == 0)
      {
            return;
      }

      zeronp_float *a = (zeronp_float *)zeronp_malloc(len * sizeof(zeronp_float));
      zeronp_int *index_a = (zeronp_int *)zeronp_malloc(len * sizeof(zeronp_int));
      memcpy(a, array, len * sizeof(zeronp_float));

      for (i = 0; i < len; i++)
      {
            index_a[i] = i;
      }

      zeronp_int pos = 0;

      // bubble sort

      while (pos < k)
      {
            zeronp_int m = a[pos];
            zeronp_int index = pos;
            for (i = pos + 1; i < len; i++)
            {
                  if (m < a[i])
                  {
                        m = a[i];
                        index = i;
                  }
            } // Exchange pos and index
            if (pos != index)
            {
                  zeronp_int temp = index_a[index];
                  a[index] = a[pos];
                  index_a[index] = index_a[pos];
                  a[pos] = m;
                  index_a[pos] = temp;
            }
            pos++;
      }
      memcpy(output, index_a, k * sizeof(zeronp_int));
      zeronp_free(a);
      zeronp_free(index_a);
}

zeronp_float Uniform_dis(zeronp_float range)
{
      // Generate uniform distribution in (0,range)
      zeronp_float x;
      x = ((zeronp_float)rand()) / (RAND_MAX);
      x = x - ((zeronp_float)rand() / RAND_MAX) / (RAND_MAX);
      x = MAX(1e-16, x);
      if (x > 1)
      {
            x = 1e-16;
      }
      return range * x;
}

void Gaussian(
    zeronp_float mean,
    zeronp_float stddev,
    zeronp_int length,
    zeronp_float *x)
{
      zeronp_int i;
      zeronp_float u1, u2;
      i = 0;
      while (i < length)
      {
            u1 = Uniform_dis(1);
            u2 = Uniform_dis(1);
            x[i] = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
            x[i] = mean + stddev * x[i];
            /*  if (isnan(x[i])) {
                  x[i] = 1;
              }*/
            // if (i + 1 < length) {
            //     x[i+1] = sqrt(-2.0 * log(u1)) * sin(2.0 * PI * u2);
            //     x[i+1] = mean + stddev * x[i+1];
            ///*     if (isnan(x[i+1])) {
            //         x[i+1] = 1;
            //     }*/
            // }
            i += 1;
      }
      return;
}

void Uniform_sphere(
    zeronp_float *x,
    zeronp_int dim,
    zeronp_float radius)
{
      Gaussian(0, 1, dim, x);
      zeronp_float x_norm = ZERONP(norm)(x, dim);

      ZERONP(set_as_scaled_array)
      (x, x, radius / x_norm, dim);
      return;
}
/*

/*
 * Get number of nonzero elements in a dense matrix
 */
/*
static zeronp_int nonzero_elements(const zeronp_float * in_matrix, const zeronp_int nrows, const zeronp_int ncols) {
   int i_row, i_col;
   zeronp_int num_nonzero = 0;
   for (i_row = 0; i_row < nrows; i_row++) {
         for (i_col = 0; i_col < ncols; i_col++) {
               if (in_matrix[i_col + i_row * ncols] != 0.0) {
                     num_nonzero++;
               }
         }
   }
   return num_nonzero;
}

static zeronp_int arr_ind(const zeronp_int i_col, const zeronp_int i_row, const zeronp_int nrows, const zeronp_int ncols, const zeronp_int format) {
   return (format == RowMajor) ? (i_col + i_row * ncols) : (i_row + i_col * nrows);
}

void *dense_to_csc_matrix(zeronp_float * in_matrix,
                     const zeronp_int nrows,
                     const zeronp_int ncols,
                     const zeronp_int format,
                     zeronp_int **p,
                     zeronp_int **i,
                     zeronp_float **x)
   {
   zeronp_int i_row, i_col, ind_mat, ind_val = 0, num_col_nnz = 0;
   zeronp_float *values;
   zeronp_int *rows, *col_nnz;
   const zeronp_int nnz = nonzero_elements(in_matrix, nrows, ncols);
   values = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * nnz);
   rows = (zeronp_int *)zeronp_malloc(sizeof(zeronp_int) * nnz);
   col_nnz = (zeronp_int *)zeronp_malloc(sizeof(zeronp_int) * (ncols + 1));

   // Fill values
   col_nnz[0] = (zeronp_int)0;
   for (i_col = 0; i_col < ncols; i_col++) {
         num_col_nnz = 0;
         for (i_row = 0; i_row < nrows; i_row++) {
               ind_mat = arr_ind(i_col, i_row, nrows, ncols, format);
               if (in_matrix[ind_mat] != 0.0) {
                     values[ind_val] = in_matrix[ind_mat];
                     rows[ind_val] = i_row;
                     ind_val++;
                     num_col_nnz++;
               }
         }
         col_nnz[i_col + 1] = col_nnz[i_col] + num_col_nnz;
   }

   // Create CSR structure
   *p = col_nnz;
   *i = rows;
   *x = values;
}*/