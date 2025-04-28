#pragma once
#ifndef LINALG_H_GUARD
#define LINALG_H_GUARD
#define PI 3.14159265358979323846
#ifdef __cplusplus
extern "C"
{
#endif

#include "osqp.h"
#include "zeronp.h"
#include <math.h>
#include <stdlib.h>
	// #include "osqp.h"

	zeronp_float ZERONP(vec_mean)(
		zeronp_float *x,
		zeronp_int len);

	void ZERONP(set_as_scaled_array)(
		zeronp_float *x,
		const zeronp_float *a,
		const zeronp_float b,
		zeronp_int len);

	void ZERONP(set_as_sqrt)(
		zeronp_float *x,
		const zeronp_float *v,
		zeronp_int len);

	void ZERONP(set_as_sq)(
		zeronp_float *x,
		const zeronp_float *v,
		zeronp_int len);

	void ZERONP(scale_array)(
		zeronp_float *a,
		const zeronp_float b,
		zeronp_int len);

	zeronp_float ZERONP(dot)(
		const zeronp_float *x,
		const zeronp_float *y,
		zeronp_int len);

	zeronp_float ZERONP(norm_sq)(
		const zeronp_float *v,
		zeronp_int len);

	zeronp_float ZERONP(norm_1)(
		const zeronp_float *x,
		const zeronp_int len);

	zeronp_float ZERONP(cone_norm_1)(
		const zeronp_float *x,
		const zeronp_int len);

	zeronp_float ZERONP(norm)(
		const zeronp_float *v,
		zeronp_int len);

	zeronp_float ZERONP(norm_inf)(
		const zeronp_float *a,
		zeronp_int len);

	void ZERONP(add_array)(
		zeronp_float *a,
		const zeronp_float b,
		zeronp_int len);

	void ZERONP(add_scaled_array)(
		zeronp_float *a,
		const zeronp_float *b,
		zeronp_int n,
		const zeronp_float sc);

	zeronp_float ZERONP(norm_diff)(
		const zeronp_float *a,
		const zeronp_float *b,
		zeronp_int len);

	zeronp_float ZERONP(norm_inf_diff)(
		const zeronp_float *a,
		const zeronp_float *b,
		zeronp_int len);

	void ZERONP(Ax)(
		zeronp_float *ax,
		const zeronp_float *a,
		const zeronp_float *x,
		zeronp_int m,
		zeronp_int n);

	/* Rank 1 update of matrix :h  =  h + alpha * x x^T*/
	void ZERONP(rank1update)(
		zeronp_int n,
		zeronp_float *h,
		zeronp_float alpha,
		zeronp_float *x);

	void ZERONP(transpose)(
		const zeronp_int m,
		const zeronp_int n,
		const zeronp_float *A,
		zeronp_float *AT);

	void ZERONP(AB)(
		zeronp_float *c,
		const zeronp_float *a,
		const zeronp_float *b,
		zeronp_int m,
		zeronp_int n,
		zeronp_int p);

	zeronp_float ZERONP(min)(zeronp_float *a, zeronp_int len);
	zeronp_float ZERONP(max)(zeronp_float *a, zeronp_int len);

	zeronp_int countA_sys(
		zeronp_int m,
		zeronp_int n,
		zeronp_float *A);
	zeronp_int countA(
		zeronp_int m,
		zeronp_int n,
		zeronp_float *A);
	void calculate_csc_sys(
		c_int m,
		c_int n,
		c_float *A,
		c_float *A_x,
		c_int *A_i,
		c_int *A_p);
	void calculate_csc(
		c_int m,
		c_int n,
		c_float *A,
		c_float *A_x,
		c_int *A_i,
		c_int *A_p);
	void max_kelement(
		zeronp_float *array,
		zeronp_int len,
		zeronp_int k,
		zeronp_int *output);
	zeronp_float Uniform_dis(zeronp_float range);
	void Gaussian(
		zeronp_float mean,
		zeronp_float stddev,
		zeronp_int length,
		zeronp_float *x);
	void Uniform_sphere(
		zeronp_float *x,
		zeronp_int dim,
		zeronp_float radius);
#ifdef __cplusplus
}
#endif
#endif
