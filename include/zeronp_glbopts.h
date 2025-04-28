#ifndef GLB_H_GUARD
#define GLB_H_GUARD

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>

//#define DLONG

#ifndef ZERONP
#define ZERONP(x) zeronp_##x
#endif

#ifndef SUBNP
#define SUBNP(x) subnp_##x
#endif

/* ZERONP VERSION NUMBER ----------------------------------------------    */
#define ZERONP_VERSION                                                            \
  ("1.0.0") /* string literals automatically null-terminated */



#ifdef MATLAB_MEX_FILE
#include "mex.h"
#define zeronp_printf mexPrintf
#define _zeronp_free mxFree
#define _zeronp_malloc mxMalloc
#define _zeronp_calloc mxCalloc
#define _zeronp_realloc mxRealloc
#elif defined PYTHON
#include <Python.h>
#include <stdlib.h>
#define zeronp_printf(...)                                                            \
{                                                                                               \
      PyGILState_STATE gilstate = PyGILState_Ensure();       \
      PySys_WriteStdout(__VA_ARGS__);                                 \
      PyGILState_Release(gilstate);                                          \
}
#define _zeronp_free free
#define _zeronp_malloc malloc
#define _zeronp_calloc calloc
#define _zeronp_realloc realloc
#else
#include <stdio.h>
#include <stdlib.h>
#define zeronp_printf printf
#define _zeronp_free free
#define _zeronp_malloc malloc
#define _zeronp_calloc calloc
#define _zeronp_realloc realloc
#endif

#define zeronp_free(x)   \
      _zeronp_free(x);          \
      x = ZERONP_NULL 
#define zeronp_malloc(x) _zeronp_malloc(x)
#define zeronp_calloc(x, y) _zeronp_calloc(x, y)
#define zeronp_realloc(x, y) _zeronp_realloc(x, y)

// //#ifdef DLONG
// //#ifdef _WIN64
// //typedef __int64 zeronp_int; 
// //#else
// //typedef long zeronp_int;
// //#endif
// //#else
// //typedef int zeronp_int;
// //#endif
// typedef int zeronp_int;


#ifdef DLONG
/*#ifdef _WIN64
#include <stdint.h>
typedef int64_t zeronp_int;
#else
typedef long zeronp_int;
#endif
*/
typedef long long zeronp_int;
#else
typedef int zeronp_int;
#endif


#ifndef SFLOAT
typedef double zeronp_float;
#ifndef NAN
#define NAN ((zeronp_float)0x7ff8000000000000)
#endif
#ifndef INFINITY
#define INFINITY NAN
#endif
#else
typedef float zeronp_float;
#ifndef NAN
#define NAN ((float)0x7fc00000)
#endif
#ifndef INFINITY
#define INFINITY NAN
#endif
#endif

#define ZERONP_NULL 0

#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef ABS
#define ABS(x) (((x) < 0) ? -(x) : (x))
#endif

#ifndef POWF
#ifdef SFLOAT
#define POWF powf
#else
#define POWF pow
#endif
#endif

#ifndef SQRTF
#ifdef SFLOAT
#define SQRTF sqrtf
#else
#define SQRTF sqrt
#endif
#endif

#if EXTRA_VERBOSE > 1
#if (defined _WIN32 || defined _WIN64 || defined _WINDLL)
#define __func__ __FUNCTION__
#endif
#define DEBUG_FUNC zeronp_printf("IN function: %s, time: %4f ms, file: %s, line: %i\n", __func__,  ZERONP(tocq)(&global_timer), __FILE__, __LINE__);
#define RETURN 
      zeronp_printf("EXIT function: %s, time: %4f ms, file: %s, line: %i\n", __func__, ZERONP(tocq)(&global_timer), __FILE__, __LINE__);   \
      return
#else
#define DEBUG_FUNC
#define RETURN return
#endif

#define EPS_TOL (1E-18)
#define EPS (1E-8) // for condition number in subnp
#define SAFEDIV_POS(X, Y) ((Y) < EPS_TOL ? ((X) / EPS_TOL) : (X) / (Y))

#define CONVERGED_INTERVAL (1)
#define INDETERMINATE_TOL (1e-9)

#ifdef __cplusplus
}
#endif
#endif
