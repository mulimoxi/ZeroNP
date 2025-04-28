#ifndef UTIL_H_GUARD
#define UTIL_H_GUARD

#ifdef __cplusplus
extern "C"
{
#endif

#include "zeronp.h"
#include <stdlib.h>
#include <stdio.h>

#if (defined NOTIMER)
      typedef void *ZERONP(timer);

#elif (defined _WIN32 || defined _WIN64 || defined _WINDLL)

#include <windows.h>
typedef struct ZERONP(timer)
{
      LARGE_INTEGER tic;
      LARGE_INTEGER toc;
      LARGE_INTEGER freq;
} ZERONP(timer);

#elif (defined __APPLE__)

#include <mach/mach_time.h>
typedef struct ZERONP(timer)
{
      uint64_t tic;
      uint64_t toc;
      mach_timebase_info_data_t tinfo;
} ZERONP(timer);

#else

#include <time.h>
typedef struct ZERONP(timer)
{
      struct timespec tic;
      struct timespec toc;
} ZERONP(timer);

#endif

#if EXTRA_VERBOSE > 1
      extern ZERONP(timer) global_timer;
#endif

      void ZERONP(tic)(ZERONP(timer) * t);
      zeronp_float ZERONP(toc)(ZERONP(timer) * t);
      zeronp_float ZERONP(str_toc)(char *str, ZERONP(timer) * t);
      zeronp_float ZERONP(tocq)(ZERONP(timer) * t);

      ZERONPConstraint *malloc_constriant(zeronp_int n, zeronp_int nic);
      ZERONPCost *malloc_cost(zeronp_int nec, zeronp_int nic, void *func, void *grad, void *hess);
      void ZERONP(set_default_settings)(ZERONPSettings *stgs, zeronp_int np);
      void ZERONP(free_sol)(ZERONPSol *sol);
      void ZERONP(free_info)(ZERONPInfo *info);
      void ZERONP(free_stgs)(ZERONPSettings *stgs);
      ZERONPProb *ZERONP(init_prob)(zeronp_int np, zeronp_int nic, zeronp_int nec);
      void ZERONP(free_prob)(ZERONPProb *prob);

      // C interface for zeronp
      // typedef void cost_temple(double *p, double *result, int np, int nfeval);
      typedef void cost_temple(double *p, double *result, int np);
      typedef void g_temple(double *p, double *result);
      typedef void h_temple(double *p, double *result);
      void ZERONP_PLUS(
          ZERONPProb *prob,
          ZERONPSettings *stgs,
          ZERONPSol *sol,
          ZERONPInfo *info,
          cost_temple *cost_fun,
          g_temple *grad_fun,
          h_temple *hess_fun,
          zeronp_float *l,
          zeronp_float *h);
#ifdef __cplusplus
}
#endif
#endif
