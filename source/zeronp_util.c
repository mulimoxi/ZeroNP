#define _CRT_SECURE_NO_WARNINGS
#include "zeronp_glbopts.h"
#include "zeronp_util.h"

#if (defined NOTIMER)

void ZERONP(tic)(
    ZERONP(timer) * t) {}

zeronp_float ZERONP(tocq)(
    ZERONP(timer) * t)
{
      return NAN;
}

#elif (defined _WIN32 || _WIN64 || defined _WINDLL)

void ZERONP(tic)(
    ZERONP(timer) * t)
{
      QueryPerformanceFrequency(&t->freq);
      QueryPerformanceCounter(&t->tic);
}

zeronp_float ZERONP(tocq)(
    ZERONP(timer) * t)
{
      QueryPerformanceCounter(&t->toc);
      return (1e3 * (t->toc.QuadPart - t->tic.QuadPart) / (zeronp_float)t->freq.QuadPart);
}

#elif (defined __APPLE__)

void ZERONP(tic)(
    ZERONP(timer) * t)
{
      /* read current clock cycles */
      t->tic = mach_absolute_time();
}

zeronp_float ZERONP(tocq)(
    ZERONP(timer) * t)
{
      uint64_t duration;

      t->toc = mach_absolute_time();
      duration = t->toc - t->tic;

      mach_timebase_info(&(t->tinfo));
      duration *= t->tinfo.numer;
      duration /= t->tinfo.denom;

      return (zeronp_float)duration / 1e6;
}

#else

void ZERONP(tic)(
    ZERONP(timer) * t)
{
      clock_gettime(CLOCK_MONOTONIC, &t->tic);
}

zeronp_float ZERONP(tocq)(
    ZERONP(timer) * t)
{
      struct timespec temp;

      clock_gettime(CLOCK_MONOTONIC, &t->toc);

      if ((t->toc.tv_nsec - t->tic.tv_nsec) < 0)
      {
            temp.tv_sec = t->toc.tv_sec - t->tic.tv_sec - 1;
            temp.tv_nsec = 1e9 + t->toc.tv_nsec - t->tic.tv_nsec;
      }
      else
      {
            temp.tv_sec = t->toc.tv_sec - t->tic.tv_sec;
            temp.tv_nsec = t->toc.tv_nsec - t->tic.tv_nsec;
      }

      return (zeronp_float)temp.tv_sec * 1e3 + (zeronp_float)temp.tv_nsec / 1e6;
}

#endif

zeronp_float ZERONP(toc)(
    ZERONP(timer) * t)
{
      zeronp_float time = ZERONP(tocq)(t);
      zeronp_printf("time: %8.4f milli-seconds.\n", time);
      return time;
}

zeronp_float ZERONP(str_toc)(
    char *str,
    ZERONP(timer) * t)
{
      zeronp_float time = ZERONP(tocq)(t);
      zeronp_printf("%s - time: %8.4f milli-seconds.\n", str, time);
      return time;
}

ZERONPConstraint *malloc_constriant(zeronp_int n, zeronp_int nic)
{
      ZERONPConstraint *constraint = (ZERONPConstraint *)zeronp_calloc(1, sizeof(ZERONPConstraint));

      constraint->n = n;
      constraint->nic = nic;

      if (nic > 0)
      {
            constraint->il = (zeronp_float *)zeronp_malloc(nic * sizeof(zeronp_float));
            constraint->iu = (zeronp_float *)zeronp_malloc(nic * sizeof(zeronp_float));
      }
      else
      {
            constraint->il = ZERONP_NULL;
            constraint->iu = ZERONP_NULL;
      }

      if (n > 0)
      {
            constraint->pl = (zeronp_float *)zeronp_malloc(n * sizeof(zeronp_float));
            constraint->pu = (zeronp_float *)zeronp_malloc(n * sizeof(zeronp_float));
      }
      else
      {
            constraint->pl = ZERONP_NULL;
            constraint->pu = ZERONP_NULL;
      }

      constraint->Ipc = (zeronp_int *)zeronp_malloc(2 * sizeof(zeronp_int));
      constraint->Ipb = (zeronp_int *)zeronp_malloc(2 * sizeof(zeronp_int));

      return constraint;
}

ZERONPCost *malloc_cost(zeronp_int nec, zeronp_int nic, void *func, void *grad, void *hess)
{
      ZERONPCost *c = (ZERONPCost *)zeronp_calloc(1, sizeof(ZERONPCost));

      c->nec = nec;
      c->nic = nic;

      if (nec > 0)
      {
            c->ec = (zeronp_float *)zeronp_malloc(nec * sizeof(zeronp_float));
      }
      else
      {
            c->ec = ZERONP_NULL;
      }

      if (nic > 0)
      {
            c->ic = (zeronp_float *)zeronp_malloc(nic * sizeof(zeronp_float));
      }
      else
      {
            c->ic = ZERONP_NULL;
      }

      c->cost = func;
      c->grad = grad;
      c->hess = hess;
      return c;
}

void ZERONP(set_default_settings)(ZERONPSettings *stgs, zeronp_int np)
{
      stgs->rho = 1;
      stgs->pen_l1 = 1;
      stgs->max_iter = 50;
      stgs->min_iter = 10;
      stgs->max_iter_rescue = 50;
      stgs->min_iter_rescue = 10;
      stgs->delta = 1.;
      stgs->tol = 1e-4;
      stgs->tol_con = 1e-3;
      stgs->ls_time = 10;
      stgs->batchsize = MAX(MIN(50, np / 4), 1);
      stgs->tol_restart = 1.;
      stgs->re_time = 5;
      stgs->delta_end = 1e-5;
      stgs->maxfev = 500 * np;
      stgs->noise = 1;
      stgs->qpsolver = 1;
      stgs->scale = 1;
      stgs->bfgs = 1;
      stgs->rs = 0;
      stgs->grad = 1;
      stgs->k_i = 3.;
      stgs->k_r = 9;
      stgs->c_r = 10.;
      stgs->c_i = 30.;
      stgs->ls_way = 1;
      stgs->rescue = 0;
      stgs->drsom = 0;
      stgs->cen_diff = 0;
      stgs->gd_step = 1e-1;
      stgs->step_ratio = 1. / 3;
      stgs->h = 1e-3;
      stgs->verbose = 1;
}

void ZERONP(free_sol)(ZERONPSol *sol)
{
      if (sol)
      {
            if (sol->p)
            {
                  zeronp_free(sol->p);
            }
            if (sol->ic)
            {
                  zeronp_free(sol->ic);
            }
            if (sol->jh)
            {
                  zeronp_free(sol->jh);
            }
            if (sol->ch)
            {
                  zeronp_free(sol->ch);
            }
            if (sol->l)
            {
                  zeronp_free(sol->l);
            }
            if (sol->h)
            {
                  zeronp_free(sol->h);
            }
            if (sol->count_h)
            {
                  zeronp_free(sol->count_h);
            }
            if (sol->best_fea_p)
            {
                  zeronp_free(sol->best_fea_p);
            }

            zeronp_free(sol);
      }
}

void ZERONP(free_info)(ZERONPInfo *info)
{
      if (info)
      {
            zeronp_free(info);
      }
}

void ZERONP(free_stgs)(ZERONPSettings *stgs)
{
      if (stgs)
      {
            zeronp_free(stgs);
      }
}

ZERONPProb *ZERONP(init_prob)(zeronp_int np, zeronp_int nic, zeronp_int nec)
{
      ZERONPProb *prob = (ZERONPProb *)zeronp_calloc(1, sizeof(ZERONPProb));
      prob->np = np;
      prob->nic = nic;
      prob->nec = nec;
      prob->nc = nic + nec;

      prob->ib0 = ZERONP_NULL;
      prob->ibu = ZERONP_NULL;
      prob->ibl = ZERONP_NULL;
      prob->p0 = ZERONP_NULL;
      prob->pbl = ZERONP_NULL;
      prob->pbu = ZERONP_NULL;
      prob->Ipc = ZERONP_NULL;
      prob->Ipb = ZERONP_NULL;

      return prob;
}

void ZERONP(free_prob)(ZERONPProb *prob)
{
      if (prob)
      {
            if (prob->ibl)
            {
                  zeronp_free(prob->ibl);
            }
            if (prob->ibu)
            {
                  zeronp_free(prob->ibu);
            }
            if (prob->pbl)
            {
                  zeronp_free(prob->pbl);
            }
            if (prob->pbu)
            {
                  zeronp_free(prob->pbu);
            }
            if (prob->ib0)
            {
                  zeronp_free(prob->ib0);
            }
            if (prob->p0)
            {
                  zeronp_free(prob->p0);
            }
            if (prob->Ipc)
            {
                  zeronp_free(prob->Ipc);
            }
            if (prob->Ipb)
            {
                  zeronp_free(prob->Ipb);
            }

            zeronp_free(prob);
      }
}