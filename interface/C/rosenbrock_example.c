#include "zeronp.h"
#include "zeronp_util.h"
#include <stdio.h>

// void rosenbrock(zeronp_float *x, zeronp_float *result, zeronp_int np, zeronp_int nfeval)
void rosenbrock(zeronp_float *x, zeronp_float *result, zeronp_int np)
{
    result[0] = (1.0 - x[0]) * (1.0 - x[0]) + 100.0 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]);
}

int main(void)
{

    ZERONPSettings *stgs = (ZERONPSettings *)zeronp_calloc(1, sizeof(ZERONPSettings));
    ZERONPSol *sol = (ZERONPSol *)zeronp_calloc(1, sizeof(ZERONPSol));
    ZERONPInfo *info = (ZERONPInfo *)zeronp_calloc(1, sizeof(ZERONPInfo));
    zeronp_float *l = ZERONP_NULL;
    zeronp_float *h = ZERONP_NULL;
    ZERONPProb *prob = ZERONP(init_prob)(2, 0, 0);

    ZERONP(set_default_settings)
    (stgs, prob->np);

    prob->pbl = (zeronp_float *)zeronp_malloc(prob->np * sizeof(zeronp_float));
    prob->pbl[0] = -1.0;
    prob->pbl[1] = -1.0;

    prob->pbu = (zeronp_float *)zeronp_malloc(prob->np * sizeof(zeronp_float));
    prob->pbu[0] = 2.0;
    prob->pbu[1] = 2.0;

    prob->p0 = (zeronp_float *)zeronp_malloc(prob->np * sizeof(zeronp_float));
    memset(prob->p0, 0, prob->np * sizeof(zeronp_float));

    ZERONP_PLUS(prob, stgs, sol, info, rosenbrock, ZERONP_NULL, ZERONP_NULL, l, h);

    printf("ZERONP_PLUS done in %d iterations, %f seconds\nThe objective is %f\n", sol->iter, info->total_time, sol->obj);

    // free the memory allocated to call zeronp
    ZERONP(free_sol)
    (sol);
    ZERONP(free_info)
    (info);
    ZERONP(free_stgs)
    (stgs);
    ZERONP(free_prob)
    (prob);
    if (l)
    {
        zeronp_free(l);
    }
    if (h)
    {
        zeronp_free(h);
    }

    return 0;
}
