#include "zeronp_util.h"

// void cost(zeronp_float *x, zeronp_float *result, zeronp_int np, zeronp_int nfeval)
void cost(zeronp_float *x, zeronp_float *result, zeronp_int np)
{
    result[0] = pow(x[0] - 5, 2) + pow(x[1], 2) - 25;
    result[1] = -pow(x[0], 2) + x[1];
}

int main(void)
{
    ZERONPSettings *stgs = (ZERONPSettings *)zeronp_calloc(1, sizeof(ZERONPSettings));
    ZERONPSol *sol = (ZERONPSol *)zeronp_calloc(1, sizeof(ZERONPSol));
    ZERONPInfo *info = (ZERONPInfo *)zeronp_calloc(1, sizeof(ZERONPInfo));
    zeronp_float *l = ZERONP_NULL;
    zeronp_float *h = ZERONP_NULL;
    ZERONPProb *prob = ZERONP(init_prob)(2, 1, 0);

    ZERONP(set_default_settings)
    (stgs, prob->np);

    prob->p0 = (zeronp_float *)zeronp_malloc(prob->np * sizeof(zeronp_float));
    prob->p0[0] = 4.9;
    prob->p0[1] = 0.1;

    prob->ib0 = (zeronp_float *)zeronp_malloc(prob->nic * sizeof(zeronp_float));
    prob->ib0[0] = 1.0;

    prob->ibl = (zeronp_float *)zeronp_malloc(prob->nic * sizeof(zeronp_float));
    prob->ibl[0] = 0.0;

    ZERONP_PLUS(prob, stgs, sol, info, cost, ZERONP_NULL, ZERONP_NULL, l, h);

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