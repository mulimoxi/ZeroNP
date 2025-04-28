#include "zeronp.h"
#include "zeronp_util.h"
#include <stdio.h>

// void p78(zeronp_float *x, zeronp_float *result, zeronp_int np, zeronp_int nfeval)
void p78(zeronp_float *x, zeronp_float *result, zeronp_int np)
{
    result[0] = 1.0;
    for (int i = 0; i < np; ++i)
    {
        result[0] *= x[i];
    }
    result[1] = -10.0;
    for (int i = 0; i < np; ++i)
    {
        result[1] += x[i] * x[i];
    }
    result[2] = x[1] * x[2] - 5 * x[3] * x[4];
    result[3] = x[0] * x[0] * x[0] + x[1] * x[1] * x[1] + 1.0;
}

int main(void)
{

    ZERONPSettings *stgs = (ZERONPSettings *)zeronp_calloc(1, sizeof(ZERONPSettings));
    ZERONPSol *sol = (ZERONPSol *)zeronp_calloc(1, sizeof(ZERONPSol));
    ZERONPInfo *info = (ZERONPInfo *)zeronp_calloc(1, sizeof(ZERONPInfo));
    zeronp_float *l = ZERONP_NULL;
    zeronp_float *h = ZERONP_NULL;
    ZERONPProb *prob = ZERONP(init_prob)(5, 0, 3);

    ZERONP(set_default_settings)
    (stgs, prob->np);

    prob->p0 = (zeronp_float *)zeronp_malloc(prob->np * sizeof(zeronp_float));
    prob->p0[0] = -2.0;
    prob->p0[1] = 1.5;
    prob->p0[2] = 2.0;
    prob->p0[3] = -1.0;
    prob->p0[4] = -1.0;

    ZERONP_PLUS(prob, stgs, sol, info, p78, ZERONP_NULL, ZERONP_NULL, l, h);

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
