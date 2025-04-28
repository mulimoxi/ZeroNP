#include "zeronp.h"
#include "zeronp_util.h"
#include <stdio.h>

// void p93(zeronp_float *x, zeronp_float *result, zeronp_int np, zeronp_int nfeval)
void p93(zeronp_float *x, zeronp_float *result, zeronp_int np)
{
    result[0] = 0.0;
    result[0] += 0.0204 * x[0] * x[3] * (x[0] + x[1] + x[2]);
    result[0] += 0.0187 * x[1] * x[2] * (x[0] + 1.57 * x[1] + x[3]);
    result[0] += 0.0607 * x[0] * x[3] * x[4] * x[4] * (x[0] + x[1] + x[2]);
    result[0] += 0.0437 * x[1] * x[2] * x[5] * x[5] * (x[0] + 1.57 * x[1] + x[3]);
    result[1] = 0.001 * x[0] * x[1] * x[2] * x[3] * x[4] * x[5] - 2.07;
    result[2] = 1.0;
    result[2] -= 0.00062 * x[0] * x[3] * x[4] * x[4] * (x[0] + x[1] + x[2]);
    result[2] -= 0.00058 * x[1] * x[2] * x[5] * x[5] * (x[0] + 1.57 * x[1] + x[3]);
}

int main(void)
{

    ZERONPSettings *stgs = (ZERONPSettings *)zeronp_calloc(1, sizeof(ZERONPSettings));
    ZERONPSol *sol = (ZERONPSol *)zeronp_calloc(1, sizeof(ZERONPSol));
    ZERONPInfo *info = (ZERONPInfo *)zeronp_calloc(1, sizeof(ZERONPInfo));
    zeronp_float *l = ZERONP_NULL;
    zeronp_float *h = ZERONP_NULL;
    ZERONPProb *prob = ZERONP(init_prob)(6, 2, 0);

    ZERONP(set_default_settings)
    (stgs, prob->np);

    prob->pbl = (zeronp_float *)zeronp_malloc(prob->np * sizeof(zeronp_float));
    prob->ibl = (zeronp_float *)zeronp_malloc(prob->nic * sizeof(zeronp_float));
    prob->ib0 = (zeronp_float *)zeronp_malloc(prob->nic * sizeof(zeronp_float));
    prob->p0 = (zeronp_float *)zeronp_malloc(prob->np * sizeof(zeronp_float));

    memset(prob->pbl, 0, prob->np * sizeof(zeronp_float));
    memset(prob->ibl, 0, prob->nic * sizeof(zeronp_float));
    prob->ib0[0] = 1.0;
    prob->ib0[1] = 1.0;
    prob->p0[0] = 5.54;
    prob->p0[1] = 4.4;
    prob->p0[2] = 12.02;
    prob->p0[3] = 11.82;
    prob->p0[4] = 0.702;
    prob->p0[5] = 0.852;

    ZERONP_PLUS(prob, stgs, sol, info, p93, ZERONP_NULL, ZERONP_NULL, l, h);

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
