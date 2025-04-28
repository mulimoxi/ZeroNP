#pragma once
#ifndef SUBNP_H_GUARD
#define SUBNP_H_GUARD
#ifdef __cplusplus
extern "C" {
#endif

#include "zeronp.h"
typedef struct SUBNP_WORK SUBNPWork;
typedef struct SUBNP_JACOB SUBNPJacob;

struct SUBNP_JACOB
{
    zeronp_int n;
    zeronp_int nec;
    zeronp_int nic;
    zeronp_int nc;
    zeronp_int npic;
    zeronp_int a_row;
    zeronp_float *g; // gradient of f(s, x)
    zeronp_float *a; // Jacob of g(s, x) = 0
    zeronp_float* anew;
};


struct SUBNP_WORK
{
    zeronp_float *alp;
    zeronp_int n_scale; // length of scale
    zeronp_float *scale;
    zeronp_float* p_cand;
    zeronp_int ch;
    zeronp_int mm;
    zeronp_int nc;
    zeronp_int npic;
    ZERONPCost* ob_cand;
    SUBNPJacob *J;
    zeronp_float *b; // rhs if linear constraints exist
    zeronp_int* constr_index;
};
zeronp_int unscale_ob
(
    ZERONPCost* ob,
    const zeronp_float* scale
);
zeronp_int rescale_ob
(
    ZERONPCost* ob,
    const zeronp_float* scale
);
zeronp_int subnp_qp(ZERONPWork* w, ZERONPSettings* stgs, ZERONPInfo* info);
ZERONPCost* init_cost(zeronp_int nec, zeronp_int nic);
void copyZERONPCost(ZERONPCost* ob1, ZERONPCost* ob2);
// zeronp_float qpsolver_time = 0;

#ifdef __cplusplus
}
#endif
#endif