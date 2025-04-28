#include "subnp.h"

zeronp_float* Est_noise(
    zeronp_int nf,
    zeronp_float* fval
);

zeronp_float* Est_second_diff_sub(
    zeronp_float fval,
    zeronp_float delta,
    zeronp_float fnoise,
    zeronp_float* p,
    zeronp_float* d,
    SUBNPWork* w_sub,
    ZERONPWork* w,
    ZERONPSettings* stgs

);
zeronp_float Est_second_diff(
    zeronp_float fnoise,
    zeronp_float fval,
    zeronp_float* p,
    zeronp_float* d,
    SUBNPWork* w_sub,
    ZERONPWork* w,
    ZERONPSettings* stgs
);
zeronp_float calculate_delta(
    zeronp_int nf,
    zeronp_float fval,
    zeronp_float* p,
    SUBNPWork* w_sub,
    ZERONPWork* w,
    ZERONPSettings* stgs
);

zeronp_float calculate_infeas_scaledob
(
    ZERONPCost* ob,
    ZERONPWork* w,
    zeronp_float* x,
    zeronp_float* scale
);
zeronp_float calculate_infeas_unscaleob
(
    ZERONPCost* ob,
    ZERONPWork* w,
    zeronp_float* x,
    zeronp_float* scale
);
zeronp_float calculate_infeas_scaledob_l1
(
    ZERONPCost* ob,
    ZERONPWork* w,
    zeronp_float* x
);
zeronp_float calculate_almcrit_iq
(
    ZERONPWork* w,
    zeronp_float* scale,
    zeronp_float* p
);
void calculate_scaled_cost
(
    ZERONPCost** ob,
    zeronp_float* p,
    const zeronp_float* scale,
    ZERONPSettings* stgs,
    ZERONPWork* w,
    zeronp_int nfeval
);
zeronp_float line_search_merit(
    ZERONPCost* ob,
    ZERONPWork* w,
    SUBNPWork* w_sub,
    zeronp_float* p
);
/*
void calculate_scaled_cost_rescue
(
    ZERONPCost** ob,
    zeronp_float* p,
    zeronp_float* slack,
    const zeronp_float* scale,
    ZERONPSettings* stgs,
    ZERONPWork* w,
    zeronp_int nfeval
);*/
zeronp_int calculate_scaled_grad
(
    zeronp_float* g,
    zeronp_float* p,
    const zeronp_float* scale,
    ZERONPSettings* stgs,
    ZERONPWork* w
);
zeronp_int calculate_scaled_grad_random
(
    zeronp_float* g,
    zeronp_float* p,
    ZERONPCost* ob_p,
    const zeronp_float* scale,
    ZERONPSettings* stgs,
    ZERONPWork* w
);
zeronp_int calculate_scaled_hess
(
    zeronp_float* h,
    zeronp_float* p,
    const zeronp_float* scale,
    ZERONPSettings* stgs,
    ZERONPWork* w
);
void calculate_alm_criterion
(
    ZERONPWork* w,
    SUBNPWork* w_sub,
    zeronp_float* grad
);
zeronp_int calculate_Jacob_zero
(
    SUBNPWork* w_sub,
    ZERONPWork* w,
    ZERONPSettings* stgs
);
zeronp_int calculate_Jacob_zero_rescue
(
    SUBNPWork* w_sub,
    ZERONPWork* w,
    ZERONPSettings* stgs
);
zeronp_int calculate_ALMgradient_zero_rescue
(
    SUBNPWork* w_sub,
    ZERONPWork* w,
    ZERONPSettings* stgs,
    zeronp_float* g,
    zeronp_float j
);
zeronp_int calculate_Jacob_first
(
    SUBNPWork* w_sub,
    ZERONPWork* w,
    ZERONPSettings* stgs
);
zeronp_int calculate_Jacob_first_rescue
(
    SUBNPWork* w_sub,
    ZERONPWork* w,
    ZERONPSettings* stgs
);
zeronp_float calculate_ALM
(
    ZERONPCost* ob,
    ZERONPSettings* stgs,
    zeronp_float* p,
    const ZERONPWork* w,
    const SUBNPWork* w_sub
);
//zeronp_float calculate_ALM_rescue
//(
//    ZERONPCost* ob,
//    ZERONPSettings* stgs,
//    zeronp_float* p,
//    const ZERONPWork* w,
//    const SUBNPWork* w_sub
//);
zeronp_int calculate_ALMgradient_zero
(
    SUBNPWork* w_sub,
    ZERONPWork* w,
    ZERONPSettings* stgs,
    zeronp_float* g,
    zeronp_float j
);
zeronp_int calculate_ALMgradient_first
(
    SUBNPWork* w_sub,
    ZERONPWork* w,
    ZERONPSettings* stgs,
    zeronp_float* g
);
zeronp_int calculate_ALMgradient_first_rescue
(
    SUBNPWork* w_sub,
    ZERONPWork* w,
    ZERONPSettings* stgs,
    zeronp_float* g,
    zeronp_float j
);
zeronp_int calculate_ALM_hess
(
    SUBNPWork* w_sub,
    ZERONPWork* w,
    ZERONPSettings* stgs,
    zeronp_float* h
);
void BFGSudpate
(
    ZERONPWork* w,
    SUBNPWork* w_sub,
    ZERONPSettings* stgs,
    zeronp_float* g,
    zeronp_float* yg,
    zeronp_float* sx
);
zeronp_float fun_along_2d
(
    ZERONPWork* w,
    ZERONPSettings* stgs,
    SUBNPWork* w_sub,
    zeronp_float* d1,
    zeronp_float* d2,
    zeronp_float* x,
    zeronp_float* coeff
);
zeronp_float* interpolate2d
(
    ZERONPWork* w,
    ZERONPSettings* stgs,
    SUBNPWork* w_sub,
    zeronp_float radius,
    zeronp_float* d1,
    zeronp_float* d2,
    zeronp_float* x,
    zeronp_float val
);