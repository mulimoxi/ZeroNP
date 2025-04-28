#include "zeronp.h"
#include "matrix.h"
#include "mex.h"
#include "linalg.h"
#include "zeronp_util.h"

zeronp_float cost_time = 0; // global cost timer

// #if !(DLONG > 0)
// #ifndef DLONG
// // this memory must be freed
// static zeronp_int *cast_to_zeronp_int_arr(mwIndex *arr, zeronp_int len) {
//     zeronp_int i;
//     zeronp_int *arr_out = (zeronp_int *)zeronp_malloc(sizeof(zeronp_int) * len);
//     for (i = 0; i < len; i++) {
//         arr_out[i] = (zeronp_int)arr[i];
//     }
//     return arr_out;
// }
// #endif

// #if SFLOAT > 0
// /* this memory must be freed */
// static zeronp_float *cast_to_zeronp_float_arr(double *arr, zeronp_int len) {
//     zeronp_int i;
//     zeronp_float *arr_out = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * len);
//     for (i = 0; i < len; i++) {
//         arr_out[i] = (zeronp_float)arr[i];
//     }
//     return arr_out;
// }

// static double *cast_to_double_arr(zeronp_float *arr, zeronp_int len) {
//     zeronp_int i;
//     double *arr_out = (double *)zeronp_malloc(sizeof(double) * len);
//     for (i = 0; i < len; i++) {
//         arr_out[i] = (double)arr[i];
//     }
//     return arr_out;
// }
// #endif

static void set_output_field(mxArray **pout, zeronp_float *out, zeronp_int m, zeronp_int n)
{

    *pout = mxCreateDoubleMatrix(m, n, mxREAL);
    for (int i = 0; i < m * n; i++)
    {
        mxGetPr(*pout)[i] = (double)out[i];
    }
}

void mexCallMatlabCost(ZERONPCost **c, zeronp_float *p, zeronp_int np, zeronp_int nfeval, zeronp_int action, mxArray *fun)
{
    // initiate cost function
    // action == 0 : do nothing
    // action == 1 : initiate cost
    // action == -1 : close cost handle
    static mxArray *cost_fun = NULL;
    if (action == 1)
    {
        cost_fun = fun;
        return;
    }
    if (action == -1)
    {
        cost_fun = NULL;
        return;
    }

    ZERONP(timer)
    cost_timer;
    ZERONP(tic)
    (&cost_timer);
    zeronp_int i, j;
    // mxArray *lhs, *rhs[3];
    mxArray *lhs, *rhs[2];

    nfeval = 1;

    rhs[0] = cost_fun;
    rhs[1] = mxCreateDoubleMatrix(np, nfeval, mxREAL);
    // rhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);

    for (j = 0; j < nfeval; j++)
    {
        for (i = 0; i < np; i++)
        {
            mxGetPr(rhs[1])[i + j * np] = (double)p[i + j * np];
        }
    }
    // *mxGetPr(rhs[2]) = (double)nfeval;
    lhs = ZERONP_NULL;
    // mexCallMATLAB(1, &lhs, 3, rhs, "feval");
    mexCallMATLAB(1, &lhs, 2, rhs, "feval");

    for (j = 0; j < nfeval; j++)
    {
        // double *result = mxGetPr(lhs);

        c[j]->obj = (zeronp_float)mxGetPr(lhs)[j * (1 + c[j]->nic + c[j]->nec)];

        for (i = 0; i < c[j]->nec; i++)
        {
            c[j]->ec[i] = (zeronp_float)mxGetPr(lhs)[i + 1 + j * (1 + c[j]->nic + c[j]->nec)];
        }
        for (i = 0; i < c[j]->nic; i++)
        {
            c[j]->ic[i] = (zeronp_float)mxGetPr(lhs)[i + 1 + c[j]->nec + j * (1 + c[j]->nic + c[j]->nec)];
        }
    }
    mxDestroyArray(rhs[1]);
    // mxDestroyArray(rhs[2]);

    if (lhs != ZERONP_NULL)
    {
        mxDestroyArray(lhs);
    }
    cost_time += ZERONP(tocq)(&cost_timer) / 1e3;
}

void mexCallMatlabGradient(zeronp_float *g, zeronp_float *p, zeronp_int np, zeronp_int ngeval, zeronp_int action, mxArray *fun)
{
    // initiate cost function
    // action == 0 : do nothing
    // action == 1 : initiate cost
    // action == -1 : close cost handle
    static mxArray *grad = NULL;
    if (action == 1)
    {
        grad = fun;
        return;
    }
    if (action == -1)
    {
        grad = NULL;
        return;
    }

    ZERONP(timer)
    cost_timer;
    ZERONP(tic)
    (&cost_timer);
    zeronp_int i;
    mxArray *lhs, *rhs[2];

    rhs[0] = grad;
    rhs[1] = mxCreateDoubleMatrix(np, 1, mxREAL);
    for (i = 0; i < np; i++)
    {
        mxGetPr(rhs[1])[i] = (double)p[i];
    }

    lhs = ZERONP_NULL;
    mexCallMATLAB(1, &lhs, 2, rhs, "feval");
    for (i = 0; i < np * ngeval; i++)
    {
        // double *result = mxGetPr(lhs);
        g[i] = (zeronp_float)mxGetPr(lhs)[i];
    }
    mxDestroyArray(rhs[1]);

    if (lhs != ZERONP_NULL)
    {
        mxDestroyArray(lhs);
    }
    cost_time += ZERONP(tocq)(&cost_timer) / 1e3;
}

void mexCallMatlabHessian(zeronp_float *h, zeronp_float *p, zeronp_int np, zeronp_int nheval, zeronp_int action, mxArray *fun)
{
    // initiate cost function
    // action == 0 : do nothing
    // action == 1 : initiate cost
    // action == -1 : close cost handle

    static mxArray *hess = NULL;
    if (action == 1)
    {
        hess = fun;
        return;
    }
    if (action == -1)
    {
        hess = NULL;
        return;
    }

    ZERONP(timer)
    cost_timer;
    ZERONP(tic)
    (&cost_timer);
    zeronp_int i;
    mxArray *lhs, *rhs[2];
    rhs[0] = hess;

    rhs[1] = mxCreateDoubleMatrix(np, 1, mxREAL);
    for (i = 0; i < np; i++)
    {
        mxGetPr(rhs[1])[i] = (double)p[i];
    }

    lhs = ZERONP_NULL;
    mexCallMATLAB(1, &lhs, 2, rhs, "feval");
    for (i = 0; i < np * np * nheval; i++)
    {
        // double *result = mxGetPr(lhs);
        h[i] = (zeronp_float)mxGetPr(lhs)[i];
    }
    mxDestroyArray(rhs[1]);

    if (lhs != ZERONP_NULL)
    {
        mxDestroyArray(lhs);
    }
    cost_time += ZERONP(tocq)(&cost_timer) / 1e3;
}

// mex file for matlab function: [p,jh,l,h,ic]=zeronp(cnstr,op,l,h)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    ZERONP(timer)
    total_timer;
    ZERONP(tic)
    (&total_timer);

    if (nrhs < 1)
    {
        mexErrMsgTxt("Syntax error");
    }

    const mwSize one[1] = {1};
    const int num_sol_fields = 17;
    const char *sol_fields[] = {"p", "best_fea_p", "jh", "ch", "l", "h", "ic", "iter", "count_cost", "count_grad", "count_hess", "constraint", "obj", "status", "solve_time", "restart_time", "cost_his"};

    const mxArray *cnstr;
    const mxArray *op;
    const mxArray *fun;

    const mxArray *pblmx;
    zeronp_float *pbl;

    const mxArray *pbumx;
    zeronp_float *pbu;

    const mxArray *p0mx;
    zeronp_float *p;

    const mxArray *iblmx;
    zeronp_float *ibl;

    const mxArray *ibumx;
    zeronp_float *ibu;

    const mxArray *ib0mx;
    zeronp_float *ib0;

    zeronp_int ilc = 0;
    zeronp_int iuc = 0;
    zeronp_int ic;

    mxArray *tmp;

    const size_t *dims;
    zeronp_int i;

    const mxArray *lmex = ZERONP_NULL;
    const mxArray *hmex = ZERONP_NULL;
    zeronp_float *l;
    zeronp_float *h;

    zeronp_int ns;
    zeronp_int np;
    zeronp_int nic = 0;
    zeronp_int nec = 0;
    zeronp_int nc = 0;

    zeronp_int *Ipc = (zeronp_int *)zeronp_malloc(2 * sizeof(zeronp_int));
    memset(Ipc, 0, 2 * sizeof(zeronp_int));
    zeronp_int *Ipb = (zeronp_int *)zeronp_malloc(2 * sizeof(zeronp_int));
    memset(Ipb, 0, 2 * sizeof(zeronp_int));

    char *om = (char *)zeronp_malloc(sizeof(char) * 512); // error message
    sprintf(om, "ZeroNP--> ");

    cnstr = prhs[0];

    pblmx = (mxArray *)mxGetField(cnstr, 0, "pbl");
    if (pblmx && !mxIsEmpty(pblmx))
    {
        Ipc[0] = 1;
        Ipb[0] = 1;
        ns = (zeronp_int)mxGetNumberOfDimensions(pblmx);
        dims = mxGetDimensions(pblmx);
        np = (zeronp_int)dims[0];

        if (ns > 1 && dims[0] == 1)
        {
            np = (zeronp_int)dims[1];
        }

        pbl = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * np);
        for (i = 0; i < np; i++)
        {
            pbl[i] = (zeronp_float)mxGetPr(pblmx)[i];
        }
    }

    pbumx = (mxArray *)mxGetField(cnstr, 0, "pbu");
    if (pbumx && !mxIsEmpty(pbumx))
    {
        Ipc[1] = 1;
        Ipb[0] = 1;
        ns = (zeronp_int)mxGetNumberOfDimensions(pbumx);
        dims = mxGetDimensions(pbumx);
        np = (zeronp_int)dims[0];

        if (ns > 1 && dims[0] == 1)
        {
            np = (zeronp_int)dims[1];
        }

        pbu = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * np);
        for (i = 0; i < np; i++)
        {
            pbu[i] = (zeronp_float)mxGetPr(pbumx)[i];
        }
    }

    p0mx = (mxArray *)mxGetField(cnstr, 0, "p0");
    if (p0mx && !mxIsEmpty(p0mx))
    {

        ns = (zeronp_int)mxGetNumberOfDimensions(p0mx);
        dims = mxGetDimensions(p0mx);
        np = (zeronp_int)dims[0];

        if (ns > 1 && dims[0] == 1)
        {
            np = (zeronp_int)dims[1];
        }

        p = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * np);
        for (i = 0; i < np; i++)
        {
            p[i] = (zeronp_float)mxGetPr(p0mx)[i];
        }
    }

    if (Ipc[0] && Ipc[1])
    {

        if (!(p0mx && !mxIsEmpty(p0mx)))
        {
            p = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * np);
            memcpy(p, pbl, sizeof(zeronp_float) * np);
            ZERONP(add_scaled_array)
            (p, pbu, np, 1);
            ZERONP(scale_array)
            (p, 0.5, np);
        }

        for (i = 0; i < np; i++)
        {
            if (mxIsInf(p[i]))
            {
                sprintf(om + strlen(om), "The user does not provide initiate point! \n");
                mexErrMsgTxt(om);
            }
        }
    }
    else
    {
        if (!(p0mx && !mxIsEmpty(p0mx)))
        {
            sprintf(om + strlen(om), "The user does not provide initiate point! \n");
            mexErrMsgTxt(om);
        }
        if (Ipc[0] == 0)
        {

            pbl = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * np);
            for (i = 0; i < np; i++)
            {
                pbl[i] = -INFINITY;
            }
        }
        if (Ipc[1] == 0)
        {
            pbu = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * np);
            for (i = 0; i < np; i++)
            {
                pbu[i] = INFINITY;
            }
        }
    }

    if (Ipb[0] > 0.5)
    {

        for (i = 0; i < np; i++)
        {
            if (pbu[i] <= pbl[i])
            {
                sprintf(om + strlen(om), "The lower bounds of the parameter constraints \n");
                sprintf(om + strlen(om), "          must be strictly less than the upper bounds. \n");
                mexErrMsgTxt(om);
            }
            else if (p[i] <= pbl[i] || p[i] >= pbu[i])
            {
                sprintf(om + strlen(om), "Initial parameter values must be within the bounds \n");
                mexErrMsgTxt(om);
            }
        }
    }

    iblmx = (mxArray *)mxGetField(cnstr, 0, "ibl");
    if (iblmx && !mxIsEmpty(iblmx))
    {
        ilc = 1;
        ns = (zeronp_int)mxGetNumberOfDimensions(iblmx);
        dims = mxGetDimensions(iblmx);
        nic = (zeronp_int)dims[0];

        if (ns > 1 && dims[0] == 1)
        {
            nic = (zeronp_int)dims[1];
        }

        ibl = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * nic);
        for (i = 0; i < nic; i++)
        {
            ibl[i] = (zeronp_float)mxGetPr(iblmx)[i];
        }
    }

    ibumx = (mxArray *)mxGetField(cnstr, 0, "ibu");
    if (ibumx && !mxIsEmpty(ibumx))
    {
        iuc = 1;
        ns = (zeronp_int)mxGetNumberOfDimensions(ibumx);
        dims = mxGetDimensions(ibumx);
        nic = (zeronp_int)dims[0];

        if (ns > 1 && dims[0] == 1)
        {
            nic = (zeronp_int)dims[1];
        }

        ibu = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * nic);
        for (i = 0; i < nic; i++)
        {
            ibu[i] = (zeronp_float)mxGetPr(ibumx)[i];
        }
    }

    ib0mx = (mxArray *)mxGetField(cnstr, 0, "ib0");
    if (ib0mx && !mxIsEmpty(ib0mx))
    {
        ns = (zeronp_int)mxGetNumberOfDimensions(ib0mx);
        dims = mxGetDimensions(ib0mx);
        nic = (zeronp_int)dims[0];

        if (ns > 1 && dims[0] == 1)
        {
            nic = (zeronp_int)dims[1];
        }

        ib0 = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * nic);
        for (i = 0; i < nic; i++)
        {
            ib0[i] = (zeronp_float)mxGetPr(ib0mx)[i];
        }
    }

    ic = ilc || iuc;

    if (ilc && iuc)
    {

        if (!(ib0mx && !mxIsEmpty(ib0mx)))
        {

            ib0 = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * nic);
            memcpy(ib0, ibl, sizeof(zeronp_float) * nic);
            ZERONP(add_scaled_array)
            (ib0, ibu, nic, 1);
            ZERONP(scale_array)
            (ib0, 0.5, nic);

            for (i = 0; i < nic; i++)
            {
                if (mxIsInf(ib0[i]))
                {
                    sprintf(om + strlen(om), "The user does not provided initiate value of inequality constrains! \n");
                    mexErrMsgTxt(om);
                }
            }
        }

        for (i = 0; i < nic; i++)
        {
            if (ibu[i] <= ibl[i])
            {
                sprintf(om + strlen(om), "The lower bounds of the inequality constraints \n");
                sprintf(om + strlen(om), "          must be strictly less than the upper bounds. \n");
                mexErrMsgTxt(om);
            }
        }
    }

    else if (ic == 1)
    {

        if (!(ib0mx && !mxIsEmpty(ib0mx)))
        {
            sprintf(om + strlen(om), "The user does not provided initiate value of inequality constrains! \n");
            mexErrMsgTxt(om);
        }

        if (ilc == 0 && nic > 0)
        {
            ibl = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * nic);
            for (i = 0; i < nic; i++)
            {
                ibl[i] = -INFINITY;
            }
        }

        if (iuc == 0 && nic > 0)
        {
            ibu = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * nic);
            for (i = 0; i < nic; i++)
            {
                ibu[i] = INFINITY;
            }
        }
    }

    if (ic)
    {
        for (i = 0; i < nic; i++)
        {
            if (ib0[i] <= ibl[i] || ib0[i] >= ibu[i])
            {
                sprintf(om + strlen(om), "Initial inequalities must be within the bounds \n");
                mexErrMsgTxt(om);
            }
        }
    }

    if (Ipb[0] + nic >= 0.5)
    {
        Ipb[1] = 1;
    }

    ZERONPConstraint *constraint = malloc_constriant(np, nic);
    if (nic > 0)
    {
        memcpy(constraint->il, ibl, nic * sizeof(zeronp_float));
        memcpy(constraint->iu, ibu, nic * sizeof(zeronp_float));
    }

    if (np > 0)
    {
        memcpy(constraint->pl, pbl, np * sizeof(zeronp_float));
        memcpy(constraint->pu, pbu, np * sizeof(zeronp_float));
    }

    memcpy(constraint->Ipc, Ipc, 2 * sizeof(zeronp_int));
    memcpy(constraint->Ipb, Ipb, 2 * sizeof(zeronp_int));

    zeronp_free(pbl);
    zeronp_free(pbu);
    if (nic > 0)
    {
        zeronp_free(ibl);
        zeronp_free(ibu);
    }
    zeronp_free(Ipc);
    zeronp_free(Ipb);

    ZERONPSettings *stgs = (ZERONPSettings *)zeronp_malloc(sizeof(ZERONPSettings));
    ZERONP(set_default_settings)
    (stgs, np);
    stgs->maxfev = 500 * np;
    op = prhs[1];

    tmp = mxGetField(op, 0, "rho");
    if (tmp != ZERONP_NULL)
    {
        stgs->rho = (zeronp_float)*mxGetPr(tmp);
    }

    tmp = mxGetField(op, 0, "rescue");
    if (tmp != ZERONP_NULL)
    {
        stgs->rescue = (zeronp_int)*mxGetPr(tmp);
    }
    else
    {
        if (nec >= np)
        {
            stgs->rescue = 1;
        }
    }

    tmp = mxGetField(op, 0, "pen_l1");
    if (tmp != ZERONP_NULL)
    {
        stgs->pen_l1 = (zeronp_float)*mxGetPr(tmp);
    }

    tmp = mxGetField(op, 0, "max_iter");
    if (tmp != ZERONP_NULL)
    {
        stgs->max_iter = (zeronp_int)*mxGetPr(tmp);
    }

    tmp = mxGetField(op, 0, "min_iter");
    if (tmp != ZERONP_NULL)
    {
        stgs->min_iter = (zeronp_int)*mxGetPr(tmp);
    }

    tmp = mxGetField(op, 0, "max_iter_rescue");
    if (tmp != ZERONP_NULL)
    {
        stgs->max_iter_rescue = (zeronp_int)*mxGetPr(tmp);
    }
    else
    {
        stgs->max_iter_rescue = stgs->max_iter;
    }

    tmp = mxGetField(op, 0, "min_iter_rescue");
    if (tmp != ZERONP_NULL)
    {
        stgs->min_iter_rescue = (zeronp_int)*mxGetPr(tmp);
    }
    else
    {
        stgs->min_iter_rescue = stgs->min_iter;
    }

    tmp = mxGetField(op, 0, "delta");
    if (tmp != ZERONP_NULL)
    {
        stgs->delta = (zeronp_float)*mxGetPr(tmp);
    }

    tmp = mxGetField(op, 0, "step_ratio");
    if (tmp != ZERONP_NULL)
    {
        stgs->step_ratio = (zeronp_float)*mxGetPr(tmp);
    }

    tmp = mxGetField(op, 0, "max_fev");
    if (tmp != ZERONP_NULL)
    {
        stgs->maxfev = (zeronp_float)*mxGetPr(tmp);
    }
    tmp = mxGetField(op, 0, "tol");
    if (tmp != ZERONP_NULL)
    {
        stgs->tol = (zeronp_float)*mxGetPr(tmp);
    }

    tmp = mxGetField(op, 0, "gd_step");
    if (tmp != ZERONP_NULL)
    {
        stgs->gd_step = (zeronp_float)*mxGetPr(tmp);
    }

    tmp = mxGetField(op, 0, "tol_con");
    if (tmp != ZERONP_NULL)
    {
        stgs->tol_con = (zeronp_float)*mxGetPr(tmp);
    }

    tmp = mxGetField(op, 0, "ls_time");
    if (tmp != ZERONP_NULL)
    {
        stgs->ls_time = (zeronp_int)*mxGetPr(tmp);
    }

    tmp = mxGetField(op, 0, "batchsize");
    if (tmp != ZERONP_NULL)
    {
        stgs->batchsize = (zeronp_int)*mxGetPr(tmp);
    }
    else
    {
        stgs->batchsize = MAX(MIN(50, np / 4), 1);
    }

    tmp = mxGetField(op, 0, "tol_restart");
    if (tmp != ZERONP_NULL)
    {
        stgs->tol_restart = (zeronp_float)*mxGetPr(tmp);
    }

    tmp = mxGetField(op, 0, "re_time");
    if (tmp != ZERONP_NULL)
    {
        stgs->re_time = (zeronp_int)*mxGetPr(tmp);
    }

    tmp = mxGetField(op, 0, "delta_end");
    if (tmp != ZERONP_NULL)
    {
        stgs->delta_end = (zeronp_float)*mxGetPr(tmp);
    }

    tmp = mxGetField(op, 0, "maxfev");
    if (tmp != ZERONP_NULL)
    {
        stgs->maxfev = (zeronp_int)*mxGetPr(tmp);
    }
    tmp = mxGetField(op, 0, "noise");
    if (tmp != ZERONP_NULL)
    {
        stgs->noise = (zeronp_int)*mxGetPr(tmp);
    }
    if ((!stgs->noise) && mxGetField(op, 0, "delta") == ZERONP_NULL)
    {
        stgs->delta = 1e-5;
    }
    tmp = mxGetField(op, 0, "qpsolver");
    if (tmp != ZERONP_NULL)
    {
        stgs->qpsolver = (zeronp_int)*mxGetPr(tmp);
    }

    tmp = mxGetField(op, 0, "scale");
    if (tmp != ZERONP_NULL)
    {
        stgs->scale = (zeronp_int)*mxGetPr(tmp);
    }

    tmp = mxGetField(op, 0, "bfgs");
    if (tmp != ZERONP_NULL)
    {
        stgs->bfgs = (zeronp_int)*mxGetPr(tmp);
    }

    tmp = mxGetField(op, 0, "rs");
    if (tmp != ZERONP_NULL)
    {
        stgs->rs = (zeronp_int)*mxGetPr(tmp);
        stgs->grad = 1;
    }

    tmp = mxGetField(op, 0, "drsom");
    if (tmp != ZERONP_NULL)
    {
        stgs->drsom = (zeronp_int)*mxGetPr(tmp);
    }

    tmp = mxGetField(op, 0, "cen_diff");
    if (tmp != ZERONP_NULL)
    {
        stgs->cen_diff = (zeronp_int)*mxGetPr(tmp);
    }

    tmp = mxGetField(op, 0, "k_i");
    if (tmp != ZERONP_NULL)
    {
        stgs->k_i = (zeronp_float)*mxGetPr(tmp);
    }
    tmp = mxGetField(op, 0, "k_r");
    if (tmp != ZERONP_NULL)
    {
        stgs->k_r = (zeronp_float)*mxGetPr(tmp);
    }
    tmp = mxGetField(op, 0, "c_r");
    if (tmp != ZERONP_NULL)
    {
        stgs->c_r = (zeronp_float)*mxGetPr(tmp);
    }
    tmp = mxGetField(op, 0, "c_i");
    if (tmp != ZERONP_NULL)
    {
        stgs->c_i = (zeronp_float)*mxGetPr(tmp);
    }
    tmp = mxGetField(op, 0, "ls_way");
    if (tmp != ZERONP_NULL)
    {
        stgs->ls_way = (zeronp_int)*mxGetPr(tmp);
    }

    if (stgs->rescue)
    {
        stgs->min_iter = 1;
    }

    fun = prhs[2];
    tmp = mxGetField(fun, 0, "cost");
    if (tmp != ZERONP_NULL)
    {
        mexCallMatlabCost(ZERONP_NULL, ZERONP_NULL, 0, 0, 1, tmp);
    }
    else
    {
        sprintf(om + strlen(om), "The user does not provided cost function in the fun structure \n");
        mexErrMsgTxt(om);
    }

    if (stgs->noise == 3 && stgs->step_ratio <= 0)
    {
        sprintf(om + strlen(om), "The step_ratio must be a positive number! \n");
        mexErrMsgTxt(om);
    }

    // first call of cost function to get nec
    mxArray *lhs, *rhs[2];
    zeronp_int m;
    zeronp_float *ob;

    rhs[0] = tmp;
    rhs[1] = mxCreateDoubleMatrix(np, 1, mxREAL);
    // rhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);

    for (i = 0; i < np; i++)
    {
        mxGetPr(rhs[1])[i] = p[i];
    }
    // *mxGetPr(rhs[2]) = 1;

    lhs = ZERONP_NULL;

    ZERONP(timer)
    cost_timer;
    ZERONP(tic)
    (&cost_timer);
    mexCallMATLAB(1, &lhs, 2, rhs, "feval");
    cost_time += ZERONP(tocq)(&cost_timer) / 1e3;

    m = mxGetM(lhs);
    if (m == 1)
    {
        m = mxGetN(lhs);
    }

    tmp = mxGetField(fun, 0, "grad");
    if (tmp != ZERONP_NULL)
    {
        mexCallMatlabGradient(ZERONP_NULL, ZERONP_NULL, 0, 0, 1, tmp);
        stgs->grad = 1;
    }
    else
    {
        zeronp_printf("ZeroNP--> The user does not provided gradient function of cost in the fun structure. \n");
        zeronp_printf("ZeroNP--> ZERONP uses zero-order method instead.\n");
        if (stgs->rs)
        {
            zeronp_printf("ZeroNP--> ZERONP uses random sampling method to estimate the gradient.\n");
        }
        else
        {
            stgs->grad = 0;
        }
    }

    tmp = mxGetField(fun, 0, "hess");
    if (tmp != ZERONP_NULL)
    {
        mexCallMatlabHessian(ZERONP_NULL, ZERONP_NULL, 0, 0, 1, tmp);
        stgs->hess = 1;
        zeronp_printf("ZeroNP--> Using second-order method in optimization. \n");
    }
    else
    {
        if (stgs->grad && !stgs->rs)
        {
            zeronp_printf("ZeroNP--> Using first-order method in optimization. \n");
        }
        stgs->hess = 0;
    }

    ob = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * m);
    for (i = 0; i < m; i++)
    {
        ob[i] = (zeronp_float)mxGetPr(lhs)[i];
    }
    nec = m - 1 - nic;
    nc = m - 1;

    mxDestroyArray(rhs[1]);
    // mxDestroyArray(rhs[2]);
    if (lhs != ZERONP_NULL)
    {
        mxDestroyArray(lhs);
    }

    if (nrhs > 3)
    {
        lmex = prhs[3];
    }
    if (lmex && !mxIsEmpty(lmex))
    {
        m = mxGetM(lmex);
        if (m == 1)
        {
            m = mxGetN(lmex);
        }
        l = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * m);

        for (i = 0; i < m; i++)
        {
            l[i] = mxGetPr(lmex)[i];
        }
    }
    else
    {
        if (nc > 0)
        {
            l = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * nc);
            for (i = 0; i < nc; i++)
            {
                l[i] = 0;
            }
        }
        else
        {
            l = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * 1);
            l[0] = 0;
        }
    }

    if (nrhs > 4)
    {
        hmex = prhs[4];
    }
    if (hmex && !mxIsEmpty(hmex))
    {

        h = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * mxGetM(hmex) * mxGetN(hmex));

        for (i = 0; i < mxGetM(hmex) * mxGetN(hmex); i++)
        {
            h[i] = mxGetPr(hmex)[i];
        }
    }
    else
    {
        if (stgs->bfgs)
        {
            h = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * (np + nic) * (np + nic));
            memset(h, 0, sizeof(zeronp_float) * (np + nic) * (np + nic));

            for (i = 0; i < np + nic; i++)
            {
                h[i * (np + nic) + i] = 1;
            }
        }
        else
        {
            h = ZERONP_NULL;
        }
    }

    ZERONPCost *cost = malloc_cost(nec, nic, &mexCallMatlabCost, &mexCallMatlabGradient, &mexCallMatlabHessian);
    cost->obj = ob[0];
    for (i = 0; i < nec; i++)
    {
        cost->ec[i] = ob[1 + i];
    }
    for (i = 0; i < nic; i++)
    {
        cost->ic[i] = ob[1 + nec + i];
    }

    zeronp_free(ob);

    ZERONPIput *input = (ZERONPIput *)zeronp_malloc(sizeof(ZERONPIput));
    input->cnstr = constraint;
    input->stgs = stgs;
    input->l = l;
    input->h = h;
    input->n = np;

    ZERONPSol *sol = (ZERONPSol *)zeronp_malloc(sizeof(ZERONPSol));
    ZERONPInfo *info = (ZERONPInfo *)zeronp_malloc(sizeof(ZERONPInfo));

    zeronp_float *ib0_p = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * (np + nic));

    if (nic > 0)
    {
        memcpy(ib0_p, ib0, nic * sizeof(zeronp_float));
        zeronp_free(ib0);
    }

    memcpy(&ib0_p[nic], p, np * sizeof(zeronp_float));

    zeronp_free(p);

    info->total_time = 0;

    // zeronp_printf("input:\n");
    // zeronp_printf("input->cnstr:\n");
    // if (nic > 0)
    // {
    //     zeronp_printf("input->cnstr->il:\n");

    //     for (
    //         int i = 0;
    //         i < nic;
    //         i++)
    //     {
    //         zeronp_printf("%f ", input->cnstr->il[i]);
    //     }
    //     zeronp_printf("\n");

    //     zeronp_printf("input->cnstr->iu:\n");

    //     for (
    //         int i = 0;
    //         i < nic;
    //         i++)
    //     {
    //         zeronp_printf("%f ", input->cnstr->iu[i]);
    //     }
    //     zeronp_printf("\n");

    //     // memcpy(constraint->il, ibl, nic * sizeof(zeronp_float));
    //     // memcpy(constraint->iu, ibu, nic * sizeof(zeronp_float));
    // }
    // else
    // {
    //     zeronp_printf("nic = 0\n");
    // }

    // if (np > 0)
    // {
    //     zeronp_printf("input->cnstr->pl:\n");

    //     for (
    //         int i = 0;
    //         i < np;
    //         i++)
    //     {
    //         zeronp_printf("%f ", input->cnstr->pl[i]);
    //     }
    //     zeronp_printf("\n");

    //     zeronp_printf("input->cnstr->pu:\n");

    //     for (
    //         int i = 0;
    //         i < np;
    //         i++)
    //     {
    //         zeronp_printf("%f ", input->cnstr->pu[i]);
    //     }
    //     zeronp_printf("\n");

    //     // memcpy(constraint->pl, pbl, np*sizeof(zeronp_float));
    //     // memcpy(constraint->pu, pbu, np*sizeof(zeronp_float));
    // }
    // else
    // {
    //     zeronp_printf("np = 0\n");
    // }

    // zeronp_printf("input->cnstr->Ipc:\n");
    // for (int i = 0; i < 2; i++)
    // {
    //     zeronp_printf("%f ", input->cnstr->Ipc[i]);
    // }
    // zeronp_printf("\n");

    // zeronp_printf("input->cnstr->Ipb:\n");
    // for (int i = 0; i < 2; i++)
    // {
    //     zeronp_printf("%f ", input->cnstr->Ipb[i]);
    // }
    // zeronp_printf("\n");

    // // memcpy(constraint->Ipc, Ipc, 2*sizeof(zeronp_float));
    // // memcpy(constraint->Ipb, Ipb, 2*sizeof(zeronp_float));

    // zeronp_printf("ib0_p:\n");
    // for (int i = 0; i < np + nic; i++)
    // {
    //     if (i == 358 || i == np + nic - 163)
    //     {
    //         zeronp_printf("\n");
    //     }
    //     zeronp_printf("%f ", ib0_p[i]);
    // }
    // zeronp_printf("\n");

    // zeronp_printf("input->stgs:\n");
    // zeronp_printf("pen_l1: %f\n", input->stgs->pen_l1);
    // zeronp_printf("rho: %f\n", input->stgs->rho);
    // zeronp_printf("max_iter: %d\n", input->stgs->max_iter);
    // zeronp_printf("min_iter: %d\n", input->stgs->min_iter);
    // zeronp_printf("max_iter_rescue: %d\n", input->stgs->max_iter_rescue);
    // zeronp_printf("min_iter_rescue: %d\n", input->stgs->min_iter_rescue);
    // zeronp_printf("delta: %f\n", input->stgs->delta);
    // zeronp_printf("tol: %f\n", input->stgs->tol);
    // zeronp_printf("tol_con: %f\n", input->stgs->tol_con);
    // zeronp_printf("ls_time: %d\n", input->stgs->ls_time);
    // zeronp_printf("tol_restart: %f\n", input->stgs->tol_restart);
    // zeronp_printf("re_time: %d\n", input->stgs->re_time);
    // zeronp_printf("delta_end: %f\n", input->stgs->delta_end);
    // zeronp_printf("maxfev: %d\n", input->stgs->maxfev);
    // zeronp_printf("noise: %d\n", input->stgs->noise);
    // zeronp_printf("qpsolver: %d\n", input->stgs->qpsolver);
    // zeronp_printf("k_r: %f\n", input->stgs->k_r);
    // zeronp_printf("k_i: %f\n", input->stgs->k_i);
    // zeronp_printf("c_r: %f\n", input->stgs->c_r);
    // zeronp_printf("c_i: %f\n", input->stgs->c_i);
    // zeronp_printf("batchsize: %d\n", input->stgs->batchsize);
    // zeronp_printf("hess: %d\n", input->stgs->hess);
    // zeronp_printf("grad: %d\n", input->stgs->grad);
    // zeronp_printf("rescue: %d\n", input->stgs->rescue);
    // zeronp_printf("ls_way: %d\n", input->stgs->ls_way);
    // zeronp_printf("bfgs: %d\n", input->stgs->bfgs);
    // zeronp_printf("rs: %d\n", input->stgs->rs);
    // zeronp_printf("scale: %d\n", input->stgs->scale);
    // zeronp_printf("drsom: %d\n", input->stgs->drsom);
    // zeronp_printf("\n");

    // zeronp_printf("\nnec: %d", nec);
    // zeronp_printf("nic: %d\n", nic);
    // zeronp_printf("Ipb[0]: %f\n", constraint->Ipb[0]);
    // zeronp_printf("Ipb[1]: %f\n", constraint->Ipb[1]);
    // zeronp_printf("Ipc[0]: %f\n", constraint->Ipc[0]);
    // zeronp_printf("Ipc[1]: %f\n", constraint->Ipc[1]);

    zeronp_int status = ZERONP(main)(input, cost, ib0_p, sol, info);

    info->total_time += ZERONP(tocq)(&total_timer) / 1e3;
    info->cost_time = cost_time;
    cost_time = 0;

    printf("total time is:%e\n", info->total_time);
    // printf("\ntime for calling q  p solver is:%f(%.2f%%)", info->qpsolver_time, info->qpsolver_time/info->total_time * 100);
    // printf("\ntime for calculating cost function is:%f(%.2f%%)\n", info->cost_time, info->cost_time / info->total_time * 100);

    zeronp_free(ib0_p);

    plhs[0] = mxCreateStructArray(1, one, num_sol_fields, sol_fields);

    tmp = mxCreateDoubleMatrix(1, 1, mxREAL);
    mxSetField(plhs[0], 0, "iter", tmp);
    *mxGetPr(tmp) = (zeronp_float)sol->iter;

    tmp = mxCreateDoubleMatrix(1, 1, mxREAL);
    mxSetField(plhs[0], 0, "count_cost", tmp);
    *mxGetPr(tmp) = (zeronp_float)(sol->count_cost);

    tmp = mxCreateDoubleMatrix(1, 1, mxREAL);
    mxSetField(plhs[0], 0, "count_grad", tmp);
    *mxGetPr(tmp) = (zeronp_float)(sol->count_grad);

    tmp = mxCreateDoubleMatrix(1, 1, mxREAL);
    mxSetField(plhs[0], 0, "count_hess", tmp);
    *mxGetPr(tmp) = (zeronp_float)(sol->count_hess);

    tmp = mxCreateDoubleMatrix(1, 1, mxREAL);
    mxSetField(plhs[0], 0, "constraint", tmp);
    *mxGetPr(tmp) = sol->constraint;

    tmp = mxCreateDoubleMatrix(1, 1, mxREAL);
    mxSetField(plhs[0], 0, "restart_time", tmp);
    *mxGetPr(tmp) = sol->restart_time;

    tmp = mxCreateDoubleMatrix(1, 1, mxREAL);
    mxSetField(plhs[0], 0, "obj", tmp);
    *mxGetPr(tmp) = sol->obj;

    tmp = mxCreateDoubleMatrix(1, 1, mxREAL);
    mxSetField(plhs[0], 0, "status", tmp);
    *mxGetPr(tmp) = (zeronp_float)sol->status;

    tmp = mxCreateDoubleMatrix(1, 1, mxREAL);
    mxSetField(plhs[0], 0, "solve_time", tmp);
    *mxGetPr(tmp) = (zeronp_float)info->total_time;

    set_output_field(&tmp, sol->p, np, 1);
    mxSetField(plhs[0], 0, "p", tmp);

    set_output_field(&tmp, sol->best_fea_p, np, 1);
    mxSetField(plhs[0], 0, "best_fea_p", tmp);

    set_output_field(&tmp, sol->ic, MAX(nic, 1), 1);
    mxSetField(plhs[0], 0, "ic", tmp);

    set_output_field(&tmp, sol->count_h, sol->iter + 1, 1);
    mxSetField(plhs[0], 0, "cost_his", tmp);

    set_output_field(&tmp, sol->jh, sol->iter + 1, 1);
    mxSetField(plhs[0], 0, "jh", tmp);

    set_output_field(&tmp, sol->ch, sol->iter + 1, 1);
    mxSetField(plhs[0], 0, "ch", tmp);

    set_output_field(&tmp, sol->l, MAX(nc, 1), 1);
    mxSetField(plhs[0], 0, "l", tmp);

    if (stgs->bfgs)
    {
        set_output_field(&tmp, sol->h, np + nic, np + nic);
    }
    else
    {
        set_output_field(&tmp, sol->h, 1, 1);
    }
    mxSetField(plhs[0], 0, "h", tmp);

    zeronp_free(info);
    ZERONP(free_sol)
    (sol);

    // close function handle
    mexCallMatlabCost(ZERONP_NULL, ZERONP_NULL, 0, 0, -1, ZERONP_NULL);
    mexCallMatlabGradient(ZERONP_NULL, ZERONP_NULL, 0, 0, -1, ZERONP_NULL);
    mexCallMatlabHessian(ZERONP_NULL, ZERONP_NULL, 0, 0, -1, ZERONP_NULL);
}
