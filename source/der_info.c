#include "der_info.h"
#include "linalg.h"
#include "linsys.h"
#include "qp_solver.h"

zeronp_float* Est_noise(
    zeronp_int nf,
    zeronp_float* fval
) {
    // This subroutine implement the ECnoise program proposed by Argonne National Laboratory
    // Jorge More' and Stefan Wild. November 2009. 
    // The input: nf: number of points 
    // fval: array of funciton values
    // Output: fnoise_inform: the first entry is estimated noise. The second entry is infomration of the output.
    zeronp_float* fnoise_inform = (zeronp_float*) zeronp_malloc(2*sizeof(zeronp_float));
    zeronp_float* level = (zeronp_float*)zeronp_calloc(nf - 1, sizeof(zeronp_float));
    zeronp_float* dsgn = (zeronp_float*)zeronp_calloc(nf - 1, sizeof(zeronp_float));
    fnoise_inform[0] = 0;
    zeronp_float gamma = 1.;
    zeronp_float fmin = zeronp_min(fval, nf);
    zeronp_float fmax = zeronp_max(fval, nf);
    if ((fmax-fmin)/(MAX(ABS(fmin),ABS(fmax))) > .1)
    {
        fnoise_inform[1] = 3;
        zeronp_free(level);
        zeronp_free(dsgn);
        return fnoise_inform;
    }
    for (zeronp_int j = 0; j < nf-1; j++) {
        zeronp_int count_zero = 0;
        for (zeronp_int i = 0; i < nf - j; i++) {
            fval[i] = fval[i + 1] - fval[i];        
        }
        for (zeronp_int i = 0; i < nf - 1; i++) {
            if (fval[i] == 0) {
                count_zero++;
            }
        }

        if (j == 1 && count_zero >= nf/2) {
            fnoise_inform[1] = 2;
            zeronp_free(level);
            zeronp_free(dsgn);
            return fnoise_inform;
        }
        gamma *= 0.5 * ((j+1.) / (2 * j + 1.));
        
        //Compute the estimates for the noise level.
        level[j] = SQRTF(gamma * ZERONP(norm_sq)(fval, nf - j) / (nf - j));

        //Determine differences in sign.
        zeronp_float emin = ZERONP(min)(fval, nf - j);
        zeronp_float emax = ZERONP(max)(fval, nf - j);
        if (emin * emax < 0) {
            dsgn[j] = 1;
        }
    }

    for (zeronp_int k = 0; k < nf - 3; k++) {
        zeronp_float emin = ZERONP(min)(&level[k], 3);
        zeronp_float emax = ZERONP(max)(&level[k], 3);
        if (emax <= 4 * emin && dsgn[k]) {
            fnoise_inform[0] = level[k];
            fnoise_inform[1] = 1;
            zeronp_free(level);
            zeronp_free(dsgn);
            return fnoise_inform;
        }
    }

    fnoise_inform[1] = 3;
    zeronp_free(level);
    zeronp_free(dsgn);
    return fnoise_inform;
}

zeronp_float* Est_second_diff_sub(
    zeronp_float fval,
    zeronp_float delta,
    zeronp_float fnoise,
    zeronp_float* p,
    zeronp_float* d,
    SUBNPWork* w_sub,
    ZERONPWork* w,
    ZERONPSettings* stgs

) {
    zeronp_float tau1 = 100;
    zeronp_float tau2 = 0.1;
    zeronp_float* res = (zeronp_float*)zeronp_malloc(2*sizeof(zeronp_float));
    zeronp_float delta_h;
    ZERONPCost** obm = (ZERONPCost**)zeronp_malloc(1 * sizeof(ZERONPCost*));
    obm[0] = init_cost(w->nec, w->nic);
    zeronp_float* ptemp = (zeronp_float*)zeronp_malloc(w_sub->J->npic * sizeof(zeronp_float));
    memcpy(ptemp, p, (w->nic+w->n) * sizeof(zeronp_float));
    zeronp_add_scaled_array(ptemp, d, w->n + w->nic,delta);
    calculate_scaled_cost(obm, ptemp, w_sub->scale, stgs, w, 1);
    zeronp_float alm_f = obm[0]->obj;

    if (w_sub->nc > 0)
    {
        alm_f = calculate_ALM(obm[0], stgs, ptemp, w, w_sub);
    }

    zeronp_add_scaled_array(ptemp, d, w->n + w->nic, -2*delta);
    calculate_scaled_cost(obm, ptemp, w_sub->scale, stgs, w, 1);
    zeronp_float alm_b = obm[0]->obj;

    if (w_sub->nc > 0)
    {
        alm_b = calculate_ALM(obm[0], stgs, ptemp, w, w_sub);
    }

    delta_h = ABS(alm_f + alm_b - fval * 2);
    res[0] = delta_h / (delta * delta);
    if (delta_h >= fnoise * tau1 && ABS(alm_f - fval) <= tau2 * MAX(ABS(fval), ABS(alm_f)) \
        && ABS(alm_b - fval) <= tau2 * MAX(ABS(fval), ABS(alm_b))) {
        res[1] = 1;
    }
    else {
        res[1] = 0;
    }

    zeronp_free(ptemp);
    free_cost(obm[0]);
    zeronp_free(obm);
    return res;
}

zeronp_float Est_second_diff(
    zeronp_float fnoise,
    zeronp_float fval,
    zeronp_float* p,
    zeronp_float* d,
    SUBNPWork* w_sub,
    ZERONPWork* w,
    ZERONPSettings* stgs
) {
    // This subroutine estimate the second-order derivative of a function  

    zeronp_float ha = pow(fnoise, 0.25);
    zeronp_float mu;
    zeronp_float* mua = Est_second_diff_sub(fval, ha, fnoise, p, d, w_sub, w, stgs);
    if (mua[1] == 1){
        mu = mua[0];
        zeronp_free(mua);
        return mu;
    }
    zeronp_float hb = pow(fnoise/mua[0], 0.25);
    zeronp_float* mub = Est_second_diff_sub(fval, hb, fnoise, p, d, w_sub, w, stgs);
    if (mub[1] == 1 || ABS(mua[0]-mub[0]) <= 0.5*mub[0] ) {
        mu = mub[0];
        zeronp_free(mua);
        zeronp_free(mub);
        return mu;
    }

    //The noise is too large.
    zeronp_free(mua);
    zeronp_free(mub);
    return 1;
}

zeronp_float calculate_delta(
    zeronp_int nf,
    zeronp_float fval,
    zeronp_float* p,
    SUBNPWork* w_sub,
    ZERONPWork* w,
    ZERONPSettings* stgs
) {
    // This subroutine is used to calculate the step size of finite-difference method
    zeronp_float* fvals = (zeronp_float*)zeronp_malloc(nf * sizeof(zeronp_float));
    ZERONPCost** obm = (ZERONPCost**)zeronp_malloc(1 * sizeof(ZERONPCost*));
    obm[0] = init_cost(w->nec, w->nic);
    zeronp_float h_default = stgs->h;
    zeronp_float* d = (zeronp_float*)zeronp_malloc((w->n + w->nic) * sizeof(zeronp_float));
    Uniform_sphere(d, w->n, 1);

    for (zeronp_int i = 0; i < (nf - 1) / 2; i++) {
        zeronp_float* ptemp = (zeronp_float*)zeronp_malloc(w_sub->J->npic * sizeof(zeronp_float));
        memcpy(ptemp, p, (w->nic + w->n) * sizeof(zeronp_float));
        zeronp_add_scaled_array(ptemp, d, w->n, h_default);
        calculate_scaled_cost(obm, ptemp, w_sub->scale, stgs, w, 1);
        fvals[i] = obm[0]->obj;

        if (w_sub->nc > 0)
        {
            fvals[i] = calculate_ALM(obm[0], stgs, ptemp, w, w_sub);
        }

        zeronp_add_scaled_array(ptemp, d, w->n, -2*h_default);
        calculate_scaled_cost(obm, ptemp, w_sub->scale, stgs, w, 1);
        fvals[nf-i-1] = obm[0]->obj;

        if (w_sub->nc > 0)
        {
            fvals[nf - i - 1] = calculate_ALM(obm[0], stgs, ptemp, w, w_sub);
        }
        zeronp_free(ptemp);
    }
    fvals[(nf - 1) / 2] = fval;
    // Estimate noise
    zeronp_float* res_ecnoise = Est_noise(nf, fvals);
    if (res_ecnoise[1] != 1) {
        if (res_ecnoise[1] == 2) {
            stgs->h *= 10;
        }
        else {
            stgs->h /= 10;
        }

        zeronp_free(res_ecnoise);
        zeronp_free(d);
        zeronp_free(fvals);
        free_cost(obm[0]);
        return -1.;
    }
    // Estimate second-order derivative
    zeronp_float v2 = Est_second_diff(res_ecnoise[0], fval, p, d, w_sub, w, stgs);
    zeronp_float delta = pow(8, 0.25) * pow(res_ecnoise[0] / v2, 0.5);

    zeronp_free(res_ecnoise);
    zeronp_free(d);
    zeronp_free(fvals);
    free_cost(obm[0]);
    zeronp_free(obm);
    return delta;
}

zeronp_float calculate_infeas_scaledob(
    ZERONPCost *ob,
    ZERONPWork *w,
    zeronp_float *x,
    zeronp_float *scale)
{
    zeronp_float con = 0;
    zeronp_int i;
    // calculate constraints
    if (w->nic > 0.5)
    {
        for (i = 0; i < w->nic; i++)
        {
            if (ob->ic[i] < w->pb->il[i])
            {
                con += (w->pb->il[i] - ob->ic[i]) * (w->pb->il[i] - ob->ic[i]) * scale[1 + w->nec + i] * scale[1 + w->nec + i];
            }
            if (ob->ic[i] > w->pb->iu[i])
            {
                con += (ob->ic[i] - w->pb->iu[i]) * (ob->ic[i] - w->pb->iu[i]) * scale[1 + w->nec + i] * scale[1 + w->nec + i];
            }
        }
    }
    if (w->nec > 0.5)
    {
        for (i = 0; i < w->nec; i++)
        {
            con += ob->ec[i] * ob->ec[i] * scale[1 + i] * scale[1 + i];
        }
    }
    for (i = 0; i < w->pb->n; i++)
    {
        if (x[i] > w->pb->pu[i])
        {
            con += (x[i] - w->pb->pu[i]) * (x[i] - w->pb->pu[i]) * scale[1 + w->nec + w->nic + i] * scale[1 + w->nec + w->nic + i];
        }
        if (x[i] < w->pb->pl[i])
        {
            con += (x[i] - w->pb->pl[i]) * (x[i] - w->pb->pl[i]) * scale[1 + w->nec + w->nic + i] * scale[1 + w->nec + w->nic + i];
        }
    }
    con = SQRTF(con);
    return con;
}

zeronp_float calculate_infeas_scaledob_l1(
    ZERONPCost *ob,
    ZERONPWork *w,
    zeronp_float *x)
{
    zeronp_float con = 0;
    zeronp_int i;
    // calculate constraints
    if (w->nic > 0.5)
    {
        for (i = 0; i < w->nic; i++)
        {
            if (ob->ic[i] < w->pb->il[i])
            {
                con += (w->pb->il[i] - ob->ic[i]) * (w->pb->il[i] - ob->ic[i]);
            }
            if (ob->ic[i] > w->pb->iu[i])
            {
                con += (ob->ic[i] - w->pb->iu[i]) * (ob->ic[i] - w->pb->iu[i]);
            }
        }
    }
    if (w->nec > 0.5)
    {
        for (i = 0; i < w->nec; i++)
        {
            con += (ob->ec[i] + x[i + w->n - 2 * w->nec] - x[i + w->n - w->nec]) * (ob->ec[i] + x[i + w->n - 2 * w->nec] - x[i + w->n - w->nec]);
        }
    }
    for (i = 0; i < w->n - 2 * w->nec; i++)
    {
        if (x[i] > w->pb->pu[i])
        {
            con += (x[i] - w->pb->pu[i]) * (x[i] - w->pb->pu[i]);
        }
        if (x[i] < w->pb->pl[i])
        {
            con += (x[i] - w->pb->pl[i]) * (x[i] - w->pb->pl[i]);
        }
    }
    con = SQRTF(con);
    return con;
}

zeronp_float calculate_infeas_unscaleob(
    ZERONPCost *ob,
    ZERONPWork *w,
    zeronp_float *x,
    zeronp_float *scale)
{
    zeronp_float con = 0;
    zeronp_int i;
    // calculate constraints
    if (w->nic > 0.5)
    {
        for (i = 0; i < w->nic; i++)
        {
            if (ob->ic[i] < w->pb->il[i] * scale[1 + w->nec + i])
            {
                con += (w->pb->il[i] * scale[1 + w->nec + i] - ob->ic[i]) * (w->pb->il[i] * scale[1 + w->nec + i] - ob->ic[i]);
            }
            if (ob->ic[i] > w->pb->iu[i] * scale[1 + w->nec + i])
            {
                con += (ob->ic[i] - w->pb->iu[i] * scale[1 + w->nec + i]) * (ob->ic[i] - w->pb->iu[i] * scale[1 + w->nec + i]);
            }
        }
    }
    if (w->nec > 0.5)
    {
        for (i = 0; i < w->nec; i++)
        {
            con += ob->ec[i] * ob->ec[i];
        }
    }
    for (i = 0; i < w->pb->n; i++)
    {
        if (x[i] > w->pb->pu[i] * scale[1 + w->nec + w->nic + i])
        {
            con += (x[i] - w->pb->pu[i] * scale[1 + w->nec + w->nic + i]) * (x[i] - w->pb->pu[i] * scale[1 + w->nec + w->nic + i]);
        }
        if (x[i] < w->pb->pl[i] * scale[1 + w->nec + w->nic + i])
        {
            con += (x[i] - w->pb->pl[i] * scale[1 + w->nec + w->nic + i]) * (x[i] - w->pb->pl[i] * scale[1 + w->nec + w->nic + i]);
        }
    }
    con = SQRTF(con);
    return con;
}
zeronp_float calculate_almcrit_iq(
    ZERONPWork *w,
    zeronp_float *scale,
    zeronp_float *p)
{
    zeronp_float almcrit = 0;
    zeronp_float temp;
    zeronp_int i;
    for (i = 0; i < w->nic; i++)
    {
        temp = p[i] * scale[w->nec + 1 + i] - scale[0] * w->l[w->nec + i] / scale[w->nec + 1 + i];
        if (temp > w->pb->iu[i] * scale[w->nec + 1 + i])
        {
            temp = w->pb->iu[i] * scale[w->nec + 1 + i];
        }
        else if (temp < w->pb->il[i] * scale[w->nec + 1 + i])
        {
            temp = w->pb->il[i] * scale[w->nec + 1 + i];
        }
        almcrit += (temp - p[i] * scale[w->nec + 1 + i]) * (temp - p[i] * scale[w->nec + 1 + i]);
    }
    return almcrit;
}
void calculate_scaled_cost(
    ZERONPCost **ob,
    zeronp_float *p,
    const zeronp_float *scale,
    ZERONPSettings *stgs,
    ZERONPWork *w,
    zeronp_int nfeval)
{
    zeronp_int i, j;
    zeronp_int len = stgs->rescue ? w->n - 2 * w->nec : w->n;
    w->pb->n = len;
    zeronp_float *x = (zeronp_float *)zeronp_malloc(len * nfeval * sizeof(zeronp_float));
    for (j = 0; j < nfeval; j++)
    {
        for (i = 0; i < len; i++)
        {
            x[i + j * len] = p[i + j * w->n] * scale[1 + w->nc + i];
        }
    }
    // calculate rescaled cost
    // (*w->ob->cost)(ob, x, w->n, 0);
    w->ob->cost(ob, x, len, nfeval, 0);
    w->count_cost += nfeval;

    for (j = 0; j < nfeval; j++)
    {
        zeronp_float con = calculate_infeas_unscaleob(ob[j], w, &x[j * len], scale);
        if (con <= stgs->tol_con && w->bestobj > ob[j]->obj)
        {
            w->bestcon = con;
            w->bestobj = ob[j]->obj;
            memcpy(w->bestp, &x[j * len], len * sizeof(zeronp_float));
            memcpy(w->bestl, w->l, MAX(w->nc, 1) * sizeof(zeronp_float));
            copyZERONPCost(w->bestob, ob[j]);
        }

        if (con < w->best_fea_con)
        {
            w->best_fea_con = con;
            memcpy(w->best_fea_p, &x[j * len], len * sizeof(zeronp_float));
            memcpy(w->best_fea_l, w->l, MAX(w->nc, 1) * sizeof(zeronp_float));
            copyZERONPCost(w->best_fea_ob, ob[j]);
        }
        if (stgs->rescue && w->nec)
        {
            // ob[j]->obj = ZERONP(norm_sq)(w->p + w->nic + w->n - w->nec, w->nec);
            //  Modify equality constraints and objective
            for (i = 0; i < w->nec; i++)
            {
                ob[j]->obj += w->pen_l1 * (p[i + w->n - 2 * w->nec + j * w->n] * scale[1 + w->nc + i + w->n - 2 * w->nec] + p[i + w->n - w->nec + j * w->n] * scale[1 + w->nc + i + w->n - w->nec]);
            }

            for (i = 0; i < w->nec; i++)
            {
                ob[j]->ec[i] += -p[i + w->n - 2 * w->nec + j * w->n] * scale[1 + w->nc + i + w->n - 2 * w->nec] + p[i + w->n - w->nec + j * w->n] * scale[1 + w->nc + i + w->n - w->nec];
            }
        }
        rescale_ob(ob[j], scale);
    }
    if (w->count_cost >= stgs->maxfev)
    {
        w->exit = 1;
    }
    w->pb->n = w->n;
    // free x
    zeronp_free(x);
}
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
)
{
    zeronp_int i, j;
    zeronp_int len = w->n - w->nec;
    zeronp_float* x = (zeronp_float*)zeronp_malloc(len * nfeval * sizeof(zeronp_float));
    for (j = 0; j < nfeval; j++) {
        for (i = 0; i < len; i++) {
            x[i + j * len] = p[i + j * len] * scale[1 + w->nc + i];
        }
    }
    // calculate rescaled cost
    // (*w->ob->cost)(ob, x, len, 0);
    w->ob->cost(ob, x, len, nfeval, 0);
    w->count_cost += nfeval;

    for (j = 0; j < nfeval; j++) {
        if (stgs->rescue) {
            w->n -= w->nec;
        }

        zeronp_float con = calculate_infeas_unscaleob(ob[j], w, &x[j * len], scale);
        if (stgs->rescue) {
            w->n += w->nec;
        }

        if (con <= stgs->tol_con && w->bestobj > ob[j]->obj) {
            w->bestcon = con;
            w->bestobj = ob[j]->obj;
            memcpy(w->bestp, &x[j * len], (len) * sizeof(zeronp_float));
            for (i = 0; i < w->nc; i++) {
                w->bestl[i] = w->l[i] * scale[0] / scale[1 + i];
            }
        }
        if (con < w->best_fea_con) {
            w->best_fea_con = con;
            memcpy(w->best_fea_p, &x[j * len], (len) * sizeof(zeronp_float));
            for (i = 0; i < w->nc; i++) {
                w->best_fea_l[i] = w->l[i] * scale[0] / scale[1 + i];
            }
        }
        if (stgs->rescue && w->nec) {
            //ob[j]->obj = ZERONP(norm_sq)(w->p + w->nic + w->n - w->nec, w->nec);
            ob[j]->obj = 0;
            for (i = 0; i < w->nec; i++) {
                ob[j]->obj += slack[i] * scale[1 + w->nc + w->n - w->nec + i]* slack[i] * scale[1 + w->nc + w->n - w->nec + i];
            }

            for (i = 0; i < w->nec; i++) {
                ob[j]->ec[i] -= slack[i] * scale[1 + w->nc + w->n - w->nec + i];
            }
        }
        rescale_ob(ob[j], scale);
    }
    if (w->count_cost >= stgs->maxfev) {
        w->exit = 1;
    }

    // free x
    zeronp_free(x);
}*/

zeronp_float line_search_merit(
    ZERONPCost *ob,
    ZERONPWork *w,
    SUBNPWork *w_sub,
    zeronp_float *p)
{
    // Input: ob after modification
    // Reminder: The function is for rescue case! I will write the  general function later.
    // Output: L1 exact penlaty function value
    zeronp_float result;
    zeronp_int i;

    result = ob->obj * w_sub->scale[0];
    for (i = 0; i < w->nec; i++)
    {
        // recover objective
        result -= w->pen_l1 * (p[w->nic + w->n + i - 2 * w->nec] * w_sub->scale[1 + w->nc + i + w->n - 2 * w->nec] + p[w->nic + w->n + i - 1 * w->nec] * w_sub->scale[1 + w->nc + i + w->n - 1 * w->nec]);
        // equality penalty
        result += w->pen_l1 * ABS(ob->ec[i] + p[w->nic + w->n + i - 2 * w->nec] * w_sub->scale[1 + w->nc + i + w->n - 2 * w->nec] - p[w->nic + w->n + i - 1 * w->nec] * w_sub->scale[1 + w->nc + i + w->n - 1 * w->nec]) * w_sub->scale[1 + i];
    }
    for (i = 0; i < w->nic; i++)
    {
        // ineqality penalty
        if (ob->ic[i] < w->pb->il[i])
        {
            result += w->pen_l1 * (w->pb->il[i] - ob->ic[i]) * w_sub->scale[w->nec + 1 + i];
        }
        if (ob->ic[i] > w->pb->iu[i])
        {
            result += w->pen_l1 * (ob->ic[i] - w->pb->iu[i]) * w_sub->scale[w->nec + 1 + i];
        }
    }
    return result;
}

zeronp_int calculate_scaled_grad(
    zeronp_float *g,
    zeronp_float *p,
    const zeronp_float *scale,
    ZERONPSettings *stgs,
    ZERONPWork *w)
{
    zeronp_float *x = (zeronp_float *)zeronp_malloc(w->n * sizeof(zeronp_float));
    zeronp_int i, j;

    for (i = 0; i < w->n; i++)
    {
        x[i] = p[i] * scale[1 + w->nc + i];
    }
    w->ob->grad(g, x, w->n, w->nc + 1, 0);
    for (i = 0; i < w->n; i++)
    {
        g[i] = g[i] / scale[0] * scale[1 + w->nc + i];
    }
    for (j = 0; j < w->nc; j++)
    {
        for (i = 0; i < w->n; i++)
        {
            g[w->n * (j + 1) + i] = g[w->n * (j + 1) + i] * scale[1 + w->nc + i] / scale[1 + j];
        }
    }
    w->count_grad++;
    zeronp_free(x);
}

zeronp_int calculate_scaled_grad_random(
    zeronp_float *g,
    zeronp_float *p,
    ZERONPCost *ob_p,
    const zeronp_float *scale,
    ZERONPSettings *stgs,
    ZERONPWork *w)
{
    zeronp_float *x = (zeronp_float *)zeronp_malloc(w->n * sizeof(zeronp_float));
    zeronp_float *x_temp = (zeronp_float *)zeronp_malloc(w->n * sizeof(zeronp_float));
    zeronp_float *diff = (zeronp_float *)zeronp_malloc(w->n * sizeof(zeronp_float));
    ZERONPCost *ob = init_cost(w->nec, w->nic);
    ZERONPCost* ob_backward = init_cost(w->nec, w->nic);
    zeronp_int *index = ZERONP_NULL;
    zeronp_int i, j, k;

    // unscale x
    for (i = 0; i < w->n; i++)
    {
        x[i] = p[i] * scale[1 + w->nc + i];
        g[i] = 0;
    }
    // unscale ob
    unscale_ob(ob_p, scale);
    // Estimate the gradient by random sampling, Batchsize = stgs->batchsize

    zeronp_int batch = MIN(stgs->batchsize, stgs->maxfev - w->count_cost);

    if (batch == 0)
    {
        w->exit = 1;
        return 0;
    }
    if (stgs->rs == 2)
    {
        index = (zeronp_int *)zeronp_calloc(sizeof(zeronp_int), w->n);
    }
    for (i = 0; i < batch; i++)
    {
        if (stgs->rs == 1)
        {
            // Generate random Guassian Vector
            Uniform_sphere(diff, w->n, 1.);
            memcpy(x_temp, x, w->n * sizeof(zeronp_float));
            ZERONP(add_scaled_array)
            (x_temp, diff, w->n, stgs->delta);
            // Estimate Gradient
            w->ob->cost(&ob, x_temp, w->n, 1, 0);
            w->count_cost += 1;

            if (stgs->cen_diff) {
                ZERONP(add_scaled_array)
                    (x_temp, diff, w->n, -2*stgs->delta);
                // Estimate Gradient
                w->ob->cost(&ob_backward, x_temp, w->n, 1, 0);
                w->count_cost += 1;
            }

            for (k = 0; k < w->n; k++)
            {
                if (stgs->cen_diff) {
                    // Use CENTRE difference to calculate gradient
                    g[k] += (ob->obj - ob_backward->obj) / (2*stgs->delta) * diff[k];
                    for (j = 1; j < w->nec + 1; j++)
                    {
                        g[k + j * w->n] += (ob->ec[j - 1] - ob_backward->ec[j - 1]) / (2 * stgs->delta) * diff[k];
                    }
                    for (j = w->nec + 1; j < w->nec + 1 + w->nic; j++)
                    {
                        g[k + j * w->n] = (ob->ic[j - w->nec - 1] - ob_backward->ic[j - w->nec - 1]) / (2 * stgs->delta) * diff[k];
                    }
                }
                else {
                    // use forward difference to calcualate gradient
                    g[k] += (ob->obj - ob_p->obj) / stgs->delta * diff[k];
                    for (j = 1; j < w->nec + 1; j++)
                    {
                        g[k + j * w->n] += (ob->ec[j - 1] - ob_p->ec[j - 1]) / stgs->delta * diff[k];
                    }
                    /*  if (isnan(g[k])) {
                          g[k] = g[k];
                      }*/
                    for (j = w->nec + 1; j < w->nec + 1 + w->nic; j++)
                    {
                        g[k + j * w->n] = (ob->ic[j - w->nec - 1] - ob_p->ic[j - w->nec - 1]) / stgs->delta * diff[k];
                    }
                }
            }
        }
        else if (stgs->rs == 2)
        {

            zeronp_int r_ind = rand() % w->n;
            while (index[r_ind])
            {
                r_ind = rand() % w->n;
            }
            index[r_ind] = 1;
            memcpy(x_temp, x, w->n * sizeof(zeronp_float));
            x_temp[r_ind] += stgs->delta;
            w->ob->cost(&ob, x_temp, w->n, 1, 0);
            w->count_cost += 1;
            if (stgs->cen_diff) {
                // Use CENTRE differnce to calculate the gradient
                x_temp[r_ind] -= 2 * stgs->delta;
                w->ob->cost(&ob_backward, x_temp, w->n, 1, 0);
                w->count_cost += 1;
                g[r_ind] = (ob->obj - ob_backward->obj) / (2*stgs->delta);
            }
            else {
                // Use forward difference to calculate the gradient
                g[r_ind] = (ob->obj - ob_p->obj) / stgs->delta;
            }
        }
    }

    // Average
    for (k = 0; k < w->n * (1 + w->nc); k++)
    {
        g[k] = g[k] / MAX(i, 1);
    }

    zeronp_float g_norm = ZERONP(norm)(g, w->n);
    // ZERONP(set_as_scaled_array)(g, g, 1 / g_norm, w->n);

    // scale g
    for (i = 0; i < w->n; i++)
    {
        g[i] = g[i] / scale[0] * scale[1 + w->nc + i];
    }
    for (j = 0; j < w->nc; j++)
    {
        for (i = 0; i < w->n; i++)
        {
            g[w->n * (j + 1) + i] = g[w->n * (j + 1) + i] * scale[1 + w->nc + i] / scale[1 + j];
        }
    }

    // sacle ob
    rescale_ob(ob_p, scale);

    w->count_grad++;
    zeronp_free(x);
    zeronp_free(x_temp);
    zeronp_free(diff);
    zeronp_free(index);
    free_cost(ob);
    free_cost(ob_backward);
}

zeronp_int calculate_scaled_hess(
    zeronp_float *h,
    zeronp_float *p,
    const zeronp_float *scale,
    ZERONPSettings *stgs,
    ZERONPWork *w)
{
    zeronp_float *x = (zeronp_float *)zeronp_malloc(w->n * sizeof(zeronp_float));
    zeronp_int i, j, k;

    for (i = 0; i < w->n; i++)
    {
        x[i] = p[i] * scale[1 + w->nc + i];
    }
    w->ob->hess(h, x, w->n, w->nc + 1, 0);

    // rescaled
    for (i = 0; i < w->n; i++)
    {
        for (j = 0; j < w->n; j++)
        {
            h[i + j * w->n] = h[i + j * w->n] / scale[0] * scale[1 + w->nc + i] * scale[1 + w->nc + j];
        }
    }
    for (k = 1; k < w->nc + 1; k++)
    {
        for (i = 0; i < w->n; i++)
        {
            for (j = 0; j < w->n; j++)
            {
                h[i + j * w->n + k * w->n * w->n] *= scale[1 + w->nc + i] * scale[1 + w->nc + j] / scale[k];
            }
        }
    }
    w->count_hess++;
    zeronp_free(x);
}
void calculate_alm_criterion(
    ZERONPWork *w,
    SUBNPWork *w_sub,
    zeronp_float *grad)
{
    // This function is used for calculate the gradient of ALM when provided gradient information
    zeronp_int i;
    zeronp_float *temp_alm = (zeronp_float *)zeronp_malloc(w_sub->npic * sizeof(zeronp_float));
    zeronp_float *gg_y = (zeronp_float *)zeronp_malloc(w->n * sizeof(zeronp_float));
    zeronp_float *temp_l = (zeronp_float *)zeronp_malloc(w->nic * sizeof(zeronp_float));
    memcpy(temp_l, w->l + w->nec, sizeof(zeronp_float) * w->nic);
    memcpy(temp_alm, w->p, w_sub->npic * sizeof(zeronp_float));
    for (i = 0; i < w->nic; i++)
    {
        temp_l[i] *= w_sub->scale[0] / (w_sub->scale[1 + w->nec + i] * w_sub->scale[1 + w->nec + i]);
    }
    // calculate ALM stop criterion
    ZERONP(add_scaled_array)
    (temp_alm + w->nic, grad, w->n, -1.);
    if (w->nc)
    {
        ZERONP(Ax)
        (gg_y, grad + w->n, w->l, w->n, w->nc);
        ZERONP(add_scaled_array)
        (temp_alm + w->nic, gg_y, w->n, 1.);
        ZERONP(add_scaled_array)
        (temp_alm, temp_l, w->nic, -1.);
    }
    proj(temp_alm, w);
    ZERONP(add_scaled_array)
    (temp_alm, w->p, w_sub->npic, -1.);
    for (i = 0; i < w->n; i++)
    {
        temp_alm[i + w->nic] *= w_sub->scale[0] / w_sub->scale[1 + w->nc + i];
    }
    for (i = 0; i < w->nic; i++)
    {
        temp_alm[i] *= w_sub->scale[1 + w->nec + i];
    }
    w->alm_crit = ZERONP(norm)(temp_alm, w_sub->npic);
    zeronp_free(temp_alm);
    zeronp_free(gg_y);
    zeronp_free(temp_l);
}

zeronp_int calculate_Jacob_zero(
    SUBNPWork *w_sub,
    ZERONPWork *w,
    ZERONPSettings *stgs)
{
    zeronp_int len = stgs->rescue ? w->n - 2 * w->nec : w->n;
    w->pb->n = len;
    ZERONPCost **ob = (ZERONPCost **)zeronp_malloc(1 * sizeof(ZERONPCost *));
    zeronp_float *p = (zeronp_float *)zeronp_malloc(1 * w->n * sizeof(zeronp_float));
    memcpy(p, &w->p[w->nic], w->n * sizeof(zeronp_float));

    zeronp_float infeas_cand, temp;
    zeronp_float *atemp = (zeronp_float *)zeronp_malloc(w->nc * len * sizeof(zeronp_float));

    zeronp_int i, j, tag;

    tag = 1;
    //// Initiation
    // for (j = 0; j < len; j++) {
    //     ob[j] = init_cost(w->nec, w->nic);
    //     memcpy(&p[j * w->n], &w->p[w->nic], w->n * sizeof(zeronp_float));
    //     p[j * w->n + j] += stgs->delta;
    // }
    //// parallel calculation
    // calculate_scaled_cost(ob, p, w_sub->scale, stgs, w, len);
    ob[0] = init_cost(w->nec, w->nic);

    for (i = 0; i < len; i++)
    {

        p[i] += stgs->delta;

        calculate_scaled_cost(ob, p, w_sub->scale, stgs, w, 1);

        infeas_cand = calculate_infeas_scaledob(ob[0], w, p, w_sub->scale);
        if (infeas_cand <= stgs->tol_con && MIN(p[i] - w->pb->pl[i], w->pb->pu[i] - p[i]) > 0 && w_sub->ob_cand->obj > ob[0]->obj)
        {
            memcpy(w_sub->p_cand, w->p, w->nic * sizeof(zeronp_float));
            memcpy(&w_sub->p_cand[w->nic], p, w->n * sizeof(zeronp_float));
            copyZERONPCost(w_sub->ob_cand, ob[0]);
        }
        p[i] -= stgs->delta;
        // calculate Jacobian approximately
        w_sub->J->g[w->nic + i] = (ob[0]->obj - w->ob->obj) / stgs->delta;
        for (j = 0; j < w->nec; j++)
        {
            atemp[j + i * w_sub->nc] = (ob[0]->ec[j] - w->ob->ec[j]) / stgs->delta;
        }
        for (j = 0; j < w->nic; j++)
        {
            atemp[(w->nec + j) + (i)*w_sub->nc] = (ob[0]->ic[j] - w->ob->ic[j]) / stgs->delta;
        }

        if (w_sub->J->g[w->nic + i] != 0)
        {
            tag = 0;
        }
        // restore perturb

        temp = w->p[w->nic + i] * w_sub->scale[w->nc + 1 + i] - w_sub->J->g[w->nic + i] * w_sub->scale[0] / w_sub->scale[w->nc + 1 + i];
        if (w->nc > 0)
        {
            temp += w_sub->scale[0] * ZERONP(dot)(&atemp[i * w_sub->nc], w->l, w->nc) / w_sub->scale[w->nc + 1 + i];
        }
        if (temp > w->pb->pu[i] * w_sub->scale[w->nc + 1 + i])
        {
            temp = w->pb->pu[i] * w_sub->scale[w->nc + 1 + i];
        }
        else if (temp < w->pb->pl[i] * w_sub->scale[w->nc + 1 + i])
        {
            temp = w->pb->pl[i] * w_sub->scale[w->nc + 1 + i];
        }
        w->alm_crit += (temp - w->p[w->nic + i] * w_sub->scale[w->nc + 1 + i]) * (temp - w->p[w->nic + i] * w_sub->scale[w->nc + 1 + i]);
    }
    // copy atemp to J->a
    if (w->n > w->nec || stgs->rescue)
    {
        memcpy(&w_sub->J->a[w->nic * w->nc], atemp, w->nc * len * sizeof(zeronp_float));
    }
    else
    {
        for (i = 0; i < w->n; i++)
        {
            for (j = 0; j < w->n - 1; j++)
            {
                w_sub->J->a[(w->nic + i) * w_sub->J->a_row + j] = atemp[i * w->nc + w_sub->constr_index[j]];
            }
        }
    }
    zeronp_free(atemp);

    if (tag)
    {
        w->const_time++;
    }

    if (stgs->rescue == 0)
    {
        // Determin the condtion number of the Jocobi
        zeronp_float *aT = (zeronp_float *)zeronp_malloc(w_sub->J->a_row * w_sub->npic * sizeof(zeronp_float));
        ZERONP(transpose)
        (w_sub->J->a_row, w_sub->npic, w_sub->J->a, aT);
        zeronp_float *aaT = (zeronp_float *)zeronp_malloc(w_sub->J->a_row * w_sub->J->a_row * sizeof(zeronp_float));
        ZERONP(AB)
        (aaT, w_sub->J->a, aT, w_sub->J->a_row, w_sub->npic, w_sub->J->a_row);
        zeronp_float *cond = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float));
        ZERONP(cond)
        (w_sub->J->a_row, aaT, cond);
        zeronp_free(aT);
        zeronp_free(aaT);
        // calculate condition number
        // TODO: calculate condition number
        if (*cond <= EPS)
        {
            zeronp_printf("ZeroNP--> ");
            zeronp_printf("Redundant constraints were found. Poor              \n");
            zeronp_printf("         ");
            zeronp_printf("intermediate results may result.  Suggest that you  \n");
            zeronp_printf("         ");
            zeronp_printf("remove redundant constraints and re-OPTIMIZE.       \n");
        }
        // free pointers
        zeronp_free(cond);
    }
    zeronp_free(p);
    free_cost(ob[0]);
    zeronp_free(ob);
    w->pb->n = w->n;
    return 0;
}

zeronp_int calculate_Jacob_zero_rescue(
    SUBNPWork *w_sub,
    ZERONPWork *w,
    ZERONPSettings *stgs)
{
    zeronp_int i;
    calculate_Jacob_zero(w_sub, w, stgs);
    /*
    for (i = 0; i < w->n - 2*w->nec; i++) {
        w_sub->J->g[i] = 0;
    }*/
    for (i = 0; i < 2 * w->nec; i++)
    {
        w_sub->J->g[w->nic + w->n - 2 * w->nec + i] = w_sub->scale[1 + w->nc + w->n - 2 * w->nec + i]; // w_sub->scale[0];
    }
    for (i = 0; i < w->nec; i++)
    {
        w_sub->J->a[w_sub->J->a_row * (w->nic + w->n - 2 * w->nec) + i + i * w_sub->J->a_row] = -w_sub->scale[1 + w->nc + w->n - 2 * w->nec + i] / w_sub->scale[1 + i];
    }
    for (i = w->nec; i < 2 * w->nec; i++)
    {
        w_sub->J->a[w_sub->J->a_row * (w->nic + w->n - 2 * w->nec + i) + i - w->nec] = w_sub->scale[1 + w->nc + w->n - 2 * w->nec + i] / w_sub->scale[1 + i - w->nec];
    }
    return 0;
}

zeronp_int calculate_Jacob_first(
    SUBNPWork *w_sub,
    ZERONPWork *w,
    ZERONPSettings *stgs)
{
    zeronp_int i;
    zeronp_float *grad = (zeronp_float *)zeronp_malloc((w->nc + 1) * w->n * sizeof(zeronp_float));

    // calculate_scaled_grad(grad, w->p + w->nic, w_sub->scale, stgs, w);
    if (stgs->rs)
    {
        // use random sampling to calcualte gradient
        calculate_scaled_grad_random(grad, w->p + w->nic, w->ob, w_sub->scale, stgs, w);
    }
    else
    {
        // use coordinate-wise finite difference to calculate gradient
        calculate_scaled_grad(grad, w->p + w->nic, w_sub->scale, stgs, w);
    }
    if (w->exit == 1)
    {
        return 0;
    }
    for (i = 0; i < w->nic; i++)
    {
        w_sub->J->anew[i + (i + w->nec) * w_sub->npic] = -1.;
        w_sub->J->a[i + w->nec + i * w->nc] = -1;
        w_sub->J->g[i] = 0;
    }
    for (i = 0; i < w->nc; i++)
    {
        memcpy(w_sub->J->anew + w->nic + i * w_sub->npic, grad + (i + 1) * w->n, w->n * sizeof(zeronp_float));
    }

    ZERONP(transpose)
    (w->n, w->nc, grad + w->n, w_sub->J->a + w->nc * w->nic);
    memcpy(w_sub->J->g + w->nic, grad, w->n * sizeof(zeronp_float));

    calculate_alm_criterion(w, w_sub, grad);
    zeronp_free(grad);
}

zeronp_int calculate_Jacob_first_rescue(
    SUBNPWork *w_sub,
    ZERONPWork *w,
    ZERONPSettings *stgs)
{
    zeronp_int i;
    if (stgs->rescue)
    {
        w->n -= 2 * w->nec;
    }
    calculate_Jacob_first(w_sub, w, stgs);
    if (stgs->rescue)
    {
        w->n += 2 * w->nec;
    }
    /*
    for (i = 0; i < w->n - 2*w->nec; i++) {
        w_sub->J->g[i] = 0;
    }*/
    for (i = 0; i < 2 * w->nec; i++)
    {
        w_sub->J->g[w->nic + w->n - 2 * w->nec + i] = w_sub->scale[1 + w->nc + w->n - 2 * w->nec + i]; // w_sub->scale[0];
    }
    for (i = 0; i < w->nec; i++)
    {
        w_sub->J->a[w_sub->J->a_row * (w->nic + w->n - 2 * w->nec) + i + i * w_sub->J->a_row] = -w_sub->scale[1 + w->nc + w->n - 2 * w->nec + i] / w_sub->scale[1 + i];
    }
    for (i = w->nec; i < 2 * w->nec; i++)
    {
        w_sub->J->a[w_sub->J->a_row * (w->nic + w->n - 2 * w->nec + i) + i - w->nec] = w_sub->scale[1 + w->nc + w->n - 2 * w->nec + i] / w_sub->scale[1 + i - w->nec];
    }
    return 0;
}

zeronp_float calculate_ALM(
    ZERONPCost *ob,
    ZERONPSettings *stgs,
    zeronp_float *p,
    const ZERONPWork *w,
    const SUBNPWork *w_sub)
{

    zeronp_float result = ob->obj;
    if (w->nc)
    {
        zeronp_float *ap = (zeronp_float *)zeronp_malloc(w_sub->J->a_row * sizeof(zeronp_float));
        ZERONP(Ax)
        (ap, w_sub->J->a, p, w_sub->J->a_row, w_sub->npic);
        if (w->nic)
        {
            zeronp_float *ic = (zeronp_float *)zeronp_malloc(w->nic * sizeof(zeronp_float));
            memcpy(ic, ob->ic, w->nic * sizeof(zeronp_float));
            ZERONP(add_scaled_array)
            (ic, p, w->nic, -1.0);
            if (w->n > w->nec || stgs->rescue)
            {
                ZERONP(add_scaled_array)
                (ic, ap + w->nec, w->nic, -1.0);
                ZERONP(add_scaled_array)
                (ic, w_sub->b + w->nec, w->nic, 1.0);
            }
            result = result + w->rho * ZERONP(norm_sq)(ic, w->nic);
            result = result - ZERONP(dot)(w->l + w->nec, ic, w->nic);
            zeronp_free(ic);
        }
        if (w->nec)
        {
            zeronp_float *ec = (zeronp_float *)zeronp_malloc(w->nec * sizeof(zeronp_float));
            memcpy(ec, ob->ec, w->nec * sizeof(zeronp_float));
            if (w->n > w->nec || stgs->rescue)
            {
                ZERONP(add_scaled_array)
                (ec, ap, w->nec, -1.0);
                ZERONP(add_scaled_array)
                (ec, w_sub->b, w->nec, 1.0);
            }
            result = result + w->rho * ZERONP(norm_sq)(ec, w->nec);
            result = result - ZERONP(dot)(w->l, ec, w->nec);
            zeronp_free(ec);
        }
        zeronp_free(ap);
    }
    return result;
}
/*
zeronp_float calculate_ALM_rescue
(
    ZERONPCost* ob,
    ZERONPSettings* stgs,
    zeronp_float* p,
    const ZERONPWork* w,
    const SUBNPWork* w_sub
) {

    zeronp_int i;
    zeronp_float result = 0;
    for (i = 0; i < w->nec; i++) {
        result += (w_sub->scale[1 + w->nc + w->n - w->nec + i] * w->p[w->n - w->nec + i]) * (w_sub->scale[1 + w->nc + w->n - w->nec + i] * w->p[w->n - w->nec + i]);
    }
    result /= w_sub->scale[0];
    if (w->nc) {
        zeronp_float* ap = (zeronp_float*)zeronp_malloc(w_sub->J->a_row * sizeof(zeronp_float));
        ZERONP(Ax)(ap, w_sub->J->a, p, w_sub->J->a_row, w_sub->npic);
        if (w->nic) {
            zeronp_float* ic = (zeronp_float*)zeronp_malloc(w->nic * sizeof(zeronp_float));
            memcpy(ic, ob->ic, w->nic * sizeof(zeronp_float));
            ZERONP(add_scaled_array)(ic, p, w->nic, -1.0);
            ZERONP(add_scaled_array)(ic, ap + w->nec, w->nic, -1.0);
            ZERONP(add_scaled_array)(ic, w_sub->b + w->nec, w->nic, 1.0);
            result = result + w->rho * ZERONP(norm_sq)(ic, w->nic);
            result = result - ZERONP(dot)(w->l + w->nec, ic, w->nic);
            zeronp_free(ic);
        }
        if (w->nec) {
            zeronp_float* ec = (zeronp_float*)zeronp_malloc(w->nec * sizeof(zeronp_float));
            memcpy(ec, ob->ec, w->nec * sizeof(zeronp_float));
            ZERONP(add_scaled_array)(ec, ap, w->nec, -1.0);
            ZERONP(add_scaled_array)(ec, w_sub->b, w->nec, 1.0);
            result = result + w->rho * ZERONP(norm_sq)(ec, w->nec);
            result = result - ZERONP(dot)(w->l, ec, w->nec);
            zeronp_free(ec);
        }
        zeronp_free(ap);
    }
    return result;
}*/
zeronp_int calculate_ALMgradient_zero(
    SUBNPWork *w_sub,
    ZERONPWork *w,
    ZERONPSettings *stgs,
    zeronp_float *g,
    zeronp_float j)
{   
    // Calculate the gradient of Augmented Lagrangian Function

    ZERONPCost **obm = (ZERONPCost **)zeronp_malloc(1 * sizeof(ZERONPCost *));
    ZERONPCost** obm_backward = (ZERONPCost**)zeronp_malloc(1 * sizeof(ZERONPCost*));
    zeronp_int i, tag;
    tag = 1;
    zeronp_int len = stgs->rescue ? w->n - 2 * w->nec : w->n;
    zeronp_float alm,alm_backward;
    zeronp_float infeas, temp;
    zeronp_float *contemp = (zeronp_float *)zeronp_malloc(w->nc * sizeof(zeronp_float));
    zeronp_float *p = (zeronp_float *)zeronp_malloc(1 * w->n * sizeof(zeronp_float));
    memcpy(p, &w->p[w->nic], w->n * sizeof(zeronp_float));

    obm[0] = init_cost(w->nec, w->nic);
    obm_backward[0] = init_cost(w->nec, w->nic);


    for (i = 0; i < len; i++)
    {

        p[i] += stgs->delta;
        calculate_scaled_cost(obm, p, w_sub->scale, stgs, w, 1);

        // contemp and temp variable is used for calculate the projected alm gradient(as stop criterion)
       
        if (w->nec)
        {
            memcpy(contemp, obm[0]->ec, w->nec * sizeof(zeronp_float));
        }
        if (w->nic)
        {
            memcpy(&contemp[w->nec], obm[0]->ic, w->nic * sizeof(zeronp_float));
        }

        if (stgs->rescue == 0 && obm[0]->obj != w->ob->obj)
        {
            tag = 0;
        }

        // calculate the ALM function value
        alm = obm[0]->obj;
        zeronp_float *ptemp = (zeronp_float *)zeronp_malloc(w_sub->J->npic * sizeof(zeronp_float));
        memcpy(ptemp, w->p, w->nic * sizeof(zeronp_float));
        memcpy(&ptemp[w->nic], p, w->n * sizeof(zeronp_float));
        if (w_sub->nc > 0)
        {
            alm = calculate_ALM(obm[0], stgs, ptemp, w, w_sub);
        }
        zeronp_free(ptemp);

        // calculate the infeasibility
        infeas = calculate_infeas_scaledob(obm[0], w, p, w_sub->scale);

        {
            // Record the best point
            if (infeas <= stgs->tol_con && MIN(p[i] - w->pb->pl[i], w->pb->pu[i] - p[i]) > 0 && w_sub->ob_cand->obj > obm[0]->obj)
            {
                memcpy(w_sub->p_cand, w->p, w->nic * sizeof(zeronp_float));
                memcpy(&w_sub->p_cand[w->nic], p, w->n * sizeof(zeronp_float));
                copyZERONPCost(w_sub->ob_cand, obm[0]);
            }

            // calculate gradient approximately
            if (stgs->cen_diff) {
                //use centre difference to calculate the gradient
                p[i] -= 2 * stgs->delta;
                calculate_scaled_cost(obm_backward, p, w_sub->scale, stgs, w, 1);
                alm_backward = obm_backward[0]->obj;
                zeronp_float* ptemp = (zeronp_float*)zeronp_malloc(w_sub->J->npic * sizeof(zeronp_float));
                memcpy(ptemp, w->p, w->nic * sizeof(zeronp_float));
                memcpy(&ptemp[w->nic], p, w->n * sizeof(zeronp_float));
                if (w_sub->nc > 0)
                {
                    alm_backward = calculate_ALM(obm[0], stgs, ptemp, w, w_sub);
                }
                zeronp_free(ptemp);

                g[w->nic + i] = (alm - alm_backward) / (2*stgs->delta);

                p[i] += 2 * stgs->delta;
            }
            else {
                g[w->nic + i] = (alm - j) / stgs->delta;
            }
        }
        /*
        if (w->exit == 1) {
            zeronp_free(contemp);
            break;
        }*/
        p[i] -= stgs->delta;

        temp = w->p[w->nic + i] * w_sub->scale[w->nc + 1 + i] - ((obm[0]->obj - w->ob->obj) / stgs->delta) * w_sub->scale[0] / w_sub->scale[w->nc + 1 + i];
        if (w->nc > 0)
        {
            if (w->nec)
            {
                ZERONP(add_scaled_array)
                (contemp, w->ob->ec, w->nec, -1.0);
            }
            if (w->nic)
            {
                ZERONP(add_scaled_array)
                (&contemp[w->nec], w->ob->ic, w->nic, -1.0);
            }
            ZERONP(scale_array)
            (contemp, 1 / stgs->delta, w->nc);
            temp += w_sub->scale[0] * ZERONP(dot)(contemp, w->l, w->nc) / w_sub->scale[w->nc + 1 + i];
        }
        if (temp > w->pb->pu[i] * w_sub->scale[w->nc + 1 + i])
        {
            temp = w->pb->pu[i] * w_sub->scale[w->nc + 1 + i];
        }
        else if (temp < w->pb->pl[i] * w_sub->scale[w->nc + 1 + i])
        {
            temp = w->pb->pl[i] * w_sub->scale[w->nc + 1 + i];
        }

        {
            w->alm_crit += (temp - w->p[w->nic + i] * w_sub->scale[w->nc + 1 + i]) * (temp - w->p[w->nic + i] * w_sub->scale[w->nc + 1 + i]);
        }
    }
    // if (w->nic > 0.5) {
    //     for (k = 0; k < w->nic; k++) {
    //         g[k] = 0;
    //     }
    // }

    if (tag)
    {
        w->const_time++;
    }

    zeronp_free(p);
    zeronp_free(contemp);
    free_cost(obm[0]);
    free_cost(obm_backward[0]);
    zeronp_free(obm);
    zeronp_free(obm_backward);
    return 0;
}

zeronp_int calculate_ALMgradient_zero_rescue(
    SUBNPWork *w_sub,
    ZERONPWork *w,
    ZERONPSettings *stgs,
    zeronp_float *g,
    zeronp_float j)
{
    calculate_ALMgradient_zero(w_sub, w, stgs, g, j);
    for (zeronp_int i = w->n - 2 * w->nec; i < w->n - 1 * w->nec; i++)
    {
        w_sub->J->g[w->nic + i] = w->pen_l1 * w_sub->scale[1 + w->nc + i] + w->l[i - (w->n - 2 * w->nec)] * w_sub->scale[1 + w->nc + i] / w_sub->scale[1 + i - (w->n - 2 * w->nec)] - 2 * w->rho * w->ob->ec[i - (w->n - 2 * w->nec)] * w_sub->scale[1 + w->nc + i] / w_sub->scale[1 + i - (w->n - 2 * w->nec)];
    }
    for (zeronp_int i = w->n - 1 * w->nec; i < w->n; i++)
    {
        w_sub->J->g[w->nic + i] = w->pen_l1 * w_sub->scale[1 + w->nc + i] - w->l[i - (w->n - w->nec)] * w_sub->scale[1 + w->nc + i] / w_sub->scale[1 + i - (w->n - w->nec)] + 2 * w->rho * w->ob->ec[i - (w->n - w->nec)] * w_sub->scale[1 + w->nc + i] / w_sub->scale[1 + i - (w->n - w->nec)];
    }
    return 0;
}

zeronp_int calculate_ALMgradient_first(
    SUBNPWork *w_sub,
    ZERONPWork *w,
    ZERONPSettings *stgs,
    zeronp_float *g)
{
    zeronp_int i;
    zeronp_float *grad = (zeronp_float *)zeronp_malloc((w->nc + 1) * w->n * sizeof(zeronp_float));
    zeronp_float *temp_r = (zeronp_float *)zeronp_malloc(w->nc * sizeof(zeronp_float));
    zeronp_float *temp_l = (zeronp_float *)zeronp_malloc(w->nc * w->n * sizeof(zeronp_float));

    if (stgs->rs)
    {
        calculate_scaled_grad_random(grad, w->p + w->nic, w->ob, w_sub->scale, stgs, w);
    }
    else
    {
        calculate_scaled_grad(grad, w->p + w->nic, w_sub->scale, stgs, w);
    }
    //
    if (w->exit == 1)
    {
        return 0;
    }

    for (i = 0; i < w->nic; i++)
    {
        g[i] = 0;
        w_sub->J->anew[i + (i + w->nec) * w_sub->npic] = -1.;
    }
    for (i = 0; i < w->nc; i++)
    {
        memcpy(w_sub->J->anew + w->nic + i * w_sub->npic, grad + (i + 1) * w->n, w->n * sizeof(zeronp_float));
    }

    ZERONP(transpose)
    (w->n, w->nc, grad + w->n, temp_l);
    memcpy(g + w->nic, grad, w->n * sizeof(zeronp_float));

    // calculate ALM stop criterion
    calculate_alm_criterion(w, w_sub, grad);
    zeronp_free(grad);

    if (w->nc > 0)
    {
        ZERONP(add_scaled_array)
        (temp_l, w_sub->J->a + w->nc * w->nic, w->nc * w->n, -1.);
        ZERONP(set_as_scaled_array)
        (temp_r, w->l, -1., w->nc);
        ZERONP(add_scaled_array)
        (temp_r, w->ob->ec, w->nec, w->rho);
        ZERONP(add_scaled_array)
        (temp_r + w->nec, w->ob->ic, w->nic, w->rho);
        ZERONP(add_scaled_array)
        (temp_r + w->nec, w->p, w->nic, -w->rho);
        zeronp_float *temp = (zeronp_float *)zeronp_malloc(w_sub->npic * sizeof(zeronp_float));
        ZERONP(AB)
        (temp, temp_r, temp_l, 1, w->nc, w->n);
        ZERONP(add_scaled_array)
        (g + w->nic, temp, w->n, 1.);
        zeronp_free(temp);
    }

    zeronp_free(temp_r);
    zeronp_free(temp_l);
}

zeronp_int calculate_ALMgradient_first_rescue(
    SUBNPWork *w_sub,
    ZERONPWork *w,
    ZERONPSettings *stgs,
    zeronp_float *g,
    zeronp_float j)
{
    if (stgs->rescue)
    {
        w->n -= 2 * w->nec;
    }
    calculate_ALMgradient_first(w_sub, w, stgs, g);
    if (stgs->rescue)
    {
        w->n += 2 * w->nec;
    }
    for (zeronp_int i = w->n - 2 * w->nec; i < w->n - 1 * w->nec; i++)
    {
        w_sub->J->g[w->nic + i] = w->pen_l1 * w_sub->scale[1 + w->nc + i] + w->l[i - (w->n - 2 * w->nec)] * w_sub->scale[1 + w->nc + i] / w_sub->scale[1 + i - (w->n - 2 * w->nec)] - 2 * w->rho * w->ob->ec[i - (w->n - 2 * w->nec)] * w_sub->scale[1 + w->nc + i] / w_sub->scale[1 + i - (w->n - 2 * w->nec)];
    }
    for (zeronp_int i = w->n - 1 * w->nec; i < w->n; i++)
    {
        w_sub->J->g[w->nic + i] = w->pen_l1 * w_sub->scale[1 + w->nc + i] - w->l[i - (w->n - w->nec)] * w_sub->scale[1 + w->nc + i] / w_sub->scale[1 + i - (w->n - w->nec)] + 2 * w->rho * w->ob->ec[i - (w->n - w->nec)] * w_sub->scale[1 + w->nc + i] / w_sub->scale[1 + i - (w->n - w->nec)];
    }
    return 0;
}

zeronp_int calculate_ALM_hess(
    SUBNPWork *w_sub,
    ZERONPWork *w,
    ZERONPSettings *stgs,
    zeronp_float *h)
{
    zeronp_int i, j, k;
    zeronp_float *hess = (zeronp_float *)zeronp_malloc(w->n * w->n * (w->nc + 1) * sizeof(zeronp_float));
    calculate_scaled_hess(hess, w->p + w->nic, w_sub->scale, stgs, w);

    for (k = 1; k < w->nc + 1; k++)
    {
        zeronp_float con_value;
        if (k - 1 < w->nec)
        {
            con_value = w->ob->ec[k - 1];
        }
        else
        {
            con_value = w->ob->ic[k - 1 - w->nec] - w->p[k - 1 - w->nec];
        }
        ZERONP(add_scaled_array)
        (hess, hess + k * w->n * w->n, w->n * w->n, w->rho * con_value - w->l[k - 1]);
    }

    for (i = 0; i < w_sub->npic; i++)
    {
        for (j = 0; j < w_sub->npic; j++)
        {
            if (i < w->nic || j < w->nic)
            {
                h[i + j * w_sub->npic] = 0;
            }
            else
            {
                h[i + j * w_sub->npic] = hess[(i - w->nic) + (j - w->nic) * w->n];
            }
        }
    }
    for (k = 0; k < w->nc; k++)
    {
        ZERONP(rank1update)
        (w_sub->npic, h, w->rho, w_sub->J->anew + k * w_sub->npic);
    }
    zeronp_free(hess);
}

void BFGSudpate(
    ZERONPWork *w,
    SUBNPWork *w_sub,
    ZERONPSettings *stgs,
    zeronp_float *g,
    zeronp_float *yg,
    zeronp_float *sx)
{
    ZERONP(add_scaled_array)
    (yg, g, w_sub->J->npic, -1.0);
    ZERONP(scale_array)
    (yg, -1.0, w_sub->J->npic);
    ZERONP(add_scaled_array)
    (sx, w->p, w_sub->J->npic, -1.0);
    ZERONP(scale_array)
    (sx, -1.0, w_sub->J->npic);
    zeronp_float sc[2];
    zeronp_float *temp = (zeronp_float *)zeronp_malloc(w_sub->J->npic * sizeof(zeronp_float));
    ZERONP(Ax)
    (temp, w->h, sx, w_sub->J->npic, w_sub->J->npic);
    sc[0] = ZERONP(dot)(sx, temp, w_sub->J->npic);
    sc[1] = ZERONP(dot)(sx, yg, w_sub->J->npic);
    if (sc[0] * sc[1] > 0)
    {
        memcpy(sx, temp, w_sub->J->npic * sizeof(zeronp_float));
        // two rank 1 updates
        ZERONP(rank1update)
        (w_sub->J->npic, w->h, -1 / sc[0], sx);
        ZERONP(rank1update)
        (w_sub->J->npic, w->h, 1 / sc[1], yg);
    }
    zeronp_free(temp);
}

// void LBFGS_H_inv_v
//(
//     zeronp_int m,
//     zeronp_int n,
//     zeronp_int start_index,
//     zeronp_float* V,
//     zeronp_float* Y,
//     zeronp_float* S
//) {
//     // This subrontine calculate the H^{-1} vector product given the past data
// }

zeronp_float fun_along_2d(
    ZERONPWork *w,
    ZERONPSettings *stgs,
    SUBNPWork *w_sub,
    zeronp_float *d1,
    zeronp_float *d2,
    zeronp_float *x,
    zeronp_float *coeff)
{
    // This subroutine calculate the function value along two direction
    zeronp_float *p = (zeronp_float *)zeronp_malloc(w->n * sizeof(zeronp_float));
    zeronp_float val;

    // calculate the target point
    memcpy(p, x, w->n * sizeof(zeronp_float));
    ZERONP(add_scaled_array)
    (p, d1, w->n, coeff[0]);
    ZERONP(add_scaled_array)
    (p, d2, w->n, coeff[1]);

    // calculate the value at p
    ZERONPCost *Ob = init_cost(w->nec, w->nic);
    calculate_scaled_cost(&Ob, p, w_sub->scale, stgs, w, 1);
    val = Ob->obj;

    free_cost(Ob);
    zeronp_free(p);
    return val;
}

zeronp_float *interpolate2d(
    ZERONPWork *w,
    ZERONPSettings *stgs,
    SUBNPWork *w_sub,
    zeronp_float radius,
    zeronp_float *d1,
    zeronp_float *d2,
    zeronp_float *x,
    zeronp_float val)
{
    // This subroutine do quadratic interpolation to the function.
    // Input:   two directions d1 and d2
    //          current point x
    //          current value val
    //          interpolation radius radius
    // Output:  linear coeffience g and quadratic coeffience Q, stored in the vector g_and_Q
    // REMEMBER TO FREE g_and_Q after using it!
    zeronp_float *g_and_Q = (zeronp_float *)zeronp_malloc(6 * sizeof(zeronp_float));
    zeronp_float *init_pt = (zeronp_float *)zeronp_malloc(10 * sizeof(zeronp_float));
    zeronp_int i, j, k;
    // Set up init_pt

    for (i = 0; i < 5; i++)
    {
        init_pt[2 * i] = cos(2 * i * PI / 5) * radius;
        init_pt[2 * i + 1] = sin(2 * i * PI / 5) * radius;
    }

    zeronp_float *coeff_matrix = (zeronp_float *)zeronp_malloc(25 * sizeof(zeronp_float));
    zeronp_float *fval = (zeronp_float *)zeronp_malloc(5 * sizeof(zeronp_float));

    // Set up the coeff matrix and favl
    for (i = 0; i < 5; i++)
    {
        coeff_matrix[i] = init_pt[2 * i];
        coeff_matrix[i + 5] = init_pt[2 * i + 1];
        fval[i] = fun_along_2d(w, stgs, w_sub, d1, d2, x, &init_pt[2 * i]) - val;
        for (j = 0; j < 2; j++)
        {
            for (k = 0; k <= j; k++)
            {
                if (k == j)
                {
                    coeff_matrix[i + 5 * (j * (j + 1) / 2 + k + 2)] = init_pt[2 * i + j] * init_pt[2 * i + k];
                }
                else
                {
                    coeff_matrix[i + 5 * (j * (j + 1) / 2 + k + 2)] = 2 * init_pt[2 * i + j] * init_pt[2 * i + k];
                }
            }
        }
    }

    // Solve linear equations coeff*x = fval
    ZERONP(solve_general_lin_sys)
    (5, coeff_matrix, fval);
    // Record the result
    memcpy(g_and_Q, fval, 2 * sizeof(zeronp_float));
    g_and_Q[2] = fval[2];
    g_and_Q[3] = fval[3];
    g_and_Q[4] = fval[3];
    g_and_Q[5] = fval[4];

    // Free variable and return
    zeronp_free(fval);
    zeronp_free(coeff_matrix);
    zeronp_free(init_pt);
    return g_and_Q;
}
