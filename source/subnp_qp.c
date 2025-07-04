#include "der_info.h"
#include "zeronp_util.h"
#include "qp_solver.h"

// zeronp_float qpsolver_time = 0;

void copyZERONPCost(ZERONPCost *ob1, ZERONPCost *ob2)
{
    ob1->obj = ob2->obj;
    if (ob2->nec)
    {
        memcpy(ob1->ec, ob2->ec, ob2->nec * sizeof(zeronp_float));
    }
    else
    {
        ob1->ec = ZERONP_NULL;
    }
    if (ob2->nic)
    {
        memcpy(ob1->ic, ob2->ic, ob2->nic * sizeof(zeronp_float));
    }
    else
    {
        ob1->ic = ZERONP_NULL;
    }
    // ob1->cost = ob2->cost;
}

ZERONPCost *init_cost(zeronp_int nec, zeronp_int nic)
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

    c->cost = ZERONP_NULL;

    c->grad = ZERONP_NULL;
    c->hess = ZERONP_NULL;

    return c;
}

SUBNPWork *init_work_subnp(
    ZERONPWork *w,
    ZERONPSettings *stgs)
{
    zeronp_int i, j;
    SUBNPWork *w_sub = (SUBNPWork *)zeronp_malloc(sizeof(SUBNPWork));

    // calloc alp
    w_sub->alp = (zeronp_float *)zeronp_calloc(3, sizeof(zeronp_float));

    // malloc scale
    w_sub->n_scale = 1 + w->nec + w->nic + w->n;
    w_sub->scale = (zeronp_float *)zeronp_malloc(w_sub->n_scale * sizeof(zeronp_float));
    // init ch
    w_sub->ch = 1;
    w_sub->ob_cand = init_cost(w->nec, w->nic);
    w_sub->ob_cand->obj = INFINITY;
    // init mm
    if (w->pb->Ipb[1] >= 0.5)
    {
        if (w->pb->Ipb[0] <= 0.5)
        {
            w_sub->mm = w->nic;
        }
        else
        {
            w_sub->mm = w->nic + w->n;
        }
    }
    else
    {
        w_sub->mm = 0;
    }

    // init nc and npic
    w_sub->nc = w->nc;
    w_sub->npic = w->nic + w->n;
    // malloc Jacob
    w_sub->J = (SUBNPJacob *)zeronp_malloc(sizeof(SUBNPJacob));
    w_sub->J->a_row = w_sub->nc - MAX(0, w->nec - w->n + 1);
    w_sub->J->n = w->n;
    w_sub->J->nec = w->nec;
    w_sub->J->nic = w->nic;
    w_sub->J->nc = w->nc;
    w_sub->J->npic = w->nic + w->n;
    w_sub->J->g = (zeronp_float *)zeronp_calloc(w_sub->J->npic, sizeof(zeronp_float));
    w_sub->J->a = (zeronp_float *)zeronp_calloc(w_sub->J->a_row * (w_sub->npic + 1), sizeof(zeronp_float));
    w_sub->p_cand = (zeronp_float *)zeronp_malloc((w->n + w->nic) * sizeof(zeronp_float));
    w_sub->J->anew = (zeronp_float *)zeronp_calloc(w_sub->nc * (w_sub->npic), sizeof(zeronp_float));

    // malloc b
    if (w->nc > 0.5)
    {
        w_sub->b = (zeronp_float *)zeronp_malloc(w_sub->J->a_row * sizeof(zeronp_float));
    }
    else
    {
        w_sub->b = ZERONP_NULL;
    }

    if (w->n <= w->nec)
    {
        w_sub->constr_index = (zeronp_int *)zeronp_malloc((w->n - 1) * sizeof(zeronp_int));
    }
    else
    {
        w_sub->constr_index = ZERONP_NULL;
    }

    return w_sub;
}

SUBNPWork *SUBNP(init)(
    ZERONPWork *w,
    ZERONPSettings *stgs)
{
    SUBNPWork *w_sub;
    w_sub = init_work_subnp(w, stgs);

    return w_sub;
}

zeronp_float *compute_scale(
    ZERONPWork *w,
    ZERONPSettings *stgs)
{
    zeronp_int i, n_scale = 1 + w->nec + w->nic + w->n;
    zeronp_float *scale = (zeronp_float *)zeronp_malloc(n_scale * sizeof(zeronp_float));

    // objective value and equality constraints
    if (w->nec > 0)
    {
        scale[0] = w->ob->obj;
        zeronp_float ec_norm_inf = ZERONP(norm_inf)(w->ob->ec, w->nec);
        if (ec_norm_inf <= stgs->tol_con)
        {
            ec_norm_inf = stgs->tol_con;
        }
        if (stgs->scale == 0)
        {
            ec_norm_inf = 1;
        }
        for (i = 0; i < w->nec; i++)
        {
            scale[1 + i] = ec_norm_inf;
        }
    }
    else
    {
        scale[0] = 1.;
    }

    // inequality constraints and bounds of decision variables
    if (w->pb->Ipb[1] <= 0 && stgs->scale)
    {
        memcpy(&(scale[1 + w->nec]), w->p, (w->nic + w->n) * sizeof(zeronp_float));
    }
    else
    {
        for (i = 0; i < (w->nic + w->n); i++)
        {
            scale[1 + w->nec + i] = 1.;
        }
    }

    // bound scale
    for (i = 0; i < n_scale; i++)
    {
        scale[i] = MIN(MAX(ABS(scale[i]), stgs->tol), SAFEDIV_POS(1., stgs->tol));
    }

    return scale;
}

zeronp_int rescale_ob(
    ZERONPCost *ob,
    const zeronp_float *scale)
{
    zeronp_int i;
    ob->obj /= scale[0];
    for (i = 0; i < ob->nec; i++)
    {
        ob->ec[i] /= scale[1 + i];
    }
    for (i = 0; i < ob->nic; i++)
    {
        ob->ic[i] /= scale[1 + ob->nec + i];
    }

    return 0;
}

zeronp_int unscale_ob(
    ZERONPCost *ob,
    const zeronp_float *scale)
{
    zeronp_int i;
    ob->obj *= scale[0];
    for (i = 0; i < ob->nec; i++)
    {
        ob->ec[i] *= scale[1 + i];
    }
    for (i = 0; i < ob->nic; i++)
    {
        ob->ic[i] *= scale[1 + ob->nec + i];
    }

    return 0;
}

zeronp_int rescale_p(
    zeronp_float *p,
    zeronp_float *scale,
    ZERONPWork *w)
{
    zeronp_int i;
    for (i = 0; i < (w->nic + w->n); i++)
    {
        p[i] /= scale[1 + w->nec + i];
    }

    return 0;
}

zeronp_int rescale(
    SUBNPWork *w_sub,
    ZERONPWork *w,
    zeronp_float *scale,
    ZERONPSettings *stgs)
{
    zeronp_int i, j;

    // rescale ob
    rescale_ob(w->ob, scale);

    // rescale p
    rescale_p(w->p, scale, w);

    // rescale pb
    if (w->pb->Ipb[1] >= 0.5)
    {
        for (i = 0; i < w->nic; i++)
        {
            w->pb->il[i] /= scale[1 + w->nec + i];
            w->pb->iu[i] /= scale[1 + w->nec + i];
        }
        if (w->pb->Ipb[0] > 0.5)
        {
            for (i = 0; i < w->n; i++)
            {
                w->pb->pl[i] /= scale[1 + w->nec + w->nic + i];
                w->pb->pu[i] /= scale[1 + w->nec + w->nic + i];
            }
        }
    }

    // rescale lagrangian multipliers
    if (w->nc > 0.5)
    {
        for (i = 0; i < w->nc; i++)
        {
            w->l[i] = scale[1 + i] * w->l[i] / scale[0];
        }
    }

    if (stgs->bfgs)
    {
        // rescale Hessian
        for (i = 0; i < w_sub->npic; i++)
        {
            for (j = 0; j <= i; j++)
            {
                w->h[i + j * w_sub->npic] = scale[1 + w->nec + i] * w->h[i + j * w_sub->npic] * scale[1 + w->nec + j] / scale[0];

                // use symmetric
                if (j < i)
                {
                    w->h[j + i * w_sub->npic] = w->h[i + j * w_sub->npic];
                }
            }
        }
    }

    // record scale in w_sub
    for (i = 0; i < w_sub->n_scale; i++)
    {
        w_sub->scale[i] *= scale[i];
    }

    return 0;
}

void unscale(
    SUBNPWork *w_sub,
    ZERONPWork *w,
    ZERONPSettings *stgs)
{
    zeronp_int i;
    for (i = 0; i < w_sub->n_scale; i++)
    {
        w_sub->scale[i] = 1 / w_sub->scale[i];
    }
    rescale(w_sub, w, w_sub->scale, stgs);
}

zeronp_int update_work_subnp(
    SUBNPWork *w_sub,
    ZERONPWork *w,
    ZERONPSettings *stgs)
{
    zeronp_int i;

    // init scale
    for (i = 0; i < w_sub->n_scale; i++)
    {
        w_sub->scale[i] = 1.;
    }

    // init Jacob
    for (i = 0; i < w->nic; i++)
    {
        // a[k+nec, k] = -1
        w_sub->J->a[i + w_sub->J->a_row - w->nic + i * w_sub->J->a_row] = -1.;
    }

    zeronp_float *scale = compute_scale(w, stgs);
    rescale(w_sub, w, scale, stgs);

    zeronp_free(scale);
    return 0;
}

zeronp_int linesearch(
    SUBNPWork *w_sub,
    ZERONPWork *w,
    ZERONPSettings *stgs,
    zeronp_float *j,
    zeronp_float *reduce,
    zeronp_float *p0,
    zeronp_float *sx)
{
    zeronp_int i, k, tag = 0;
    zeronp_float go = 1.0;
    zeronp_float obm, obn, alm_cand;
    zeronp_float old_penalty_value;
    ZERONPCost *ob1 = init_cost(w->nec, w->nic);
    ZERONPCost *ob2 = init_cost(w->nec, w->nic);
    ZERONPCost *ob3 = init_cost(w->nec, w->nic);
    ZERONPCost **obtmp;
    zeronp_float **pt = (zeronp_float **)zeronp_malloc(3 * sizeof(zeronp_float *));

    ////Try a different merit function
    // zeronp_float* l = (zeronp_float*)zeronp_malloc(w->nc * sizeof(zeronp_float));
    // memcpy(l, w->l, w->nc * sizeof(zeronp_float));
    // ZERONP(set_as_scaled_array)(w->l, w->l, 0, w->nc);

    for (i = 0; i < 3; i++)
    {
        pt[i] = (zeronp_float *)zeronp_malloc(w_sub->J->npic * sizeof(zeronp_float));
    }
    zeronp_float sob[3];
    copyZERONPCost(ob1, w->ob);
    copyZERONPCost(ob2, w->ob);
    obtmp = &ob3;
    calculate_scaled_cost(obtmp, &p0[w->nic], w_sub->scale, stgs, w, 1);
    /* if (isnan(ob3->obj)) {
         return;
     }*/
    memcpy(pt[0], w->p, w_sub->J->npic * sizeof(zeronp_float));
    memcpy(pt[1], w->p, w_sub->J->npic * sizeof(zeronp_float));
    memcpy(pt[2], p0, w_sub->J->npic * sizeof(zeronp_float));
    old_penalty_value = calculate_ALM(ob1, stgs, w->p, w, w_sub);
    sob[0] = old_penalty_value;
    sob[1] = sob[0];
    sob[2] = calculate_ALM(ob3, stgs, p0, w, w_sub); // line_search_merit(ob3, w, w_sub, p0);// need to modify
    w_sub->alp[0] = 0.0;
    w_sub->alp[2] = 1.0;
    zeronp_int lstime = 0;
    if (stgs->ls_way == 1)
    {
        while (go > stgs->tol && lstime < stgs->ls_time)
        {
            if (ZERONP(min)(sob, 3) < *j)
            {
                break;
            }
            if (w->exit == 1)
            {
                break;
            }
            lstime++;
            memcpy(pt[1], p0, w_sub->J->npic * sizeof(zeronp_float));
            w_sub->alp[1] = (w_sub->alp[0] + w_sub->alp[2]) / 2.0;
            ZERONP(scale_array)
            (pt[1], w_sub->alp[1], w_sub->npic);
            ZERONP(add_scaled_array)
            (pt[1], w->p, w_sub->npic, (1 - w_sub->alp[1]));
            obtmp = &ob2;
            calculate_scaled_cost(obtmp, pt[1] + w->nic, w_sub->scale, stgs, w, 1);
            sob[1] = calculate_ALM(ob2, stgs, pt[1], w, w_sub); // line_search_merit(ob2, w, w_sub, pt[1]);
            obm = ZERONP(max)(sob, 3);
            if (obm < *j)
            {
                obn = ZERONP(min)(sob, 3);
                go = stgs->tol * (obm - obn) / (*j - obm);
            }
            if (sob[1] >= sob[0])
            {
                sob[2] = sob[1];
                copyZERONPCost(ob3, ob2);
                w_sub->alp[2] = w_sub->alp[1];
                memcpy(pt[2], pt[1], w_sub->J->npic * sizeof(zeronp_float));
            }
            else if (sob[0] <= sob[2])
            {
                sob[2] = sob[1];
                copyZERONPCost(ob3, ob2);
                w_sub->alp[2] = w_sub->alp[1];
                memcpy(pt[2], pt[1], w_sub->J->npic * sizeof(zeronp_float));
            }
            else
            {
                sob[0] = sob[1];
                copyZERONPCost(ob1, ob2);
                w_sub->alp[0] = w_sub->alp[1];
                memcpy(pt[0], pt[1], w_sub->J->npic * sizeof(zeronp_float));
            }
            if (go >= stgs->tol)
            {
                go = w_sub->alp[2] - w_sub->alp[0];
            }
        }
    }
    else if (stgs->ls_way == 2)
    {
        zeronp_float *DT_D = (zeronp_float *)zeronp_malloc(9 * sizeof(zeronp_float));

        DT_D[0] = 1;
        DT_D[1] = -3;
        DT_D[2] = 2;
        DT_D[3] = -3;
        DT_D[4] = 26;
        DT_D[5] = -24;
        DT_D[6] = 2;
        DT_D[7] = -24;
        DT_D[8] = 24;

        zeronp_float *qcoeff = (zeronp_float *)zeronp_calloc(3, sizeof(zeronp_float));
        // calculate middle point
        while (ZERONP(min)(sob, 3) >= *j && lstime < stgs->ls_time)
        {
            /*
            if (lstime == 0) {
                zeronp_float* A = (zeronp_float*)zeronp_malloc(9 * sizeof(zeronp_float));
                A[0] = 1;  A[1] = 0; A[2] = 0;
                A[3] = 1.;  A[4] = 1./2; A[5] = 1./4;
                A[6] = 1.;  A[7] = 1.; A[8] = 1.;
                w_sub->alp[1] = 1/2;
                memcpy(pt[1], p0, w_sub->J->npic * sizeof(zeronp_float));
                w_sub->alp[1] = (w_sub->alp[0] + w_sub->alp[2]) / 2.0;
                ZERONP(scale_array)(pt[1], w_sub->alp[1], w_sub->npic);
                ZERONP(add_scaled_array)(pt[1], w->p, w_sub->npic, (1 - w_sub->alp[1]));
                obtmp = &ob2;
                calculate_scaled_cost(obtmp, pt[1] + w->nic, w_sub->scale, stgs, w, 1);
                sob[1] = calculate_ALM(ob2, stgs, pt[1], w, w_sub);
                zeronp_float* temp = (zeronp_float*)zeronp_malloc(3 * sizeof(zeronp_float));
                ZERONP(Ax)(qcoeff, A, sob, 3, 3);
                ZERONP(Ax)(temp, DT_D, qcoeff, 3, 3);
                memcpy(qcoeff, temp, 3 * sizeof(zeronp_float));
                lstime++;
                zeronp_free(A);
                zeronp_free(temp);
            }
            else {
                if (qcoeff[2] > 0 && qcoeff[1] < 0) {
                    w_sub->alp[1] = -qcoeff[1] / (2 * qcoeff[2]);
                }else{
                    w_sub->alp[1] = (w_sub->alp[1] + w_sub->alp[0]) / 2;
                }

                lstime++;
                memcpy(pt[1], p0, w_sub->J->npic * sizeof(zeronp_float));
                ZERONP(scale_array)(pt[1], w_sub->alp[1], w_sub->npic);
                ZERONP(add_scaled_array)(pt[1], w->p, w_sub->npic, (1 - w_sub->alp[1]));
                obtmp = &ob2;
                calculate_scaled_cost(obtmp, pt[1] + w->nic, w_sub->scale, stgs, w, 1);
                sob[1] = calculate_ALM(ob2, stgs, pt[1], w, w_sub);
                zeronp_float* new_d = (zeronp_float*)zeronp_malloc(3 * sizeof(zeronp_float));
                new_d[0] = 1; new_d[1] = w_sub->alp[1]; new_d[2] = w_sub->alp[1] * w_sub->alp[1];
                zeronp_float* DT_D_newd = (zeronp_float*)zeronp_malloc(3 * sizeof(zeronp_float));
                ZERONP(Ax)(DT_D_newd, DT_D, new_d, 3, 3);
                zeronp_float newd_qcoeff, newd_DT_D_newd;
                newd_qcoeff = ZERONP(dot)(new_d, qcoeff, 3);
                newd_DT_D_newd = ZERONP(dot)(new_d, DT_D_newd, 3);
                ZERONP(add_scaled_array)(qcoeff, DT_D_newd, 3, (sob[1] - newd_qcoeff) / (1 + newd_DT_D_newd));
                // rank one update
                ZERONP(rank1update)(3, DT_D, -1 / (1 + newd_DT_D_newd), DT_D_newd);
                zeronp_free(new_d);
                zeronp_free(DT_D_newd);
            }*/
            lstime++;
            if (qcoeff[2] > 0 && qcoeff[1] < 0)
            {
                w_sub->alp[1] = -qcoeff[1] / (2 * qcoeff[2]);
            }
            else
            {
                w_sub->alp[1] = (w_sub->alp[0] + w_sub->alp[2]) / 2;
            }
            memcpy(pt[1], p0, w_sub->J->npic * sizeof(zeronp_float));
            ZERONP(scale_array)
            (pt[1], w_sub->alp[1], w_sub->npic);
            ZERONP(add_scaled_array)
            (pt[1], w->p, w_sub->npic, (1 - w_sub->alp[1]));
            obtmp = &ob2;
            calculate_scaled_cost(obtmp, pt[1] + w->nic, w_sub->scale, stgs, w, 1);
            sob[1] = calculate_ALM(ob2, stgs, pt[1], w, w_sub);

            zeronp_float *inpt = (zeronp_float *)zeronp_malloc(9 * sizeof(zeronp_float));

            inpt[0] = 1.;
            inpt[3] = w_sub->alp[0];
            inpt[6] = w_sub->alp[0] * w_sub->alp[0];
            inpt[1] = 1.;
            inpt[4] = w_sub->alp[1];
            inpt[7] = w_sub->alp[1] * w_sub->alp[1];
            inpt[2] = 1.;
            inpt[5] = w_sub->alp[2];
            inpt[8] = w_sub->alp[2] * w_sub->alp[2];

            memcpy(qcoeff, sob, 3 * sizeof(zeronp_float));
            ZERONP(solve_general_lin_sys)
            (3, inpt, qcoeff);
            zeronp_free(inpt);

            zeronp_float alpnew = -qcoeff[1] / (2 * qcoeff[2]);
            if (alpnew > w_sub->alp[1])
            {
                sob[2] = sob[1];
                copyZERONPCost(ob3, ob2);
                w_sub->alp[2] = w_sub->alp[1];
                memcpy(pt[2], pt[1], w_sub->J->npic * sizeof(zeronp_float));
            }
            else
            {
                sob[0] = sob[1];
                copyZERONPCost(ob1, ob2);
                w_sub->alp[0] = w_sub->alp[1];
                memcpy(pt[0], pt[1], w_sub->J->npic * sizeof(zeronp_float));
            }
        }
        zeronp_free(DT_D);
        zeronp_free(qcoeff);
    }
    // zeronp_printf("Line search time: %d\n", lstime);
    memcpy(sx, w->p, w_sub->J->npic * sizeof(zeronp_float));
    w_sub->ch = 1;
    obn = ZERONP(min)(sob, 3);
    if (w_sub->ob_cand->obj != INFINITY)
    {
        alm_cand = calculate_ALM(w_sub->ob_cand, stgs, w_sub->p_cand, w, w_sub); // line_search_merit(w_sub->ob_cand, w, w_sub, w_sub->p_cand);
        obn = MIN(obn, alm_cand);
    }
    if (*j <= obn)
    {
        tag = 1;
    }
    *reduce = (old_penalty_value - obn) / MAX(1, fabs(*j));
    if (stgs->noise == 1 || stgs->noise == 3)
    {
        // change gradient calculation step size
        if (*reduce > stgs->c_i * stgs->delta)
        {
            stgs->delta = stgs->delta * stgs->k_i;
        }
        if (*reduce < MAX(stgs->tol, stgs->c_r * stgs->delta))
        {
            stgs->delta = MAX(stgs->delta_end, stgs->delta / stgs->k_r);
            tag = 1;
        }
    }
    else
    {
        if (*reduce < stgs->tol)
        {
            tag = 1;
        }
    }

    if (sob[0] < sob[1])
    {
        //*j = calculate_ALM(ob1, stgs, pt[0], w, w_sub);;
        memcpy(w->p, pt[0], w_sub->J->npic * sizeof(zeronp_float));
        copyZERONPCost(w->ob, ob1);
    }
    else if (sob[2] < sob[1])
    {
        //*j = calculate_ALM(ob3, stgs, pt[2], w, w_sub);;
        memcpy(w->p, pt[2], w_sub->J->npic * sizeof(zeronp_float));
        copyZERONPCost(w->ob, ob3);
    }
    else
    {
        //*j = calculate_ALM(ob2, stgs, pt[1], w, w_sub);
        memcpy(w->p, pt[1], w_sub->J->npic * sizeof(zeronp_float));
        copyZERONPCost(w->ob, ob2);
    }
    if (w->ob->obj > w_sub->ob_cand->obj && stgs->noise != 0)
    {
        //*j = calculate_ALM(w_sub->ob_cand, stgs, w_sub->p_cand, w, w_sub);
        w_sub->ch = 1;
        copyZERONPCost(w->ob, w_sub->ob_cand);
        memcpy(w->p, w_sub->p_cand, w_sub->npic * sizeof(zeronp_float));
    }
    // recover lagrangian multiplier
    /*memcpy(w->l, l, w->nc * sizeof(zeronp_float));*/
    // zeronp_free(l);

    free_cost(ob1);
    free_cost(ob2);
    free_cost(ob3);

    // zeronp_printf("free P\n");
    // for (i = 0; i < 3; i++)
    // {
    //     zeronp_printf("freeing P[%d]\n", i);
    //     zeronp_free(pt[i]);
    // }
    zeronp_free(pt);

    return tag;
}

zeronp_int SUBNP(solve)(
    SUBNPWork *w_sub,
    ZERONPWork *w,
    ZERONPSettings *stgs,
    ZERONPInfo *info)
{
    zeronp_int minit = 0;
    /* scale procedure */
    update_work_subnp(w_sub, w, stgs);

    /* calculate Jacobian matrices */
    if (w->nec >= w->n)
    {
        max_kelement(w->ob->ec, w->nec, w->n - 1, w_sub->constr_index);
    }

    if (w->nc > 0.5)
    {
        if (stgs->grad)
        {
            stgs->rescue ? calculate_Jacob_first_rescue(w_sub, w, stgs) : calculate_Jacob_first(w_sub, w, stgs);
        }
        else
        {
            w->alm_crit = calculate_almcrit_iq(w, w_sub->scale, w->p);
            stgs->rescue ? calculate_Jacob_zero_rescue(w_sub, w, stgs) : calculate_Jacob_zero(w_sub, w, stgs);
            w->alm_crit = SQRTF(w->alm_crit);
        }

        if (w->exit == 1)
        {
            unscale(w_sub, w, stgs);
            return 0;
        }
        // record constraint value into last column of A
        if (w->nec >= w->n)
        {
            for (zeronp_int i = 0; i < w->n - 1; i++)
            {
                w_sub->J->a[w_sub->npic * w_sub->J->a_row + i] = w->ob->ec[w_sub->constr_index[i]];
            }
        }
        else
        {
            memcpy(&w_sub->J->a[w_sub->npic * w_sub->J->a_row], w->ob->ec, (w_sub->J->a_row - w->nic) * sizeof(zeronp_float));
        }

        if (w->nic > 0.5)
        {
            memcpy(&w_sub->J->a[w_sub->J->a_row - w->nic + w_sub->npic * w_sub->J->a_row], w->ob->ic, w->nic * sizeof(zeronp_float));
            ZERONP(add_scaled_array)
            (&w_sub->J->a[w_sub->J->a_row - w->nic + w_sub->npic * w_sub->J->a_row], w->p, w->nic, -1.0);
        }
        ZERONP(Ax)
        (w_sub->b, w_sub->J->a, w->p, w_sub->J->a_row, w_sub->npic);
        ZERONP(add_scaled_array)
        (w_sub->b, &w_sub->J->a[w_sub->npic * w_sub->J->a_row], w_sub->J->a_row, -1.0);
    }
    // corrspond to subnp line 73
    /* find interior (near-)feasible solution */
    if (w->nc > 0.5)
    {
        if (stgs->qpsolver == 1)
        {
            find_int_feas_sol_aff(w_sub, w, stgs, info);
        }
        else
        {
            find_int_feas_sol_osqp(w_sub, w, stgs, info);
        }
    }
    // recalculate cost if find new solution p
    // subnp.m line 139
    zeronp_float j;
    if (w_sub->ch > 0)
    {
        ZERONPCost **obtmp = &w->ob;
        calculate_scaled_cost(obtmp, w->p + w->nic, w_sub->scale, stgs, w, 1);
    }
    j = calculate_ALM(w->ob, stgs, w->p, w, w_sub);
    if (w->exit == 1)
    {
        unscale(w_sub, w, stgs);
        return 0;
    }
    // solve su bproblem QP and BFGS update Hessian
    // TODO: subnp.m line start at 152
    zeronp_int tag;
    zeronp_float *yg = (zeronp_float *)zeronp_malloc(w_sub->J->npic * sizeof(zeronp_float));
    zeronp_float *sx = (zeronp_float *)zeronp_malloc(w_sub->J->npic * sizeof(zeronp_float));
    zeronp_float *p0 = (zeronp_float *)zeronp_malloc(w_sub->J->npic * sizeof(zeronp_float));
    zeronp_float *y = (zeronp_float *)zeronp_calloc(w->nc, sizeof(zeronp_float));
    zeronp_float reduce;
    memcpy(p0, w->p, w_sub->J->npic * sizeof(zeronp_float));
    if (stgs->noise == 2)
    {
        // Use sampling to decide delta
        zeronp_float delta = calculate_delta(7, j, w->p, w_sub, w, stgs);
        if (delta > 0)
        {
            stgs->delta = delta;
        }
    }
    while (minit < stgs->min_iter)
    {
        if (minit > 0)
        {
            w_sub->ob_cand->obj = INFINITY;
        }
        minit++;
        if (w_sub->ch > 0)
        {
            // calculate the gradient

            if (stgs->grad)
            {
                // calculate gradient using given gradient function
                // Note that for convenient, we put random sampling here
                stgs->rescue ? calculate_ALMgradient_first_rescue(w_sub, w, stgs, w_sub->J->g, j) : calculate_ALMgradient_first(w_sub, w, stgs, w_sub->J->g);
            }
            else
            {
                // rescue == 1 means we add l1 penalty for the problem
                w->alm_crit = calculate_almcrit_iq(w, w_sub->scale, w->p);
                stgs->rescue ? calculate_ALMgradient_zero_rescue(w_sub, w, stgs, w_sub->J->g, j) : calculate_ALMgradient_zero(w_sub, w, stgs, w_sub->J->g, j);
                w->alm_crit = SQRTF(w->alm_crit);
            }

            if (w->exit == 1)
            {
                break;
            }
        }

        if (stgs->hess)
        {
            calculate_ALM_hess(w_sub, w, stgs, w->h);
        }
        else if (minit > 1 && stgs->bfgs)
        {
            // BFGS update
            BFGSudpate(w, w_sub, stgs, w_sub->J->g, yg, sx);
        }

        // solve QP subproblem
        zeronp_float mf;
        if (stgs->qpsolver == 1)
        {
            if (stgs->hess)
            {
                solve_qp_aff(w, w_sub, p0, y, w_sub->J->g, 2, info);
            }
            else
            {
                if (stgs->bfgs)
                {
                    solve_qp_aff(w, w_sub, p0, y, w_sub->J->g, 1, info);
                }
            }
        }
        else
        {
            if (stgs->hess)
            {
                solve_qp_aff(w, w_sub, p0, y, w_sub->J->g, 2, info);
            }
            else
            {
                if (stgs->bfgs)
                {
                    solve_qp_osqp(w, w_sub, p0, y, w_sub->J->g, info);
                }
            }
        }

        if (!stgs->hess && !stgs->bfgs)
        {
            if (stgs->drsom == 0 || (stgs->drsom && !w->p_old))
            {
                w->p_old = (zeronp_float *)zeronp_malloc(w->n * sizeof(zeronp_float));
                memcpy(w->p_old, w->p, w->n * sizeof(zeronp_float));
                // Gradient Descent
                Gradient_descent(w, p0, w_sub->J->g, stgs->gd_step);
            }
            else
            {
                // calculate momentum
                zeronp_float *m = (zeronp_float *)zeronp_malloc(w->n * sizeof(zeronp_float));
                memcpy(m, w->p, w->n * sizeof(zeronp_float));
                zeronp_add_scaled_array(m, w->p_old, w->n, -1.);

                // update old p
                memcpy(w->p_old, w->p, w->n * sizeof(zeronp_float));

                if (ZERONP(norm)(m, w->n) > 1e-8)
                {
                    // Use DRSOM update
                    mf = drsom(w, w_sub, stgs, p0, w_sub->J->g, m, w->radius, j);
                }
                else
                {
                    // Gradient Descent
                    Gradient_descent(w, p0, w_sub->J->g, stgs->gd_step);
                }
                zeronp_free(m);
            }
        }

        /*********************************************************/
        // Test trust region problem solver
        /*zeronp_float* G = (zeronp_float*)zeronp_calloc(4, sizeof(zeronp_float));
        zeronp_float* Q = (zeronp_float*)zeronp_malloc(4 * sizeof(zeronp_float));
        zeronp_float* c = (zeronp_float*)zeronp_malloc(2 * sizeof(zeronp_float));
        G[0] = 1; G[3] = 1;
        Q[0] = -1; Q[1] = 0.5; Q[2] = 0.5; Q[3] = -1;
        c[0] = 1; c[1] = 1.5;

        zeronp_float* alpha_mval = sub_trustreg(Q, c, G, 1, 1e-6);

        zeronp_printf("the alpha is (%f,%f), value is %f\n", alpha_mval[0], alpha_mval[1], alpha_mval[2]);

        zeronp_free(alpha_mval);
        zeronp_free(G);
        zeronp_free(Q);
        zeronp_free(c);*/

        // Test interpolation2d
        /* zeronp_float* d1 = (zeronp_float*)zeronp_calloc(w->n, sizeof(zeronp_float));
         zeronp_float* d2 = (zeronp_float*)zeronp_calloc(w->n, sizeof(zeronp_float));
         d1[0] = 1 / sqrt(3); d1[1] = 1 / sqrt(3); d1[2] = 1 / sqrt(3);
         d2[0] = 1 / sqrt(3); d2[1] = -1 / (2*sqrt(3)); d2[2] = -1 / (2 * sqrt(3));

         zeronp_float* g_and_Q = interpolate2d(w, stgs, w_sub, 1e-2, d1, d2, w->p, j);

         zeronp_free(g_and_Q);
         zeronp_free(d1);
         zeronp_free(d2);*/
        /*********************************************************/

        // Line Search
        tag = linesearch(w_sub, w, stgs, &j, &reduce, p0, sx);
        if (stgs->drsom)
        {
            // change radius of interpolatoion
            zeronp_float ratio = -(j - w->ob->obj) / mf;
            if (ratio >= 0.2)
            {
                w->radius = MIN(1.5 * w->radius, 100);
            }
            else if (ratio <= 0.01)
            {
                w->radius = MAX(w->radius / 1.5, stgs->tol / 10);
            }
        }
        j = calculate_ALM(w->ob, stgs, w->p, w, w_sub); // w->ob->obj;
        if (w->exit >= 1)
        {
            break;
        }
        memcpy(yg, w_sub->J->g, w_sub->J->npic * sizeof(zeronp_float));
        if (tag)
        {
            break;
        }
    }
    memcpy(w->l, y, w->nc * sizeof(zeronp_float));
    // TODO:
    /* unscale and record information */
    unscale(w_sub, w, stgs);
    zeronp_free(y);
    zeronp_free(yg);
    zeronp_free(sx);
    zeronp_free(p0);
    /*
    if (reduce > MAX(stgs->tol,10*stgs->delta) && w->exit == 0) {
        printf("ZeroNP--> Minor optimization routine did not converge \n         in the specified number of minor iterations.\n         You may need to increase the number of minor iterations. \n ");
    }*/

    return 0;
}

zeronp_int free_Jacob(SUBNPJacob *J)
{
    if (J)
    {
        if (J->g)
        {
            zeronp_free(J->g);
        }
        if (J->a)
        {
            zeronp_free(J->a);
        }
        if (J->anew)
        {
            zeronp_free(J->anew);
        }
        zeronp_free(J);
    }
    return 0;
}

zeronp_int free_work_subnp(SUBNPWork *w_sub)
{
    if (w_sub)
    {
        if (w_sub->alp)
        {
            zeronp_free(w_sub->alp);
        }
        if (w_sub->scale)
        {
            zeronp_free(w_sub->scale);
        }
        if (w_sub->J)
        {
            free_Jacob(w_sub->J);
        }
        if (w_sub->b)
        {
            zeronp_free(w_sub->b);
        }
        if (w_sub->p_cand)
        {
            zeronp_free(w_sub->p_cand);
        }
        if (w_sub->constr_index)
        {
            zeronp_free(w_sub->constr_index);
        }
        if (w_sub->ob_cand)
        {
            free_cost(w_sub->ob_cand);
        }
        zeronp_free(w_sub);
    }

    return 0;
}

zeronp_int SUBNP(finish)(
    SUBNPWork *w_sub,
    ZERONPWork *w,
    ZERONPSettings *stgs)
{

    if (w_sub)
    {
        free_work_subnp(w_sub);
    }
    return 0;
}

zeronp_int subnp_qp(ZERONPWork *w, ZERONPSettings *stgs, ZERONPInfo *info)
{
    SUBNPWork *w_sub = SUBNP(init)(w, stgs);

    if (w_sub)
    {
        SUBNP(solve)
        (w_sub, w, stgs, info);

        SUBNP(finish)
        (w_sub, w, stgs);
    }

    // info->qpsolver_time = qpsolver_time;
    // qpsolver_time = 0;

    return 0;
}
