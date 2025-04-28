#include "zeronp.h"
#include "zeronp_util.h"

cost_temple *cost_fun_c;
g_temple *grad_fun_c;
h_temple *hess_fun_c;

void def_c_callback(cost_temple *cost, g_temple *grad, h_temple *hess)
{
    cost_fun_c = cost;
    grad_fun_c = grad;
    hess_fun_c = hess;
}

void C_call_cost(ZERONPCost **c, zeronp_float *p, zeronp_int np, zeronp_int nfeval)
{

    ZERONP(timer)
    cost_timer;
    ZERONP(tic)
    (&cost_timer);
    zeronp_int i, j;

    double *result = (double *)zeronp_malloc(sizeof(double) * (1 + c[0]->nic + c[0]->nec) * nfeval);
    // cost_fun_c(p, result, np, nfeval);
    cost_fun_c(p, result, np);

    for (j = 0; j < nfeval; j++)
    {
        c[j]->obj = (zeronp_float)result[j * (1 + c[j]->nic + c[j]->nec)];

        for (i = 0; i < c[j]->nec; i++)
        {
            c[j]->ec[i] = (zeronp_float)result[i + 1 + j * (1 + c[j]->nic + c[j]->nec)];
        }
        for (i = 0; i < c[j]->nic; i++)
        {
            c[j]->ic[i] = (zeronp_float)result[i + 1 + c[j]->nec + j * (1 + c[j]->nic + c[j]->nec)];
        }
    }

    zeronp_free(result);
    // cost_time += ZERONP(tocq)(&cost_timer) / 1e3;
}

void C_call_grad(zeronp_float *g, zeronp_float *p, zeronp_int np, zeronp_int ngeval)
{

    ZERONP(timer)
    cost_timer;
    ZERONP(tic)
    (&cost_timer);
    zeronp_int i;

    double *result = (double *)zeronp_malloc(sizeof(double) * np * ngeval);
    grad_fun_c(p, result);

    for (i = 0; i < np * ngeval; i++)
    {
        g[i] = (zeronp_float)result[i];
    }
    // cost_time += ZERONP(tocq)(&cost_timer) / 1e3;
}

void C_call_hess(zeronp_float *h, zeronp_float *p, zeronp_int np, zeronp_int nheval)
{

    ZERONP(timer)
    cost_timer;
    ZERONP(tic)
    (&cost_timer);
    zeronp_int i;

    double *result = (double *)zeronp_malloc(sizeof(double) * np * np * nheval);
    hess_fun_c(p, result);

    for (i = 0; i < np * np * nheval; i++)
    {
        h[i] = (zeronp_float)result[i];
    }
    // cost_time += ZERONP(tocq)(&cost_timer) / 1e3;
}

void ZERONP_C_INTERFACE(
    // input
    ZERONPSettings *stgs,
    zeronp_float *ibl,
    zeronp_float *ibu,
    zeronp_float *pbl,
    zeronp_float *pbu,
    zeronp_int *Ipc,
    zeronp_int *Ipb,
    zeronp_float *ib0,
    zeronp_float *p,
    zeronp_float *l,
    zeronp_float *h,
    zeronp_int np,
    zeronp_int nic,
    zeronp_int nec,
    // output
    ZERONPSol *sol,
    ZERONPInfo *info)
{

    ZERONP(timer)
    total_timer;
    ZERONP(tic)
    (&total_timer);

    zeronp_int index, i;

    /*--------------------- check function ------------------------*/
    if (cost_fun_c == ZERONP_NULL)
    {
        zeronp_printf("ZeroNP--> The user does not provided cost function in the fun structure. \n");
        zeronp_printf("ZeroNP--> ZERONP stops.\n");
        exit(1);
    }

    if (grad_fun_c != ZERONP_NULL)
    {
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

    if (hess_fun_c != ZERONP_NULL)
    {
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

    /*--------------------- construct constraint ------------------------*/
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

    /*--------------------- assemble input ------------------------*/
    zeronp_int nc = nec + nic;
    zeronp_int h_len = (np + nic) * (np + nic);
    ZERONPIput *input = (ZERONPIput *)zeronp_malloc(sizeof(ZERONPIput));
    input->cnstr = constraint;
    input->stgs = (ZERONPSettings *)zeronp_malloc(sizeof(ZERONPSettings));
    memcpy(input->stgs, stgs, sizeof(ZERONPSettings));
    // input->stgs = stgs;
    input->n = np;
    input->h = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * h_len);
    input->l = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * nc);

    memcpy(input->h, h, h_len * sizeof(zeronp_float));
    memcpy(input->l, l, nc * sizeof(zeronp_float));
    /*--------------------- construct cost ------------------------*/
    ZERONP(timer)
    cost_timer;
    ZERONP(tic)
    (&cost_timer);
    // first call cost function
    double *result = (double *)zeronp_malloc(sizeof(double) * (nec + nic + 1));
    // cost_fun_c(p, result, np, 1);
    cost_fun_c(p, result, np);
    zeronp_int m = nec + nic + 1;
    zeronp_float *ob = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * m);
    for (i = 0; i < m; i++)
    {
        ob[i] = (zeronp_float)result[i];
    }
    ZERONPCost *cost = malloc_cost(nec, nic, C_call_cost, C_call_grad, C_call_hess);

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
    // zeronp_free(result);
    /*--------------------- construct other params ------------------------*/

    zeronp_float *ib0_p = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * (np + nic));

    if (nic > 0)
    {
        memcpy(ib0_p, ib0, nic * sizeof(zeronp_float));
    }

    memcpy(&ib0_p[nic], p, np * sizeof(zeronp_float));

    info->total_time = 0;
    /*--------------------- call zeronp ------------------------*/
    zeronp_int status = ZERONP(main)(input, cost, ib0_p, sol, info);
    /*--------------------- assemble output ------------------------*/
    info->total_time += ZERONP(tocq)(&total_timer) / 1e3;

    zeronp_printf("ZeroNP--> ZERONP finished.\n");
}

zeronp_int check_strict_in_bound(zeronp_float *x, zeronp_float *lb, zeronp_float *ub, zeronp_int len)
{
    /*
    check lb < x < ub
    */

    zeronp_int inbound_status = 1;

    for (zeronp_int i = 0; i < len; i++)
    {
        if (x[i] <= lb[i] || x[i] >= ub[i])
        {
            inbound_status = 0;
            break;
        }
    }

    return inbound_status;
}

void cal_avg_arr(zeronp_float *x, zeronp_float *y, zeronp_float *x_y_2, zeronp_int len)
{

    for (zeronp_int i = 0; i < len; i++)
    {
        x_y_2[i] = (x[i] + y[i]) / 2;
    }
}

zeronp_int check_var_bound(zeronp_float *x, zeronp_float *lb, zeronp_float *ub, zeronp_int len)
{
    /*
    check
        - x is not None
        - len(x) == len(lb) == len(ub)
        - lb < x < ub
    */

    zeronp_int var_bound_status = 1;

    if (x == ZERONP_NULL)
    {
        var_bound_status = 0;
    }
    else
    {
        if (check_strict_in_bound(x, lb, ub, len) == 0)
        {
            var_bound_status = 0;
        }
    }
    return var_bound_status;
}

zeronp_int check_bound(zeronp_float *lb, zeronp_float *ub, zeronp_int len)
{
    /*
    check lb < ub
    */

    zeronp_int bound_status = 1;

    for (zeronp_int i = 0; i < len; i++)
    {
        if (lb[i] >= ub[i])
        {
            bound_status = 0;
            break;
        }
    }

    return bound_status;
}

void fill_array(zeronp_float *arr, zeronp_float value, zeronp_int len)
{
    /*
    fill array with value
    */

    for (zeronp_int i = 0; i < len; i++)
    {
        arr[i] = value;
    }
}

zeronp_int check_prob(ZERONPProb *prob)
{
    /*
    check if prob is legal
    and init those None items
    */

    zeronp_int prob_status = 1;

    // Ipc, Ipb
    if (prob->Ipc == ZERONP_NULL)
    {
        prob->Ipc = (zeronp_int *)zeronp_malloc(sizeof(zeronp_int) * 2);
        prob->Ipc[0] = 0;
        prob->Ipc[1] = 0;
    }

    if (prob->Ipb == ZERONP_NULL)
    {
        prob->Ipb = (zeronp_int *)zeronp_malloc(sizeof(zeronp_int) * 2);
        prob->Ipb[0] = 0;
        prob->Ipb[1] = 0;
    }

    if (prob->pbl)
    {
        prob->Ipc[0] = 1;
        prob->Ipb[0] = 1;
    }

    if (prob->pbu)
    {
        prob->Ipc[1] = 1;
        prob->Ipb[0] = 1;
    }

    if (prob->Ipb[0] + prob->nic > 0.5)
    {
        prob->Ipb[1] = 1;
    }

    // ibl, ibu and ib0
    if (prob->nic > 0)
    {
        // init ibl, ibu
        if (prob->ibl == ZERONP_NULL)
        {
            prob->ibl = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * prob->nic);
            fill_array(prob->ibl, -INFINITY, prob->nic);
        }
        if (prob->ibu == ZERONP_NULL)
        {
            prob->ibu = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * prob->nic);
            fill_array(prob->ibu, INFINITY, prob->nic);
        }
        // check and set ib0
        if (check_bound(prob->ibl, prob->ibu, prob->nic) == 0)
        {
            prob_status = 0;
            zeronp_printf("Inequality bound error!");
            exit(1);
        }
        if (check_var_bound(prob->ib0, prob->ibl, prob->ibu, prob->nic) == 0)
        {
            if (prob->ib0 == ZERONP_NULL)
            {
                prob->ib0 = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * prob->nic);
            }
            cal_avg_arr(prob->ibl, prob->ibu, prob->ib0, prob->nic);
        }
    }
    // pbl, pbu and p0
    if (prob->np > 0)
    {
        // init pbl, pbu
        if (prob->pbl == ZERONP_NULL)
        {
            prob->pbl = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * prob->np);
            fill_array(prob->pbl, -INFINITY, prob->np);
        }
        if (prob->pbu == ZERONP_NULL)
        {
            prob->pbu = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * prob->np);
            fill_array(prob->pbu, INFINITY, prob->np);
        }
        // check and set p0
        if (check_bound(prob->pbl, prob->pbu, prob->np) == 0)
        {
            prob_status = 0;
            zeronp_printf("Variable bound error!");
            exit(1);
        }
        if (check_var_bound(prob->p0, prob->pbl, prob->pbu, prob->np) == 0)
        {
            if (prob->p0 == ZERONP_NULL)
            {
                prob->p0 = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * prob->np);
            }
            cal_avg_arr(prob->pbl, prob->pbu, prob->p0, prob->np);
        }
    }
    else
    {
        prob_status = 0;
        zeronp_printf("Dim of variables is 0!");
        exit(1);
    }

    return prob_status;
}

void ZERONP_PLUS(
    ZERONPProb *prob,
    ZERONPSettings *stgs,
    ZERONPSol *sol,
    ZERONPInfo *info,
    cost_temple *cost_fun,
    g_temple *grad_fun,
    h_temple *hess_fun,
    zeronp_float *l,
    zeronp_float *h)
{

    if (check_prob(prob) == 0)
    {
        zeronp_printf("Illegal Problem!");
        exit(1);
    }

    zeronp_int l_status = 1;

    if (l == ZERONP_NULL)
    {
        l_status = 0;
        l = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * prob->nc);
        fill_array(l, 0, prob->nc);
    }

    zeronp_int h_status = 1;

    if (h == ZERONP_NULL)
    {
        h_status = 0;

        h = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * (prob->np + prob->nic) * (prob->np + prob->nic));
        fill_array(h, 0, (prob->np + prob->nic) * (prob->np + prob->nic));

        for (int diag = 0; diag < (prob->np + prob->nic); diag++)
        {
            h[diag * (prob->np + prob->nic) + diag] = 1;
        }
    }

    def_c_callback(cost_fun, grad_fun, hess_fun);

    ZERONP_C_INTERFACE(
        stgs,
        prob->ibl,
        prob->ibu,
        prob->pbl,
        prob->pbu,
        prob->Ipc,
        prob->Ipb,
        prob->ib0,
        prob->p0,
        l,
        h,
        prob->np,
        prob->nic,
        prob->nec,
        sol,
        info);

    if (l_status == 0)
    {
        zeronp_free(l);
    }
    if (h_status == 0)
    {
        zeronp_free(h);
    }
}