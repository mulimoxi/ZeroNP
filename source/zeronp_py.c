#include "zeronp.h"
#include "linalg.h"
#include "zeronp_util.h"

// zeronp_float cost_time = 0; // global cost timer

// typedef void py_cost_temple(double *p, double *result, int n);
typedef void py_cost_temple(double *p, double *result);
typedef void py_g_temple(double *p, double *result);
typedef void py_h_temple(double *p, double *result);

py_cost_temple *py_cost;
py_g_temple *py_grad;
py_h_temple *py_hess;

void def_python_callback(py_cost_temple *cost, py_g_temple *grad, py_h_temple *hess)
{
    // Function called by Python once
    // Defines what "py_cost" is pointing to
    py_cost = cost;
    py_grad = grad;
    py_hess = hess;
}

void C_call_python_cost(ZERONPCost **c, zeronp_float *p, zeronp_int np, zeronp_int nfeval)
{

    ZERONP(timer)
    cost_timer;
    ZERONP(tic)
    (&cost_timer);
    zeronp_int i, j;

    double *result = (double *)zeronp_malloc(sizeof(double) * (1 + c[0]->nic + c[0]->nec) * nfeval);
    // py_cost(p, result, nfeval);
    py_cost(p, result);

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

void C_call_python_grad(zeronp_float *g, zeronp_float *p, zeronp_int np, zeronp_int ngeval)
{

    ZERONP(timer)
    cost_timer;
    ZERONP(tic)
    (&cost_timer);
    zeronp_int i;

    double *result = (double *)zeronp_malloc(sizeof(double) * np * ngeval);
    py_grad(p, result);

    for (i = 0; i < np * ngeval; i++)
    {
        g[i] = (zeronp_float)result[i];
    }
    // cost_time += ZERONP(tocq)(&cost_timer) / 1e3;
}

void C_call_python_hess(zeronp_float *h, zeronp_float *p, zeronp_int np, zeronp_int nheval)
{

    ZERONP(timer)
    cost_timer;
    ZERONP(tic)
    (&cost_timer);
    zeronp_int i;

    double *result = (double *)zeronp_malloc(sizeof(double) * np * np * nheval);
    py_hess(p, result);

    for (i = 0; i < np * np * nheval; i++)
    {
        h[i] = (zeronp_float)result[i];
    }
    // cost_time += ZERONP(tocq)(&cost_timer) / 1e3;
}

void ZERONP_C(
    // input
    zeronp_float *ibl,
    zeronp_float *ibu,
    zeronp_float *pbl,
    zeronp_float *pbu,
    zeronp_int *Ipc,
    zeronp_int *Ipb,
    zeronp_float *ib0,
    zeronp_float *p,
    zeronp_float *op,
    zeronp_float *l,
    zeronp_float *h,
    zeronp_int np,
    zeronp_int nic,
    zeronp_int nec,
    // output
    zeronp_float *scalars,
    zeronp_float *p_out,
    zeronp_float *best_fea_p,
    zeronp_float *ic,
    zeronp_float *jh,
    zeronp_float *ch,
    zeronp_float *l_out,
    zeronp_float *h_out,
    zeronp_float *count_h
    )
{
    ZERONP(timer)
    total_timer;
    ZERONP(tic)
    (&total_timer);

    zeronp_int index, i;
    /*--------------------- construct settings ------------------------*/
    ZERONPSettings *stgs = (ZERONPSettings *)zeronp_malloc(sizeof(ZERONPSettings));
    ZERONP(set_default_settings)
    (stgs, np);
    stgs->maxfev = 500 * np;
    index = 0;

    stgs->rho = (zeronp_float)op[index++];
    stgs->pen_l1 = (zeronp_float)op[index++];
    stgs->max_iter = (zeronp_int)op[index++];
    stgs->min_iter = (zeronp_int)op[index++];
    stgs->max_iter_rescue = (zeronp_int)op[index++];
    stgs->min_iter_rescue = (zeronp_int)op[index++];
    stgs->delta = (zeronp_float)op[index++];
    stgs->tol = (zeronp_float)op[index++];
    stgs->tol_con = (zeronp_float)op[index++];
    stgs->ls_time = (zeronp_int)op[index++];
    stgs->batchsize = (zeronp_int)op[index++];
    stgs->tol_restart = (zeronp_float)op[index++];
    stgs->re_time = (zeronp_int)op[index++];
    stgs->delta_end = (zeronp_float)op[index++];
    stgs->maxfev = (zeronp_int)op[index++];
    stgs->noise = (zeronp_int)op[index++];
    stgs->qpsolver = (zeronp_int)op[index++];
    stgs->scale = (zeronp_int)op[index++];
    stgs->bfgs = (zeronp_int)op[index++];
    stgs->rs = (zeronp_int)op[index++];
    stgs->grad = (zeronp_int)op[index++];
    stgs->k_i = (zeronp_float)op[index++];
    stgs->k_r = (zeronp_float)op[index++];
    stgs->c_r = (zeronp_float)op[index++];
    stgs->c_i = (zeronp_float)op[index++];
    stgs->ls_way = (zeronp_int)op[index++];
    stgs->rescue = (zeronp_int)op[index++];
    stgs->drsom = (zeronp_int)op[index++];
    stgs->cen_diff = (zeronp_int)op[index++];
    stgs->gd_step = (zeronp_int)op[index++];
    stgs->step_ratio = (zeronp_float)op[index++];
    stgs->verbose = (zeronp_int)op[index++];

    /*--------------------- check function from python ------------------------*/
    if (py_cost == ZERONP_NULL)
    {
        zeronp_printf("ZeroNP--> The user does not provided cost function in the fun structure. \n");
        zeronp_printf("ZeroNP--> ZERONP stops.\n");
        exit(1);
    }

    if (py_grad != ZERONP_NULL)
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

    if (py_hess != ZERONP_NULL)
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
    input->stgs = stgs;
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
    // first call python cost function
    double *result = (double *)zeronp_malloc(sizeof(double) * (nec + nic + 1));
    // py_cost(p, result, 1);
    py_cost(p, result);

    zeronp_int m = nec + nic + 1;
    zeronp_float *ob = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * m);
    for (i = 0; i < m; i++)
    {
        ob[i] = (zeronp_float)result[i];
    }

    ZERONPCost *cost = malloc_cost(nec, nic, &C_call_python_cost, &C_call_python_grad, &C_call_python_hess);
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
    ZERONPSol *sol = (ZERONPSol *)zeronp_malloc(sizeof(ZERONPSol));
    ZERONPInfo *info = (ZERONPInfo *)zeronp_malloc(sizeof(ZERONPInfo));

    zeronp_float *ib0_p = (zeronp_float *)zeronp_malloc(sizeof(zeronp_float) * (np + nic));

    if (nic > 0)
    {
        memcpy(ib0_p, ib0, nic * sizeof(zeronp_float));
        // zeronp_free(ib0);
    }

    memcpy(&ib0_p[nic], p, np * sizeof(zeronp_float));

    // zeronp_printf("Ipb[0]: %f\n", constraint->Ipb[0]);
    // zeronp_printf("Ipb[1]: %f\n", constraint->Ipb[1]);
    // zeronp_printf("Ipc[0]: %f\n", constraint->Ipc[0]);
    // zeronp_printf("Ipc[1]: %f\n", constraint->Ipc[1]);

    info->total_time = 0;
    /*--------------------- call zeronp ------------------------*/
    zeronp_int status = ZERONP(main)(input, cost, ib0_p, sol, info);
    /*--------------------- assemble output ------------------------*/
    info->total_time += ZERONP(tocq)(&total_timer) / 1e3;

    index = 0;

    scalars[index++] = (zeronp_float)sol->iter;
    scalars[index++] = (zeronp_float)sol->count_cost;
    scalars[index++] = (zeronp_float)sol->count_grad;
    scalars[index++] = (zeronp_float)sol->count_hess;
    scalars[index++] = (zeronp_float)sol->constraint;
    scalars[index++] = (zeronp_float)sol->restart_time;
    scalars[index++] = (zeronp_float)sol->obj;
    scalars[index++] = (zeronp_float)sol->status;
    scalars[index++] = (zeronp_float)info->total_time;
    scalars[index++] = (zeronp_float)info->qpsolver_time;

    memcpy(p_out, sol->p, np * sizeof(zeronp_float));
    memcpy(best_fea_p, sol->best_fea_p, np * sizeof(zeronp_float));
    memcpy(ic, sol->ic, MAX(nic, 1) * sizeof(zeronp_float));
    memcpy(jh, sol->jh, (sol->iter + 1) * sizeof(zeronp_float));
    memcpy(ch, sol->ch, (sol->iter + 1) * sizeof(zeronp_float));
    memcpy(count_h, sol->count_h, (sol->iter + 1) * sizeof(zeronp_float));
    memcpy(l_out, sol->l, MAX(nic + nec, 1) * sizeof(zeronp_float));
    if (stgs->bfgs)
    {
        memcpy(h_out, sol->h, (np + nic) * (np + nic) * sizeof(zeronp_float));
    }
    else
    {
        memcpy(h_out, sol->h, 1 * sizeof(zeronp_float));
    }

    zeronp_free(info);
    // ZERONP(free_sol)(sol);
    zeronp_printf("ZeroNP--> ZERONP finished.\n");
}