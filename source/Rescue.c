#include "subnp.h"
#include "linalg.h"
#include "zeronp_util.h"
#include "rescue.h"

ZERONPWork *init_work_RESCUE(
    ZERONPWork *w_old,
    ZERONPSettings *stgs)
{
    ZERONPWork *w = (ZERONPWork *)zeronp_malloc(sizeof(ZERONPWork));
    w->n = w_old->n + w_old->nec;
    w->nec = w_old->nec;
    w->nic = w_old->nic;
    w->nc = w->nec + w->nic;
    w->ob = init_cost(w->nec, w->nic);
    copyZERONPCost(w->ob, w_old->best_fea_ob);
    w->ob->cost = w_old->ob->cost;
    w->pb = (ZERONPConstraint *)zeronp_malloc(sizeof(ZERONPConstraint));

    w->pb->pu = (zeronp_float *)zeronp_malloc((w->n) * sizeof(zeronp_float));
    memcpy(w->pb->pu, w_old->pb->pu, (w->n - w->nec) * sizeof(zeronp_float));
    w->pb->pl = (zeronp_float *)zeronp_malloc((w->n) * sizeof(zeronp_float));
    memcpy(w->pb->pl, w_old->pb->pl, (w->n - w->nec) * sizeof(zeronp_float));
    for (zeronp_int i = 0; i < w->nec; i++)
    {
        w->pb->pu[i + w->n - w->nec] = INFINITY;
        w->pb->pl[i + w->n - w->nec] = -INFINITY;
    }
    w->pb->n = w_old->pb->n;
    w->pb->nic = w_old->pb->nic;
    w->pb->Ipb = (zeronp_int *)zeronp_malloc(2 * sizeof(zeronp_int));
    w->pb->Ipc = (zeronp_int *)zeronp_malloc(2 * sizeof(zeronp_int));
    w->pb->il = (zeronp_float *)zeronp_malloc(w->nic * sizeof(zeronp_float));
    w->pb->iu = (zeronp_float *)zeronp_malloc(w->nic * sizeof(zeronp_float));
    memcpy(w->pb->il, w_old->pb->il, w->nic * sizeof(zeronp_float));
    memcpy(w->pb->iu, w_old->pb->iu, w->nic * sizeof(zeronp_float));
    memcpy(w->pb->Ipb, w_old->pb->Ipb, 2 * sizeof(zeronp_int));
    memcpy(w->pb->Ipc, w_old->pb->Ipc, 2 * sizeof(zeronp_int));

    w->rho = 1;

    w->constraint = (zeronp_float *)zeronp_malloc(w->nc * sizeof(zeronp_float));
    w->p = (zeronp_float *)zeronp_malloc((w->n + w->nic) * sizeof(zeronp_float));
    w->l = (zeronp_float *)zeronp_malloc(MAX(1, w->nc) * sizeof(zeronp_float));
    w->bestl = (zeronp_float *)zeronp_malloc(MAX(1, w->nc) * sizeof(zeronp_float));
    w->best_fea_l = (zeronp_float *)zeronp_malloc(MAX(1, w->nc) * sizeof(zeronp_float));

    w->h = (zeronp_float *)zeronp_calloc((w->nic + w->n) * (w->nic + w->n), sizeof(zeronp_float));
    w->jh = (zeronp_float *)zeronp_malloc((stgs->max_iter + 2) * sizeof(zeronp_float));
    w->ch = (zeronp_float *)zeronp_malloc((stgs->max_iter + 2) * sizeof(zeronp_float));
    w->bestp = (zeronp_float *)zeronp_malloc((w->n) * sizeof(zeronp_float));
    w->best_fea_p = (zeronp_float *)zeronp_malloc((w->n) * sizeof(zeronp_float));
    w->best_fea_ob = init_cost(w->nec, w->nic);
    w->bestob = init_cost(w->nec, w->nic);

    w->const_time = 0;
    w->count_cost = w_old->count_cost;
    w->count_grad = w_old->count_grad;
    w->count_hess = w_old->count_hess;
    w->exit = 0;

    memcpy(w->p, w_old->best_fea_p, (w_old->n) * sizeof(zeronp_float));
    memcpy(w->best_fea_p, w_old->best_fea_p, (w_old->n) * sizeof(zeronp_float));

    for (zeronp_int i = 0; i < w->nec; i++)
    {
        w->p[i + w->n - w->nec] = 1;
    }
    memcpy(w->l, w_old->best_fea_l, w->nc * sizeof(zeronp_float));
    ZERONP(add_scaled_array)
    (w->ob->ec, w->p + w->n + w->nic - w->nec, w->nec, -1.0);

    return w;
}

ZERONPWork *ZERONP_RESCUE_init(
    ZERONPWork *w_old,
    ZERONPSettings *stgs)
{
    ZERONPWork *w;
    w = init_work_RESCUE(w_old, stgs);

    return w;
}

zeronp_int update_work_rescue(
    ZERONPWork *w,
    ZERONPSettings *stgs)
{
    zeronp_int nec = w->nec;
    zeronp_int nic = w->nic;
    zeronp_int nc = w->nc;
    zeronp_int i;

    w->bestobj = INFINITY;
    w->best_fea_con = INFINITY;
    if (nc > 0.5)
    {

        memcpy(w->constraint, w->ob->ec, nec * sizeof(zeronp_float));
        memcpy(&w->constraint[nec], w->ob->ic, nic * sizeof(zeronp_float));

        if (nic > 0.5)
        {
            for (i = 0; i < nic; i++)
            {
                if (w->constraint[nec + i] <= w->pb->il[i] || w->constraint[nec + i] >= w->pb->iu[i])
                    break;
            }
            if (i == nic)
                memcpy(w->p, &w->constraint[nec], nic * sizeof(zeronp_float));

            ZERONP(add_scaled_array)
            (&w->constraint[nec], w->p, nic, -1.0);
        }

        w->cons_nm1 = ZERONP(norm)(w->constraint, w->nc);
        w->bestcon = w->cons_nm1;
    }
    else
    {
        w->l[0] = 0;
        w->constraint = ZERONP_NULL;
    }

    w->mu = w->n;
    w->best_fea_con = w->cons_nm1;
    w->j = ZERONP(norm_sq)(w->p + w->nic + w->n - w->nec, w->nec);
    w->ob->obj = w->j;
    w->jh[0] = w->j;
    // initiate w->h
    for (i = 0; i < w->nic + w->n; i++)
    {
        w->h[i + i * w->n] = 1;
    }

    return 0;
}

zeronp_int ZERONP_RESCUE_solve(
    ZERONPWork *w,
    ZERONPSettings *stgs,
    ZERONPInfo *info)
{
    zeronp_int i = 0;
    zeronp_int j;

    zeronp_int n = w->n;
    zeronp_int nc = w->nc;
    zeronp_int nec = w->nec;
    zeronp_int nic = w->nic;
    zeronp_int restart = 0;
    zeronp_float delta0 = stgs->delta;
    update_work_rescue(w, stgs);

    for (i = 0; i < stgs->max_iter; i++)
    {
        subnp_qp(w, stgs, info);

        // w->ob->cost(w->ob, &w->p[w->nic], n, i);
        w->obj_dif = (w->j - w->ob->obj) / MAX(ABS(w->ob->obj), 1);
        w->j = w->ob->obj;

        if (nc > 0.5)
        {

            memcpy(w->constraint, w->ob->ec, nec * sizeof(zeronp_float));
            memcpy(&w->constraint[nec], w->ob->ic, nic * sizeof(zeronp_float));

            if (nic > 0.5)
            {

                for (j = 0; j < nic; j++)
                {
                    if (w->constraint[nec + j] <= w->pb->il[j] || w->constraint[nec + j] >= w->pb->iu[j])
                        break;
                }
                if (j == nic)
                    memcpy(w->p, &w->constraint[nec], nic * sizeof(zeronp_float));

                ZERONP(add_scaled_array)
                (&w->constraint[nec], w->p, nic, -1.0);
            }

            w->cons_nm2 = ZERONP(norm)(w->constraint, nc);
            // previous 10
            if (w->cons_nm2 < 1 * stgs->tol_con)
            {
                w->rho = 0;
                w->mu = MIN(w->mu, stgs->tol);
            }
            // previously 5
            if (w->cons_nm2 < w->cons_nm1 && w->cons_nm2 < 10 * stgs->tol_con)
            {
                w->rho /= 5;
            }
            else if (w->cons_nm2 > 10 * w->cons_nm1)
            {
                w->rho = 5 * MAX(w->rho, SQRTF(stgs->tol));
            }
            if (w->exit == 0 && MAX(w->obj_dif, w->cons_nm1 - w->cons_nm2) <= 0 && restart < stgs->re_time && (stgs->delta <= MAX(3 * stgs->tol, stgs->delta_end) || stgs->grad))
            {
                ZERONP(scale_array)
                (w->l, 0, nc);
                restart++;
                if (stgs->noise)
                {
                    stgs->delta = delta0;
                }
                // h=diag(diag(h));
                if (stgs->noise)
                {
                    stgs->delta = delta0;
                }
                for (int col = 0; col < w->nic + w->n; col++)
                {
                    for (int row = 0; row < w->nic + w->n; row++)
                    {
                        if (col != row)
                        {
                            w->h[col * (w->nic + w->n) + row] = 0;
                        }
                    }
                }
            }

            w->cons_nm1 = w->cons_nm2;
        }

        w->jh[1 + i] = w->j;
        w->jh[2 + i] = INFINITY;
        w->ch[1 + i] = w->cons_nm1;

        if ((w->cons_nm1 <= stgs->tol_con && w->obj_dif <= stgs->tol && (stgs->delta <= MAX(stgs->tol, stgs->delta_end) || stgs->grad)) || w->exit)
        {
            if (restart < stgs->re_time && w->alm_crit > stgs->tol_restart)
            {
                restart++;
                if (stgs->noise)
                {
                    stgs->delta = delta0;
                }
                // h=diag(diag(h));
                for (int col = 0; col < w->nic + w->n; col++)
                {
                    for (int row = 0; row < w->nic + w->n; row++)
                    {
                        if (col != row)
                        {
                            w->h[col * (w->nic + w->n) + row] = 0;
                        }
                    }
                }
            }
            else
            {
                i++;
                break;
            }
        }
    }

    if (w->cons_nm1 <= stgs->tol_con && w->obj_dif <= stgs->tol && (stgs->delta <= MAX(stgs->tol, stgs->delta_end) || stgs->grad == 1))
    {
        printf("ZeroNP--> RESCUE process Success! Completed in %d iterations\n", i);
    }
    else
    {
        printf("ZeroNP--> Exiting after maximum number of function evaluation. Rescue Process Fails.\n");
    }
    return i;
}

ZERONPWork *ZERONP_RESCUE(
    ZERONPWork *w_old,
    ZERONPSettings *stgs,
    ZERONPInfo *info // record running time
)
{
    zeronp_int status;
    ZERONPWork *w = ZERONP_RESCUE_init(w_old, stgs);
    if (w)
    {
        ZERONP_RESCUE_solve(w, stgs, info);
    }

    return w;
}
