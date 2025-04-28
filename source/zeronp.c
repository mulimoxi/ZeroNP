#include "zeronp.h"
#include "linalg.h"
#include "subnp.h"
#include "zeronp_util.h"
#include "rescue.h"
#include "der_info.h"

ZERONPWork *init_work(
	ZERONPIput *input,
	ZERONPCost *cost)
{
	ZERONPWork *w = (ZERONPWork *)zeronp_malloc(sizeof(ZERONPWork));
	/*if (cost->nec >= input->n) {
		input->stgs->rescue = 1;
	}
	else {
		input->stgs->rescue = 0;
	}*/
	if (input->stgs->rs|| input->stgs->drsom)
	{
		input->stgs->bfgs = 0;
		input->stgs->scale = 0;
		input->stgs->noise = 0;
		input->stgs->min_iter = 1;
		input->stgs->max_iter = input->stgs->maxfev;
		input->stgs->re_time = input->stgs->max_iter;
		input->stgs->delta = input->stgs->delta_end;
	}
	w->n = input->stgs->rescue ? input->n + 2 * cost->nec : input->n;
	w->nec = cost->nec;
	w->nic = cost->nic;
	w->nc = w->nec + w->nic;
	w->ob = cost;
	w->pb = input->cnstr;
	w->rho = input->stgs->rho;
	w->pen_l1 = input->stgs->pen_l1;
	w->restart = 0;

	w->constraint = (zeronp_float *)zeronp_malloc(w->nc * sizeof(zeronp_float));
	w->p = (zeronp_float *)zeronp_malloc((w->n + w->nic) * sizeof(zeronp_float));
	w->p_old = ZERONP_NULL;

	w->l = (zeronp_float *)zeronp_malloc(MAX(1, w->nc) * sizeof(zeronp_float));
	w->best_fea_l = (zeronp_float *)zeronp_malloc(MAX(1, w->nc) * sizeof(zeronp_float));
	w->bestl = (zeronp_float *)zeronp_calloc(MAX(1, w->nc), sizeof(zeronp_float));
	// w->h = (zeronp_float*)zeronp_calloc((w->nic + w->n) * (w->nic + w->n), sizeof(zeronp_float));
	w->jh = (zeronp_float *)zeronp_malloc((input->stgs->max_iter + 2) * sizeof(zeronp_float));
	w->ch = (zeronp_float *)zeronp_malloc((input->stgs->max_iter + 2) * sizeof(zeronp_float));
	w->count_h = (zeronp_float *)zeronp_malloc((input->stgs->max_iter + 2) * sizeof(zeronp_float));
	w->bestp = (zeronp_float *)zeronp_malloc((input->n) * sizeof(zeronp_float));
	w->best_fea_p = (zeronp_float *)zeronp_malloc((input->n) * sizeof(zeronp_float));

	w->best_fea_ob = init_cost(w->nec, w->nic);
	w->bestob = init_cost(w->nec, w->nic);
	w->radius = 1;

	copyZERONPCost(w->best_fea_ob, w->ob);
	w->best_fea_ob->cost = w->ob->cost;

	w->count_cost = 1;
	w->const_time = 0;
	w->count_grad = 0;
	w->count_hess = 0;
	w->exit = 0;

	// Set the Random Seed
	// srand((unsigned int)time(NULL));

	return w;
}

zeronp_int update_work(
	ZERONPWork *w,
	ZERONPIput *input,
	zeronp_float *ib0_p)
{
	zeronp_int nec = w->nec;
	zeronp_int nic = w->nic;
	zeronp_int nc = w->nc;
	zeronp_int i, j;

	memcpy(w->p, ib0_p, (input->n + w->nic) * sizeof(zeronp_float));
	memcpy(w->best_fea_p, w->p + w->nic, (input->n) * sizeof(zeronp_float));
	w->bestobj = INFINITY;
	if (nc > 0.5)
	{
		memcpy(w->l, input->l, nc * sizeof(zeronp_float));
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
		w->cons_nm1_orgin = w->cons_nm1;
		if (MAX(w->cons_nm1 - 10 * input->stgs->tol, nic) <= 0)
		{
			w->rho = 0;
		}
	}
	else
	{
		w->l[0] = 0;
		w->constraint = ZERONP_NULL;
	}

	w->mu = w->n;
	w->j = w->ob->obj;
	w->count_h[0] = 1;
	w->jh[0] = w->j;
	w->ch[0] = w->cons_nm1;
	w->best_fea_con = w->cons_nm1;
	// memcpy(w->h, input->h, (w->nic+w->n - 2 * w->nec)* (w->nic + w->n - 2 * w->nec) * sizeof(zeronp_float));
	if (input->stgs->bfgs)
	{
		w->h = (zeronp_float *)zeronp_calloc((w->nic + w->n) * (w->nic + w->n), sizeof(zeronp_float));
		for (i = 0; i < input->n + w->nic; i++)
		{
			for (j = 0; j < input->n + w->nic; j++)
			{
				w->h[i + j * w->n] = input->h[i + j * (input->n)];
			}
		}
	}
	else
	{
		w->h = ZERONP_NULL;
	}
	if (input->stgs->rescue == 1)
	{
		// initiate slack variable for equality constraints
		for (i = w->nic + w->n - 2 * w->nec; i < w->nic + w->n; i++)
		{
			w->p[i] = 1;
			if (input->stgs->bfgs)
			{
				w->h[i * (w->nic + w->n) + i] = 1;
			}
		}
		// Modify objective function
		for (i = 0; i < w->nec; i++)
		{
			w->ob->obj += w->pen_l1 * (w->p[i + w->nic + w->n - 2 * w->nec] + w->p[i + w->nic + w->n - w->nec]);
		}

		// Modify bound
		if (w->nec > 0)
		{
			w->pb->n = w->n;
			zeronp_float *pl_temp, *pu_temp;
			pl_temp = w->pb->pl;
			pu_temp = w->pb->pu;
			w->pb->pl = (zeronp_float *)zeronp_malloc(w->n * sizeof(zeronp_float));
			w->pb->pu = (zeronp_float *)zeronp_malloc(w->n * sizeof(zeronp_float));
			memcpy(w->pb->pl, pl_temp, (w->n - 2 * w->nec) * sizeof(zeronp_float));
			memcpy(w->pb->pu, pu_temp, (w->n - 2 * w->nec) * sizeof(zeronp_float));
			w->pb->Ipb[0] = 1;
			w->pb->Ipb[1] = 1;
			for (i = 0; i < 2 * w->nec; i++)
			{
				w->pb->pl[i + w->nic + w->n - 2 * w->nec] = 0;
				w->pb->pu[i + w->nic + w->n - 2 * w->nec] = INFINITY;
			}
			zeronp_free(pl_temp);
			zeronp_free(pu_temp);
		}
	}
}
/*
ZERONPWork* rescue
(
	ZERONPWork* w,
	ZERONPSettings* stgs,
	ZERONPInfo* info // record running time
) {
	zeronp_int ls_time = stgs->ls_time;
	zeronp_int max_iter = stgs->max_iter;
	zeronp_int min_iter = stgs->min_iter;
	zeronp_float tol = stgs->tol;
	stgs->rescue = 1;
	stgs->ls_time = 0;
	stgs->max_iter = stgs->max_iter_rescue;
	stgs->min_iter = stgs->min_iter_rescue;
	if (stgs->max_iter == 0) {
		return w;
	}
	stgs->tol = MAX(stgs->tol* stgs->tol*10,1e-6);
	zeronp_printf("ZeroNP--> Rescue Process begins. Finding a feasible solution...\n");
	ZERONPWork* w_rescue = ZERONP_RESCUE(w, stgs, info);

	w_rescue->n -= w_rescue->nec;

	ZERONPCost** ob = &w_rescue->ob;
	w->ob->cost(ob, w_rescue->bestp + w->nic, w->n, 1, 0);
	memcpy(w_rescue->best_fea_p, w_rescue->bestp + w->nic, w->n * sizeof(float));
	memcpy(w_rescue->best_fea_l, w_rescue->best_fea_l + w->nic, w->n * sizeof(float));
	copyZERONPCost(w_rescue->best_fea_ob, w_rescue->ob);

	memcpy(w_rescue->constraint, w_rescue->ob->ec, w_rescue->nec * sizeof(zeronp_float));
	memcpy(&w_rescue->constraint[w->nec], w_rescue->ob->ic, w_rescue->nic * sizeof(zeronp_float));

	if (w->nic > 0.5) {
		ZERONP(add_scaled_array)(&w_rescue->constraint[w->nec], w->p, w_rescue->nic, -1.0);
	}

	w_rescue->cons_nm2 = ZERONP(norm)(w_rescue->constraint, w_rescue->nc);
	w_rescue->j = w_rescue->ob->obj;
	w_rescue->cons_nm1 = w_rescue->cons_nm2;

	w_rescue->bestcon = w_rescue->cons_nm2;
	w_rescue->bestobj = w_rescue->ob->obj;
	stgs->rescue = 0;
	stgs->ls_time = ls_time;
	stgs->tol = tol;
	stgs->max_iter = max_iter;
	stgs->min_iter = min_iter;
	free_work(w);
	return w_rescue;
}
*/
ZERONPWork *ZERONP(init)(
	ZERONPIput *input,
	ZERONPCost *cost,
	zeronp_float *ib0_p)
{
	ZERONPWork *w;
	w = init_work(input, cost);

	return w;
}

void restart(
	ZERONPWork *w,
	ZERONPSettings *stgs,
	zeronp_int iter,
	zeronp_float delta0)
{
	zeronp_int n = stgs->rescue ? w->n - 2 * w->nec : w->n;
	w->restart++;
	if (w->exit != 0 || w->restart > 2)
	{
		if (w->bestobj != INFINITY)
		{
			memcpy(w->p, w->bestp, n * sizeof(zeronp_float));
			memcpy(w->l, w->bestl, MAX(w->nc, 1) * sizeof(zeronp_float));
			copyZERONPCost(w->ob, w->bestob);
			w->cons_nm1_orgin = w->bestcon;
		}
		else
		{
			memcpy(w->p, w->best_fea_p, n * sizeof(zeronp_float));
			memcpy(w->l, w->best_fea_l, MAX(w->nc, 1) * sizeof(zeronp_float));
			copyZERONPCost(w->ob, w->best_fea_ob);
			w->cons_nm1_orgin = w->best_fea_con;
			w->bestcon = w->best_fea_con;
		}
		w->jh[iter] = w->ob->obj;
		w->ch[iter] = w->cons_nm1_orgin;
		w->pen_l1 = 1;
		w->j = w->ob->obj + w->pen_l1 * 2 * w->nec;
		if (stgs->bfgs)
		{
			for (int diag = 0; diag < w->nic + w->n; diag++)
			{
				w->h[diag * (w->nic + w->n) + diag] = 1;
			}
		}
		if (stgs->rescue)
		{
			for (int k = 0; k < 2 * w->nec; k++)
			{
				w->p[w->nic + w->n - w->nec * 2 + k] = 1;
			}
		}
		w->exit = 0;
		w->mu = w->n;
	}
	if (stgs->noise)
	{
		stgs->delta = delta0;
	}
	if (stgs->drsom)
	{
		// zeronp_free(w->p_old); // will be freed in free_work
		w->radius = 1;
	}
	// h=diag(diag(h));
	if (stgs->bfgs)
	{
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
}

ZERONPWork *ZERONP(solve)(
	ZERONPWork *w,
	ZERONPIput *input,
	zeronp_float *ib0_p,
	ZERONPSol *sol,
	ZERONPInfo *info)
{
	zeronp_int i = 0;
	zeronp_int j;
	ZERONPSettings *stgs = input->stgs;
	stgs->h = 1e-3;
	zeronp_int n = stgs->rescue ? w->n - 2 * w->nec : w->n;
	zeronp_int nc = w->nc;
	zeronp_int nec = w->nec;
	zeronp_int nic = w->nic;
	zeronp_int change_step_gap = stgs->noise == 3 ? (zeronp_int)(1 + 1. / stgs->step_ratio) : 0;
	zeronp_int start_chg_l1_pen = w->nec > 3 * n ? 12 : 5;

	// zeronp_int rescue_tag = 0;
	zeronp_int max_iter = stgs->max_iter;
	zeronp_int iter = 0;

	zeronp_float delta0 = stgs->delta;
	update_work(w, input, ib0_p);

	for (i = 0; i < max_iter; i++)
	{
		// if (stgs->noise == 3 &&(w->count_cost > stgs->maxfev *2/3 || w->alm_crit < stgs->tol_restart/20)) {
		//	stgs->noise = 2;
		// }

		if (stgs->noise == 3 && i % change_step_gap == 0)
		{
			stgs->noise = 2;
		}
		else if (change_step_gap != 0)
		{
			stgs->noise = 3;
		}

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
			w->cons_nm2_orgin = stgs->rescue ? calculate_infeas_scaledob_l1(w->ob, w, w->p) : w->cons_nm2;
			// previous 10
			if (w->cons_nm2 < 10 * stgs->tol_con)
			{
				w->rho = 0;
				w->mu = MIN(w->mu, stgs->tol);
			}

			// adjust the ALM penalty parameter
			// && w->cons_nm2 < 10 * stgs->tol_con
			if (w->cons_nm2 < 5 * w->cons_nm1 && w->cons_nm2 < 10 * stgs->tol_con)
			{
				w->rho /= 5;
			}
			else if (w->cons_nm2 > 10 * w->cons_nm1)
			{
				w->rho = 5 * MAX(w->rho, SQRTF(stgs->tol));
			}

			// adjust the l1 penalty parameter
			if (stgs->rescue && w->cons_nm2_orgin <= 3 * w->cons_nm1_orgin && w->cons_nm2_orgin < 10 * stgs->tol_con)
			{
				w->pen_l1 = w->pen_l1 / 2;
			}
			if (stgs->rescue && w->cons_nm2_orgin >= 5 * w->cons_nm1_orgin || (w->cons_nm2_orgin > 100 * stgs->tol_con && i >= MIN(stgs->max_iter / 4, start_chg_l1_pen)))
			{
				w->pen_l1 = MIN(w->pen_l1 * 3, 1e4);
			}
			if (w->exit == 0 && MAX(w->obj_dif, 3 * w->cons_nm1 - w->cons_nm2) <= 0 && w->restart < stgs->re_time && (stgs->delta <= MAX(3 * stgs->tol, stgs->delta_end) || stgs->grad))
			{
				ZERONP(scale_array)
				(w->l, 0, nc);
				w->restart++;
				// h=diag(diag(h));
				if (stgs->noise)
				{
					stgs->delta = delta0;
				}
				if (stgs->bfgs)
				{
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
			}

			w->cons_nm1 = w->cons_nm2;
			w->cons_nm1_orgin = w->cons_nm2_orgin;
		}

		w->count_h[1 + iter] = w->count_cost;
		w->jh[1 + iter] = w->j;
		w->ch[1 + iter] = w->cons_nm1_orgin;
		iter++;
		/*
		if (i >= max_iter / 2 && rescue_tag == 0 && w->cons_nm1 > 20 * stgs->tol_con) {

			rescue_tag = 1;
			stgs->delta = delta0;

			zeronp_float* hisj = (zeronp_float*)zeronp_malloc(2 * (stgs->max_iter + 1) * sizeof(zeronp_float));
			zeronp_float* hisc = (zeronp_float*)zeronp_malloc(2 * (stgs->max_iter + 1) * sizeof(zeronp_float));
			memcpy(hisj, w->jh, (iter + 1) * sizeof(zeronp_float));
			memcpy(hisc, w->ch, (iter + 1) * sizeof(zeronp_float));
			w = rescue(w, stgs, info);

			j = 0;
			while (w->jh[j] != INFINITY) {
				j++;
			}
			memcpy(hisj + (iter + 1), w->jh, j * sizeof(zeronp_float));
			memcpy(hisc + (iter + 1), w->ch, j * sizeof(zeronp_float));
			zeronp_free(w->jh);
			zeronp_free(w->ch);
			w->jh = hisj;
			w->ch = hisc;
			iter += j;
			continue;
		}
		*/
		// if (stgs->rs) {
		//	//FOR TEST RANDOM SAMPLING ONLY
		//	w->alm_crit = 1;
		// }
		if (w->nc && stgs->rescue &&
			w->const_time > 5 * (w->n - 2 * w->nec) && w->best_fea_con <= stgs->tol_con)
		{
			break;
		}
		// && stgs->delta <= MAX(stgs->tol,stgs->delta_end))

		if ((w->cons_nm1_orgin <= stgs->tol_con && w->obj_dif <= stgs->tol) && stgs->delta <= MAX(stgs->tol, stgs->delta_end) || (stgs->drsom && w->radius <= stgs->tol / 10) || w->exit)
		{
			if (w->restart < stgs->re_time && (w->alm_crit > stgs->tol_restart || (w->exit != 0) || (stgs->drsom && w->radius > stgs->tol / 10)) && w->exit != 1)
			{
				// Restart from the best point or feasible point.
				restart(w, stgs, iter, delta0);
			}
			else
			{
				i++;
				break;
			}
		}

		if (input->stgs->verbose)
		{
			printf("ZeroNP--> Iteration %d: obj = %.4e, infeasibility = %.4e\n", i + 1, w->bestobj, w->bestcon);
		}
	}
	if (w->bestobj != INFINITY)
	{
		memcpy(w->p, w->bestp, n * sizeof(zeronp_float));
		memcpy(w->l, w->bestl, MAX(nc, 1) * sizeof(zeronp_float));
		copyZERONPCost(w->ob, w->bestob);
		w->cons_nm1_orgin = w->bestcon;
	}
	else
	{
		memcpy(w->p, w->best_fea_p, n * sizeof(zeronp_float));
		memcpy(w->l, w->best_fea_l, MAX(nc, 1) * sizeof(zeronp_float));
		copyZERONPCost(w->ob, w->best_fea_ob);
		w->cons_nm1_orgin = w->best_fea_con;
		w->bestcon = w->best_fea_con;
	}
	/*
	if (w->nec && ( isnan(w->cons_nm1) || w->cons_nm1 > 10 * stgs->tol_con) && stgs->max_iter_rescue) {
		stgs->delta = delta0;

		zeronp_float* hisj = (zeronp_float*)zeronp_malloc(2 * (stgs->max_iter+1) * sizeof(zeronp_float));
		zeronp_float* hisc = (zeronp_float*)zeronp_malloc(2 * (stgs->max_iter + 1) * sizeof(zeronp_float));
		memcpy(hisj, w->jh, (iter+1) * sizeof(zeronp_float));
		memcpy(hisc, w->ch, (iter+1) * sizeof(zeronp_float));
		w = rescue(w, stgs, info);
		j = 0;
		while (w->jh[j] != INFINITY) {
			j++;
		}
		memcpy(hisj+iter+1, w->jh, j * sizeof(zeronp_float));
		memcpy(hisc+iter+1, w->ch, j * sizeof(zeronp_float));
		zeronp_free(w->jh);
		zeronp_free(w->ch);
		w->jh = hisj;
		w->ch = hisc;
		iter += j;
	}*/
	sol->p = (zeronp_float *)zeronp_malloc(n * sizeof(zeronp_float));
	sol->l = (zeronp_float *)zeronp_malloc(MAX(1, nc) * sizeof(zeronp_float));
	if (w->bestobj != INFINITY)
	{
		memcpy(sol->p, w->bestp, n * sizeof(zeronp_float));
		memcpy(sol->l, w->bestl, MAX(nc, 1) * sizeof(zeronp_float));
	}
	else
	{
		memcpy(sol->p, &w->p[w->nic], n * sizeof(zeronp_float));
		memcpy(sol->l, w->l, MAX(nc, 1) * sizeof(zeronp_float));
		w->bestcon = w->best_fea_con;
		w->bestobj = w->ob->obj;
	}

	if (w->cons_nm1_orgin <= stgs->tol_con && ((w->obj_dif <= stgs->tol && stgs->delta <= MAX(stgs->tol, stgs->delta_end)) || w->const_time > 5 * (w->n - 2 * w->nec)) || stgs->grad == 1)
	{
		sol->status = 1; // Success
		printf("ZeroNP--> Success! Completed in %d iterations\n", iter);
		printf("         The infeasibility is %e.\n", w->bestcon);
	}
	else
	{
		if (w->exit == 1)
		{
			sol->status = 0;
			printf("ZeroNP--> Exiting after maximum number of function evaluation. Tolerance not achieved.\n");
			printf("         The infeasibility is %e.\n", w->bestcon);
			printf("         ZERONP has restarted %d times.\n", w->restart);
		}
		else if (w->exit == 2)
		{
			sol->status = -3;
			printf("ZeroNP--> Exiting because of unknown error. Tolerance not achieved.\n");
			printf("         The infeasibility is %e.\n", w->bestcon);
		}
		else if (w->cons_nm1_orgin > stgs->tol_con)
		{
			sol->status = -1; // Fail to find a feasible point.
			printf("ZeroNP--> Exiting after maximum number of iterations. Tolerance not achieved.\n");
			printf("         The infeasibility is %e. ZERONP has restarted %d times.\n", w->bestcon, w->restart);
		}
		else if (w->obj_dif > stgs->tol)
		{
			sol->status = -2; // Fail to converge
			printf("ZeroNP--> Exiting after maximum number of iterations. Tolerance of infeasibility achieved.\n");
			printf("         The infeasibility is %e. ZERONP has restarted %d times.\n", w->bestcon, w->restart);
			printf("         ZERONP fails to converge.\n");
		}
	}

	sol->iter = iter;

	sol->ic = (zeronp_float *)zeronp_malloc(MAX(nic, 1) * sizeof(zeronp_float));
	if (nic == 0)
	{
		*sol->ic = 0;
	}
	else
	{
		memcpy(sol->ic, w->p, MAX(nic, 1) * sizeof(zeronp_float));
	}

	sol->best_fea_p = (zeronp_float *)zeronp_malloc((n) * sizeof(zeronp_float));
	memcpy(sol->best_fea_p, w->best_fea_p, (n) * sizeof(zeronp_float));

	sol->count_h = (zeronp_float *)zeronp_malloc((iter + 1) * sizeof(zeronp_float));
	memcpy(sol->count_h, w->count_h, (iter + 1) * sizeof(zeronp_float));

	sol->jh = (zeronp_float *)zeronp_malloc((iter + 1) * sizeof(zeronp_float));
	memcpy(sol->jh, w->jh, (iter + 1) * sizeof(zeronp_float));

	sol->ch = (zeronp_float *)zeronp_malloc((iter + 1) * sizeof(zeronp_float));
	memcpy(sol->ch, w->ch, (iter + 1) * sizeof(zeronp_float));

	if (stgs->bfgs)
	{
		sol->h = (zeronp_float *)zeronp_malloc((nic + n) * (nic + n) * sizeof(zeronp_float));
		memcpy(sol->h, w->h, (nic + n) * (nic + n) * sizeof(zeronp_float));
	}
	else
	{
		sol->h = (zeronp_float *)zeronp_malloc(1 * sizeof(zeronp_float));
		*sol->h = 1;
	}

	sol->obj = w->bestobj;

	sol->count_cost = w->count_cost;

	sol->count_grad = w->count_grad;

	sol->count_hess = w->count_hess;

	sol->constraint = w->bestcon;

	sol->restart_time = w->restart;

	return w;
}

zeronp_int free_cost(ZERONPCost *cost)
{
	if (cost)
	// if (0)
	{
		if (cost->ec)
		{
			zeronp_free(cost->ec);
		}
		if (cost->ic)
		{
			zeronp_free(cost->ic);
		}
		if (cost->cost)
		{
			cost->cost = ZERONP_NULL;
		}
		if (cost->grad)
		{
			cost->grad = ZERONP_NULL;
		}
		if (cost->hess)
		{
			cost->hess = ZERONP_NULL;
		}
		zeronp_free(cost);
	}

	return 0;
}

zeronp_int free_constraint(ZERONPConstraint *cnstr)
{
	if (cnstr)
	{
		if (cnstr->il)
		{
			zeronp_free(cnstr->il);
		}
		if (cnstr->iu)
		{
			zeronp_free(cnstr->iu);
		}
		if (cnstr->pl)
		{
			zeronp_free(cnstr->pl);
		}
		if (cnstr->pu)
		{
			zeronp_free(cnstr->pu);
		}
		if (cnstr->Ipc)
		{
			zeronp_free(cnstr->Ipc);
		}
		if (cnstr->Ipb)
		{
			zeronp_free(cnstr->Ipb);
		}
		zeronp_free(cnstr);
	}

	return 0;
}

zeronp_int free_work(ZERONPWork *w)
{
	if (w)
	{
		if (w->ob)
		{
			free_cost(w->ob);
		}
		if (w->best_fea_ob)
		{
			free_cost(w->best_fea_ob);
		}
		if (w->bestob)
		{
			free_cost(w->bestob);
		}
		if (w->pb)
		{
			free_constraint(w->pb);
		}
		if (w->h)
		{
			zeronp_free(w->h);
		}
		if (w->constraint)
		{
			zeronp_free(w->constraint);
		}
		if (w->count_h)
		{
			zeronp_free(w->count_h);
		}
		if (w->jh)
		{
			zeronp_free(w->jh);
		}
		if (w->ch)
		{
			zeronp_free(w->ch);
		}
		if (w->p)
		{
			zeronp_free(w->p);
		}
		if (w->p_old)
		{
			zeronp_free(w->p_old);
		}
		if (w->best_fea_p)
		{
			zeronp_free(w->best_fea_p);
		}
		if (w->l)
		{
			zeronp_free(w->l);
		}
		if (w->best_fea_l)
		{
			zeronp_free(w->best_fea_l);
		}
		if (w->bestp)
		{
			zeronp_free(w->bestp);
		}
		if (w->bestl)
		{
			zeronp_free(w->bestl);
		}
		zeronp_free(w);
	}

	return 0;
}

zeronp_int free_input(ZERONPIput *input)
{
	if (input)
	{
		// already freed in ZERONPWork
		// if(input->cnstr){
		//    free_constraint(input->cnstr);
		//}
		if (input->stgs)
		{
			zeronp_free(input->stgs);
		}
		if (input->h)
		{
			// free_hessian(input->h);
			zeronp_free(input->h);
		}
		if (input->l)
		{
			zeronp_free(input->l);
		}
		zeronp_free(input);
	}

	return 0;
}

zeronp_int ZERONP(finish)(ZERONPWork *w, ZERONPIput *input)
{
	if (w)
	{
		free_work(w);
	}
	if (input)
	{
		free_input(input);
	}

	return 0;
}

zeronp_int ZERONP(main)(
	ZERONPIput *input,
	ZERONPCost *cost,
	zeronp_float *ib0_p,
	ZERONPSol *sol,
	ZERONPInfo *info)
{
	zeronp_int status;
	info->qpsolver_time = 0;

	ZERONPWork *w = ZERONP(init)(input, cost, ib0_p);

	if (w)
	{
		w = ZERONP(solve)(w, input, ib0_p, sol, info);
		ZERONP(finish)
		(w, input);
		cost = ZERONP_NULL;
	}

	return 0;
}
