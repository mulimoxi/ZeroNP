#pragma once
#ifndef ZERONP_H_GUARD
#define ZERONP_H_GUARD

#ifdef __cplusplus
extern "C"
{
#endif

#include "zeronp_glbopts.h"
#include <string.h>

      typedef struct ZERONP_WORK ZERONPWork;
      typedef struct ZERONP_COST ZERONPCost;
      typedef struct ZERONP_SETTINGS ZERONPSettings;
      typedef struct ZERONP_CONSTRAINT ZERONPConstraint;
      typedef struct ZERONP_INPUT ZERONPIput;
      typedef struct ZERONP_SOL ZERONPSol;
      typedef struct ZERONP_INFO ZERONPInfo;
      typedef struct ZERONP_PROB ZERONPProb;

      struct ZERONP_INFO
      {
            zeronp_float total_time;
            zeronp_float qpsolver_time;
            zeronp_float cost_time;
      };

      struct ZERONP_INPUT
      {
            zeronp_int n;
            ZERONPConstraint *cnstr;
            ZERONPSettings *stgs;
            zeronp_float *l;
            zeronp_float *h;
      };

      // define the lower and upper bounds of decision variable and inequality constrains
      struct ZERONP_CONSTRAINT
      {
            zeronp_int n;     // dimension of decision variable
            zeronp_int nic;   // number of inequality constrains
            zeronp_float *il; // lower bounds of inequality constrains, len of nic
            zeronp_float *iu; // upper bounds of inequality constrains, len of nic
            zeronp_float *pl; // lower bounds of decision variable, len of n
            zeronp_float *pu; // upper bounds of decision variable, len of n
            zeronp_int *Ipc;  // len of 2
            zeronp_int *Ipb;  // len of 2
      };

      // return of cost function
      struct ZERONP_COST
      {
            zeronp_int nec;   // number of equality constrains
            zeronp_int nic;   // number of inequality constrains
            zeronp_float obj; // objective of cost function
            zeronp_float *ec; // constraint function values of EC, len of nec
            zeronp_float *ic; // constraint function values of IC, len of nic

            void (*hess)(zeronp_float *h, zeronp_float *p, zeronp_int np, zeronp_int nheval, zeronp_int action); // function pointer to user-defined hess function
            void (*grad)(zeronp_float *g, zeronp_float *p, zeronp_int np, zeronp_int ngeval, zeronp_int action); // function pointer to user-defined gradient function
            void (*cost)(ZERONPCost **c, zeronp_float *p, zeronp_int np, zeronp_int nfeval, zeronp_int action);  // function pointer to user-defined cost function
      };
      struct ZERONP_WORK
      {
            zeronp_int n;   // dimension of decision variable
            zeronp_int nec; // number of equality constrains
            zeronp_int nic; // number of inequality constrains
            zeronp_int nc;  // number constrains

            ZERONPCost *ob;           // observation of cost function
            zeronp_float *constraint; // vector of constraint values
            zeronp_float obj_dif;     // the difference of the objective values between two consecutive iterations
            zeronp_float cons_nm1;    // NORM(CONSTRAINT) before a major iteration
            zeronp_float cons_nm2;    // NORM(CONSTRAINT) after a major iteration

            zeronp_float cons_nm1_orgin; // NORM(CONSTRAINT) before a major iteration
            zeronp_float cons_nm2_orgin; // NORM(CONSTRAINT) after a major iteration

            ZERONPConstraint *pb;

            zeronp_float j;
            zeronp_float *count_h;
            zeronp_float *jh;
            zeronp_float *ch;
            zeronp_float rho;
            zeronp_float pen_l1; // l1 penalty paramenter
            zeronp_float radius;

            zeronp_float mu;
            zeronp_float *p;
            zeronp_float *p_old;
            // zeronp_float *ib0;
            zeronp_float *l;
            zeronp_float *h; // Hessian matrix
            zeronp_float *bestp;
            zeronp_float *bestl;
            ZERONPCost *bestob;

            zeronp_float bestcon;
            zeronp_float bestobj;
            zeronp_float alm_crit; // use alm stop criterion to decide whether to restart

            zeronp_float *best_fea_p;
            zeronp_float best_fea_con;
            zeronp_float *best_fea_l;
            ZERONPCost *best_fea_ob;

            zeronp_int const_time;
            zeronp_int restart;

            zeronp_int count_cost; // record cost times in zeronp.c
            zeronp_int count_grad; // record gradient times in zeronp.c
            zeronp_int count_hess;
            zeronp_int exit;
      };

      struct ZERONP_SETTINGS
      {
            zeronp_float pen_l1;
            zeronp_float rho;
            zeronp_int max_iter;
            zeronp_int min_iter;
            zeronp_int max_iter_rescue;
            zeronp_int min_iter_rescue;
            zeronp_float delta;
            zeronp_float tol;
            zeronp_float tol_con;
            zeronp_int ls_time;
            zeronp_float tol_restart;
            zeronp_int re_time;
            zeronp_float delta_end;
            zeronp_int maxfev;
            zeronp_int noise;
            zeronp_int qpsolver;
            zeronp_float k_r;
            zeronp_float k_i;
            zeronp_float c_r;
            zeronp_float c_i;
            zeronp_int batchsize;
            zeronp_int hess;
            zeronp_int grad;
            zeronp_int rescue;
            zeronp_int ls_way; // 1 means bisection, 2 means non-monotonic
            zeronp_int bfgs;
            zeronp_int rs;
            zeronp_int cen_diff;
            zeronp_float gd_step;
            zeronp_int scale;
            zeronp_int drsom;
            zeronp_float h;          // estimate interval when noise == 2
            zeronp_float step_ratio; // balance of Byrd Step and Adaptive step
            zeronp_int verbose;
      };

      struct ZERONP_SOL
      {
            zeronp_int iter;
            zeronp_int status;
            zeronp_float *p;
            zeronp_float *ic;
            zeronp_float *count_h;
            zeronp_float *jh;
            zeronp_float *ch;
            zeronp_float *best_fea_p;
            zeronp_float *l;
            zeronp_float *h;
            zeronp_int count_cost;
            zeronp_int count_grad; // record gradient times in zeronp.c
            zeronp_int count_hess;
            zeronp_float constraint;
            zeronp_float obj;
            zeronp_int restart_time;
      };

      struct ZERONP_PROB
      {
            /* input data of C interface */
            zeronp_int np;
            zeronp_int nic;
            zeronp_int nec;
            zeronp_int nc;
            zeronp_int *Ipc;
            zeronp_int *Ipb;
            zeronp_float *ibl;
            zeronp_float *ibu;
            zeronp_float *pbl;
            zeronp_float *pbu;
            zeronp_float *ib0;
            zeronp_float *p0;
      };

      zeronp_int ZERONP(main)(
          ZERONPIput *input,
          ZERONPCost *cost,
          zeronp_float *ib0_p,
          ZERONPSol *sol,
          ZERONPInfo *info);

      zeronp_int free_cost(ZERONPCost *cost);

      zeronp_int subnp_qp(ZERONPWork *w, ZERONPSettings *stgs, ZERONPInfo *info);

#ifdef __cplusplus
}
#endif
#endif
