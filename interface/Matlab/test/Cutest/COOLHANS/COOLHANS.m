% This program solve problem 11 in Hock and Schittkowski test problem
% function [f,constraint, count,t] = P11(op,rep)
%% P11

prob.p0 = 0*ones(9,1);
% prob.ib0 = ones(1,1);
% prob.ibl = zeros(1,1);

op.ls_way = 1;
op.min_iter = 1;
op.noise = 1;
op.tol = 1e-4;
op.ls_time = 10;
op.tol_con = 1e-4;
% op.re_time = 10;
rep = 1;
% op.tol_restart = 1;
op.qpsolver = 1;
op.rho = 1;
op.max_iter = 50;
% op.re_time = 10;
fun.cost = @(x,nfeval)cost_para(x,nfeval,9);
% fun.grad = @grad;
% fun.hess = @hess;

f = 0;
constraint = 0;
count = 0;t = 0;
s = 0;rng(1);
for i = 1:rep   
    t1 = tic;
    info= ZERONP(prob,op,fun);
    t = t + toc(t1);
%     if info.constraint<= 1e-3 
        s = s +1;
        f = f +  info.obj;
        constraint = constraint + info.constraint;
        count = count + info.count_cost;
%     end
   
    clear cost
end
t = t/rep;
f = f/s;
count = count/s;
constraint = constraint/s;
fprintf("f average = %e, con = %e,count = %f,time = %e\n",f,constraint,count,t);
%cost(info.p,inf);
% end

function f = cost_para(x,nfeval,nc)
    f = zeros(nc+1,nfeval);
    for i = 1:nfeval
        f(:,i) = cost(x(:,i));
    end
end
