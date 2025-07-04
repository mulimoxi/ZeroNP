
p0 =  [5.54,4.4,12.02,11.82,.702,.852]';
t = 0;
con = 0;
rep = 1;
f = 0;
count = 0;
param = struct('display_degree','0','bb_output_type','OBJ PB PB PB PB','max_bb_eval',num2str(500*length(p0)),'min_poll_size','1e-4');
lb = zeros(6,1);
ub = inf*ones(6,1);
for i = 1:rep
    t1 = tic;
    [x,fval,hinf,exit_status,nfeval] = nomad(@fun,p0,lb,ub,param);
    t = t+ toc(t1);
    f = f + fval;
    count = count + nfeval;
    con = con + constraint(x);
end
fprintf("time = %e,count = %d,obj = %e,con = %e\n",t/rep,count/rep,f/rep,con/rep)
function f = fun(x)
    f = cost(x);
    f = [f;-f(2:length(f))];
end
function con = constraint(x)
    [iq,eq] = const(x);
    con = norm(eq)^2;
    con = con + norm(max(iq,0))^2;
    con = sqrt(con);
end
function [eq,iq] = const(x)
    eq = cost(x);
    eq = -eq(2:length(eq));
    iq = [];
end