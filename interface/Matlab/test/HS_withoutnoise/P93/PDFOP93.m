% This program solve problem 93 in Hock and Schittkowski test problem
p0 = [5.54,4.4,12.02,11.82,.702,.852]';
opt.rhoend = 1e-4;
t = 0;
f = 0;
con = 0;
rep = 1;
count = 0;
for i = 1:rep
    t1 = tic;
    [x,fx,oh,output] = pdfo(@fun,p0,-eye(6),zeros(6,1),@const,opt);
    t = t+ toc(t1);
 con = con + constraint(x);
    f = f + fx;
    count = count + output.funcCount;
end
fprintf("time = %e,count = %f,obj = %e,con = %e\n",t/rep,count/rep,f/rep,con/rep);

function f = fun(x)
    f = cost(x);
    f = f(1);
end
function con = constraint(x)
    [iq,eq] = const(x);
    con = norm(eq)^2;
    con = con + norm(max(iq,0))^2;
    con = sqrt(con);
end
function [iq,eq] = const(x)
    iq = cost(x);
    iq = iq(2:length(iq));
    iq = -iq;
    eq = [];
end