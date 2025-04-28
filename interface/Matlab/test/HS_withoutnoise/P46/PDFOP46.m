% This program solve problem 46 in Hock and Schittkowski test problem
clear cost
p0 = [.5*sqrt(2),1.75,.5,2,2]';
opt.rhoend = 1e-4;
t = 0;
f = 0;
con = 0;
rep = 50;
count = 0;
success = 0;
for i = 1:rep
    t1 = tic;
    [x,fx,oh,output] = pdfo(@fun,p0,@const,opt);
    t = t+ toc(t1);
    con = con + constraint(x);
    if constraint(x) <= 1e-3
        success = success+1;
    end
    f = f + fx;
    count = count + output.funcCount;
end
fprintf("time = %e,count = %d,obj = %e,con = %e\n",t/rep,count/rep,f/rep,con/rep);

function con = constraint(x)
    [iq,eq] = const(x);
    con = norm(eq)^2;
    con = con + norm(max(iq,0))^2;
    con = sqrt(con);
end
function f = fun(x)
    f = cost(x);
    f = f(1);
end
function [iq,eq] = const(x)
    eq = cost(x);
    eq = eq(2:length(eq));
    iq = [];
end