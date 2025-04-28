% This program use PDFO to solve the tumor growth problem
% Input: initiate time and drug dosage
% Ouput: optimal time and drug dosage
clear cost
nq = 4;
vmax = 1.1;
vcum = 65;
tmax = 200;
e = tmax/(nq);
x0 = ones(1,nq)*tmax/2;
x = [x0,0.5*ones(1,nq)];
opt.rhoend = 1e-8;
restart = 1;
for i = 1:restart
    tic
    [x, fx, exitflag, output] = pdfo(@(x)fun(x,vmax,vcum,tmax),x',[-eye(2*nq);eye(2*nq)],[zeros(2*nq,1);tmax*ones(nq,1);ones(nq,1)],@(x)const(x,vmax,vcum,tmax),opt);
    toc
end
% cost(x,inf)

function f = fun(x,vmax,vcum,tmax)
    f = cost(x,1,vmax,vcum,tmax);
    f = f(1);
end
function [iq,eq] = const(x,vmax,vcum,tmax)
    iq = cost(x,2,vmax,vcum,tmax);
    iq = iq(2:length(iq));
    eq = [];
    iq = [-iq;iq - [vmax;vcum]];
end