function f = cost(x,par)    
    persistent calltimes;
    persistent finaltime;
    persistent tag;
    tol = 1e-2;

%% P28
optx = [0.5, -0.5 , 0.5]';
fopt = 0;
f = [(x(1)+x(2))^2+(x(2)+x(3))^2,x(1)+2*x(2)+3*x(3)-1];

 f = f';
    if exist('par','var') && par==inf
        fprintf('COST-->  Final result: F(x) = %e\n',f(1));
        fprintf('COST-->  The algorithm has called function for %d times\n',calltimes);
        if finaltime ~= inf
            fprintf('COST-->  The algorithm requires %d evaluations to get an optimal solution\n\n ',finaltime);
        else
            fprintf('COST-->  Fail to get an optimal solution\n');
        end
        return;
    end
    if isempty(calltimes)
           calltimes = 0;
           tag = 1;
    end
    calltimes = calltimes+1;
    if isempty(finaltime)
           finaltime = inf;
    end
    if abs(f(1)-fopt)/max(abs(fopt),1) <= tol && norm(f(2:length(f)))<tol  && tag == 1
        finaltime = calltimes;
        tag = 0;
    end
     f = f.*(1 + 1e-4*randn(2,1));
end