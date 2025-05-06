#  ZeroNP README
## Basic Introduction
ZeroNP solves the general derivative-free nonlinear constrained problems of the form:

$$
\begin{aligned}
    \min\_x\ &f(x) \\ 
    \text{s.t.} \ & g(x) = 0, \\
                  & l\_h\leq h(x)\leq u\_h, \\
                  & l\_x \leq x \leq u\_x,
\end{aligned}
$$

where $f(x),g(x),h(x)$ are smooth functions.
