Last update: 7/27/2026.

I am currently writing an optimization textbook and this library will host all of the algorithms in it. I am expecting the book to be done be EoQ3/BoQ4 '26.

The main file uses three solvers which use different methodologies** to come to solutions.
1) It can solve convex problems globally
2) It can solve nonconvex problems globally (functions must be at least Lipschitz - your functions probably are)

The lpqp solver solves QPs and LPs by solving a sequence of KKT systems that treat inequalities as equalities as needed.
1) Can only solve LPs and QPs since it assumes benefits of their particular duality theory.

By convention, all optimizations are minimization.




** the first solver uses the interior point method introduced in the Boyd book which is the first one I learned,
the second solver is one I concieved of myself but is pre-existing and known as an active set method, the third
solver I also concieved of, it is also pre-existing and called a feasible direction algorithm. I developed the
algorithms in decreasing order of cognitive load. I think the first one is hardest to understand, the second the second most
complicated, the last the easiest possible to understand (although presupposes LP - so it's a toss up for the reader).
