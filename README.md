Last update: 7/21/2026.

The main file can use three solvers which use different methodologies** to come to solutions.
1) It can solve convex problems with nonempty interior globally ie f(x*) <= f(x) for all x in domain
2) It can solve nonconvex problems locally ie ∃e s.t. f(x*) <= f(x) for x in (x*-e,x*+e)

The lpqp solver solves QPs and LPs by solving a sequence of KKT systems that treat inequalities as equalities as needed.
1) Can only solve LPs and QPs since it assumes benefits of their particular duality theory.

By convention, all optimizations are minimization (wlog since inf f = -sup (-f) )




** the first solver uses the interior point method introduced in the Boyd book which is the first one I learned,
the second solver is one I concieved of myself but is pre-existing and known as an active set method, the third
solver I also concieved of, I am not sure what you call it but AI tools told me it wasn't original either. I developed the
algorithms in decreasing order of cognitive load. I think the first one is hardest to understand, the second the second most
complicated, the last the easiest possible to understand (although presupposes LP - so it's a toss up for the reader).
