Last update: 6/25/2026.

The main solver uses Netwon's method to solve the KKT system.
1) It can solve convex problems with nonempty interior globally ie f(x*) <= f(x) for all x in domain
2) It can solve nonconvex problems locally ie ∃e s.t. f(x*) <= f(x) for x in (x*-e,x*+e)

The lpqp solver solves QPs and LPs by solving a sequence of KKT systems that treat inequalities as equalities as needed.
1) Can only solve LPs and QPs since it assumes benefits of their particular duality theory.

By convention, all optimizations are minimization (wlog since inf f = -sup (-f) )
