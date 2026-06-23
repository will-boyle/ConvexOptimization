Last update: 6/23/2026.

This library uses Netwon's method to solve the KKT system. 

1) It can solve convex problems with nonempty interior globally ie f(x*) <= f(x) for all x in domain
2) It can solve nonconvex problems locally ie ∃e s.t. f(x*) <= f(x) for x in (x*-e,x*+e)

By convention, all optimizations are minimization (wlog since inf f = -sup (-f) )
