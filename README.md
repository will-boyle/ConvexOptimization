Last update: 1/4/2025.

This library can be used to solve constrained optimization problems. 

What kind of problems can it solve?


1) Linear Programs

min cx

st:
Ax <= b

x >= 0

2) Integer Programs

min cx

st 
Ax <= b

x >= 0
    
x_i is an integer for all i

3) Convex Programs

min f(x) 

st 
f_i(x) <= 0, i in {1,...,m}

Ax = b
