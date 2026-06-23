"""
Convex Optimization Solver CLI
"""

import numpy as np
import sympy as sp
from solver import parse, phase1, solve

if __name__ == "__main__":
    print("    Welcome to the convex optimization solver!")

    f_0str = input("""    Please enter the objective function...
    example... x1**2 + x2**2
        objective function...""")

    print("""    Please enter inequalities, when done press enter...
    the convention is that the RHS is zero and all inequalities are <=.
    example... x1 + 1""")
    inequalities = []
    while True:
        f_istr = input("        inequality...")
        if f_istr == "":
            break
        inequalities.append(f_istr)

    print("""    Please enter equalities, when done press enter...
    example... x1 + x2 == 5""")
    equalities = []
    while True:
        h_istr = input("        equality...")
        if h_istr == "":
            break
        equalities.append(h_istr)

    print("\n    Parsing problem...")
    grad_f0, hess_f0, f, Df, Hf, A, b, syms = parse(f_0str, inequalities, equalities)

    print("    Variables detected:", [str(s) for s in syms])

    # Build an initial point satisfying Ax = b via least-squares, then run Phase I
    n = len(syms)
    if A.shape[0] > 0:
        x_init = np.linalg.lstsq(A, b, rcond=None)[0]
    else:
        x_init = np.zeros(n)

    print("    Running Phase I to find a feasible starting point...")
    try:
        x0 = phase1(f, Df, Hf, A, b, x_init)
    except ValueError as e:
        print(f"\n    Error: {e}")
        exit(1)

    print("    Feasible point found. Running solver...")
    x, lam, nu = solve(x0, mu=10.0, grad_f0=grad_f0, hess_f0=hess_f0,
                       f=f, Df=Df, Hf=Hf, A=A, b=b)

    f0_fn  = sp.lambdify(syms, sp.sympify(f_0str), 'numpy')
    obj_val = float(f0_fn(*x))

    print("\n    Solution:")
    for i, s in enumerate(syms):
        print(f"        {s} = {x[i]:.6f}")
    print(f"\n    Objective value: {obj_val:.6f}")
