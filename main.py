"""
Convex Optimization Solver CLI
"""

import numpy as np
import sympy as sp
from solver  import parse, phase1, solve
from solver2 import solve2
from solver3 import solve3

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
    example... x1 - x2 == 0""")
    equalities = []
    while True:
        h_istr = input("        equality...")
        if h_istr == "":
            break
        equalities.append(h_istr)

    print("\n    Parsing problem...")
    grad_f0, hess_f0, f, Df, Hf, A, b, syms = parse(f_0str, inequalities, equalities)

    print("    Variables detected:", [str(s) for s in syms])

    n = len(syms)
    if A.shape[0] > 0:
        x_init = np.linalg.lstsq(A, b, rcond=None)[0]
    else:
        x_init = np.zeros(n)

    solver_choice = input("""    Choose solver:
        1. Interior Point      (solver.py  — barrier method, Phase I required)
        2. Active Set          (solver2.py — Newton on KKT, tracks active constraints)
        3. Feasible Direction  (solver3.py — LP subproblems, first-order only)
        choice (1, 2, or 3, default 1)... """).strip()

    f0_fn = sp.lambdify(syms, sp.sympify(f_0str), 'numpy')
    lam   = None

    if solver_choice == "2":
        print("    Running Active Set solver...")
        x, lam, nu = solve2(x_init, grad_f0=grad_f0, hess_f0=hess_f0,
                            f=f, Df=Df, Hf=Hf, A=A, b=b)
    elif solver_choice == "3":
        print("    Running Feasible Direction solver...")
        try:
            x = solve3(x_init, lambda v: float(f0_fn(*v)), grad_f0, f, Df, A, b)
        except ValueError as e:
            print(f"\n    Error: {e}")
            exit(1)
    else:
        print("    Running Phase I to find a feasible starting point...")
        try:
            x0 = phase1(f, Df, Hf, A, b, x_init)
        except ValueError as e:
            print(f"\n    Error: {e}")
            exit(1)
        print("    Feasible point found. Running solver...")
        x, lam, nu = solve(x0, mu=10.0, grad_f0=grad_f0, hess_f0=hess_f0,
                           f=f, Df=Df, Hf=Hf, A=A, b=b)

    obj_val = float(f0_fn(*x))

    print("\n    Solution:")
    for i, s in enumerate(syms):
        print(f"        {s} = {x[i]:.6f}")
    print(f"\n    Objective value: {obj_val:.6f}")
    if lam is not None and inequalities:
        print("\n    Shadow prices (inequality dual variables):")
        for i, ineq in enumerate(inequalities):
            print(f"        {ineq} <= 0 :  lambda = {lam[i]:.6f}")
