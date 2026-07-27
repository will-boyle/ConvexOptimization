"""
Convex Optimization Solver CLI
"""

import numpy as np
import sympy as sp
from solver  import parse, phase1, solve
from solver2 import solve2
from solver3 import solve3
from solver4 import parse_dc, solve_dc
from solver5 import parse_abb, solve_abb
from lipschitz_solver import parse_lipschitz, solve_lipschitz, estimate_L

if __name__ == "__main__":
    print("    Welcome to the convex optimization solver!")

    problem_type = input("""    Problem type:
        1. Convex program      (minimize f0(x)         s.t. fi(x) <= 0, Ax = b)
        2. D.C. program        (minimize f0(x)-g0(x)   s.t. fi(x)-gi(x) <= 0)
        3. General nonconvex   (minimize f0(x)         s.t. fi(x) <= 0, lb <= x <= ub)  [requires C2]
        4. Lipschitz           (minimize f0(x)         s.t. fi(x) <= 0, lb <= x <= ub)  [requires Lipschitz only]
        choice (1, 2, 3, or 4, default 1)... """).strip()

    # ─────────────────────────────── D.C. BRANCH ───────────────────────────
    if problem_type == "2":
        print("""
    D.C. (Difference of Convex) Programming
    Objective:   minimize f0(x) - g0(x)
    Constraints: fi(x) - gi(x) <= 0  for each i
    Both f0, g0, fi, gi must be convex.
""")
        f0_str = input("    Convex part of objective f0(x)...  example: x1**2 + x2**2\n"
                       "        f0 = ").strip()
        g0_str = input("    Concave part of objective g0(x) (subtracted)...  example: x1\n"
                       "        g0 = ").strip()

        print("""
    Enter D.C. inequality constraints  fi(x) - gi(x) <= 0.
    Press Enter with an empty fi to finish.""")
        dc_ineqs = []
        while True:
            fi_str = input("        fi (convex, or Enter to finish): ").strip()
            if fi_str == "":
                break
            gi_str = input("        gi (concave):                    ").strip()
            dc_ineqs.append((fi_str, gi_str))

        print("""
    Enter bounding box for the variables.
    The initial simplex will cover this box — make it tight for faster convergence.""")
        lb_vals, ub_vals = [], []
        # Infer number of variables by parsing first
        all_strs = [f0_str, g0_str] + [s for pair in dc_ineqs for s in pair]
        all_syms = sorted(
            set().union(*[sp.sympify(s).free_symbols for s in all_strs]),
            key=lambda s: s.name,
        )
        print(f"    Variables detected: {[str(s) for s in all_syms]}")
        for s in all_syms:
            lb_val = float(input(f"        lower bound for {s}: ").strip())
            ub_val = float(input(f"        upper bound for {s}: ").strip())
            lb_vals.append(lb_val)
            ub_vals.append(ub_val)

        tol_str = input("    Convergence tolerance (default 1e-4): ").strip()
        tol     = float(tol_str) if tol_str else 1e-4
        iter_str = input("    Max B&B iterations (default 500): ").strip()
        max_iter = int(iter_str) if iter_str else 500

        print("\n    Parsing D.C. problem...")
        (f0, g0, grad_f0, grad_g0,
         fi_fns, gi_fns, grad_fi_fns, grad_gi_fns,
         syms) = parse_dc(f0_str, g0_str, dc_ineqs)

        print("    Running D.C. branch-and-bound solver...\n")
        x_best, U = solve_dc(
            f0, g0, grad_f0, grad_g0,
            fi_fns, gi_fns, grad_fi_fns, grad_gi_fns,
            lb_vals, ub_vals,
            tol=tol, max_iter=max_iter, verbose=True,
        )

        print("\n    Solution:")
        if x_best is None:
            print("    No feasible point found in the given bounding box.")
        else:
            for i, s in enumerate(syms):
                print(f"        {s} = {x_best[i]:.6f}")
            print(f"\n    Objective f0 - g0 = {U:.6f}")

    # ─────────────────────────────── αBB BRANCH ───────────────────────────
    elif problem_type == "3":
        print("""
    αBB Global Optimizer
    Objective:   minimize f0(x)
    Constraints: fi(x) <= 0  for each i  (optional)
    A bounding box [lb, ub] is required; all global optima must lie inside it.
""")
        f0_str = input("    Objective f0(x)...  example: x1**4 - 4*x1**2\n"
                       "        f0 = ").strip()

        print("""
    Enter inequality constraints  fi(x) <= 0.
    Press Enter with an empty fi to finish.""")
        ineqs = []
        while True:
            fi_str = input("        fi (or Enter to finish): ").strip()
            if fi_str == "":
                break
            ineqs.append(fi_str)

        print("\n    Parsing problem...")
        f0_fn, grad_f0_fn, hess_f0_fn, fi_fns, grad_fi_fns, hess_fi_fns, syms = parse_abb(f0_str, ineqs)
        print(f"    Variables detected: {[str(s) for s in syms]}")

        print("""
    Enter bounding box [lb, ub] for each variable.
    All global minima must lie inside this box.""")
        lb_vals, ub_vals = [], []
        for s in syms:
            lb_val = float(input(f"        lower bound for {s}: ").strip())
            ub_val = float(input(f"        upper bound for {s}: ").strip())
            lb_vals.append(lb_val)
            ub_vals.append(ub_val)

        tol_str  = input("    Convergence tolerance (default 1e-4): ").strip()
        tol      = float(tol_str) if tol_str else 1e-4
        iter_str = input("    Max B&B iterations (default 1000): ").strip()
        max_iter = int(iter_str) if iter_str else 1000

        print("\n    Running αBB solver...\n")
        x_best, U = solve_abb(
            f0_fn, grad_f0_fn, hess_f0_fn,
            fi_fns, grad_fi_fns, hess_fi_fns,
            lb_vals, ub_vals,
            tol=tol, max_iter=max_iter, verbose=True,
        )

        print("\n    Solution:")
        if x_best is None:
            print("    No feasible point found in the given bounding box.")
        else:
            for i, s in enumerate(syms):
                print(f"        {s} = {x_best[i]:.8f}")
            print(f"\n    Objective f0 = {U:.8f}")

    # ─────────────────────────────── LIPSCHITZ BRANCH ─────────────────────
    elif problem_type == "4":
        print("""
    Lipschitz Global Optimizer
    Objective:   minimize f0(x)              (must be Lipschitz continuous)
    Constraints: fi(x) <= 0  for each i     (optional, also Lipschitz)
    A bounding box [lb, ub] is required.
    Note: Lipschitz does NOT mean all continuous functions qualify.
          f(x) = sqrt(x) near 0 is continuous but not Lipschitz.
""")
        f0_str = input("    Objective f0(x)...  example: Abs(x1) + x2**2\n"
                       "        f0 = ").strip()

        print("""
    Enter inequality constraints  fi(x) <= 0.
    Press Enter with an empty fi to finish.""")
        ineqs = []
        while True:
            fi_str = input("        fi (or Enter to finish): ").strip()
            if fi_str == "":
                break
            ineqs.append(fi_str)

        print("\n    Parsing problem...")
        f0_fn, grad_f0_fn, fi_fns, grad_fi_fns, syms = parse_lipschitz(f0_str, ineqs)
        print(f"    Variables detected: {[str(s) for s in syms]}")

        print("""
    Enter bounding box [lb, ub] for each variable.""")
        lb_vals, ub_vals = [], []
        for s in syms:
            lb_val = float(input(f"        lower bound for {s}: ").strip())
            ub_val = float(input(f"        upper bound for {s}: ").strip())
            lb_vals.append(lb_val)
            ub_vals.append(ub_val)

        lb_arr = np.array(lb_vals)
        ub_arr = np.array(ub_vals)

        L_str = input("    Lipschitz constant L (or Enter to estimate from gradient): ").strip()
        if L_str:
            L_f = float(L_str)
            L_fi = None
        else:
            print("    Estimating Lipschitz constants from gradient at box corners...")
            L_f  = estimate_L(grad_f0_fn, lb_arr, ub_arr)
            L_fi = [estimate_L(gfi, lb_arr, ub_arr) for gfi in grad_fi_fns] if grad_fi_fns else None
            print(f"    L_f = {L_f:.4f}" +
                  (f"  L_fi = {[f'{l:.4f}' for l in L_fi]}" if L_fi else ""))

        tol_str  = input("    Convergence tolerance (default 1e-4): ").strip()
        tol      = float(tol_str) if tol_str else 1e-4
        iter_str = input("    Max B&B iterations (default 50000): ").strip()
        max_iter = int(iter_str) if iter_str else 50000

        print("\n    Running Lipschitz B&B solver...\n")
        x_best, U = solve_lipschitz(
            f0_fn, fi_fns, L_f, L_fi,
            lb=lb_arr, ub=ub_arr,
            tol=tol, max_iter=max_iter, verbose=True,
        )

        print("\n    Solution:")
        if x_best is None:
            print("    No feasible point found in the given bounding box.")
        else:
            for i, s in enumerate(syms):
                print(f"        {s} = {x_best[i]:.8f}")
            print(f"\n    Objective f0 = {U:.8f}")

    # ─────────────────────────────── CONVEX BRANCH ─────────────────────────
    else:
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
