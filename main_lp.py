"""
LP Solver CLI  (uses the QP active set solver with P=0)
"""

from solver_qp import parse_qp, solve_qp

if __name__ == "__main__":
    print("    Welcome to the LP solver!")

    f0_str = input("""    Please enter the objective function (must be linear)...
    example... 2*x1 + x2
        objective function...""")

    print("""    Please enter inequalities (must be linear, RHS = 0), when done press enter...
    example... x1 + x2 - 4""")
    inequalities = []
    while True:
        s = input("        inequality...")
        if s == "":
            break
        inequalities.append(s)

    print("""    Please enter equalities (must be linear), when done press enter...
    example... x1 + x2 == 3""")
    equalities = []
    while True:
        s = input("        equality...")
        if s == "":
            break
        equalities.append(s)

    print("\n    Parsing problem...")
    P, q, c, A, b, C, d, syms = parse_qp(f0_str, inequalities, equalities)
    print("    Variables detected:", [str(s) for s in syms])

    print("    Solving...")
    x, nu, lam, active = solve_qp(P, q, A, b, C, d)

    obj_val = q @ x + c   # P=0 so no quadratic term

    print("\n    Solution:")
    for i, s in enumerate(syms):
        print(f"        {s} = {x[i]:.6f}")
    print(f"\n    Objective value: {obj_val:.6f}")

    if inequalities:
        print("\n    Shadow prices (inequality duals):")
        for i, ineq in enumerate(inequalities):
            status = "active" if i in active else "inactive"
            print(f"        {ineq} <= 0 :  lambda = {lam[i]:.6f}  ({status})")

    if equalities:
        print("\n    Shadow prices (equality duals):")
        for i, eq in enumerate(equalities):
            print(f"        {eq} :  nu = {nu[i]:.6f}")
