import numpy as np
from scipy.optimize import linprog

'''
Feasible Direction Method (Zoutendijk, 1960).

Problem:
    min  f_0(x)
    s.t. f_i(x) <= 0   (inequality constraints)
         Ax = b         (equality constraints)

=========================================================
CORE IDEA
=========================================================

At every iteration we ask: which direction d should we step?

We want d to satisfy two things simultaneously:
    1. d decreases (or at least does not increase) the quantity we care about.
    2. d keeps us within (or moves us toward) the feasible region.

The constraint on d from the equality Ax = b is: A d = 0
(moving along d must not violate the equalities to first order).

The constraint on d from an inequality f_i(x) <= 0 is: grad(f_i)^T d <= 0
(moving along d must not increase f_i to first order).

Both are LINEAR constraints on d. Adding a box constraint -1 <= d_k <= 1
(the L-infinity unit ball) keeps d bounded without introducing any quadratic
terms, so each direction-finding subproblem is a pure LP.

=========================================================
PHASE 1 — FIND A FEASIBLE POINT
=========================================================

Define the feasibility measure:

    phi(x) = sum_i max(f_i(x), 0)

phi(x) = 0  iff  x is feasible.  Its subgradient is:

    grad phi(x) = sum_{i: f_i(x) > 0} grad f_i(x)

We find the direction d that most steeply decreases phi by solving:

    min   grad_phi(x)^T d
    s.t.  A d = 0              (stay on equality surface)
          -1 <= d_k <= 1

If the LP optimal value < 0: d decreases phi, so take a step (backtracking
line search on phi) and repeat.

If the LP optimal value >= 0 and phi(x) > 0: no descent direction exists for
phi — the problem is infeasible.

If phi(x) = 0: feasible. Move to Phase 2.

=========================================================
PHASE 2 — MINIMIZE OBJECTIVE WHILE STAYING FEASIBLE
=========================================================

Find the direction that most decreases f_0 while keeping ALL constraints
(active and inactive) satisfied to first order:

    min   grad f_0(x)^T d
    s.t.  f_i(x) + grad f_i(x)^T d <= 0   for ALL i
          A d = 0
          -1 <= d_k <= 1

The constraint "f_i(x) + grad f_i(x)^T d <= 0" is the linearization of
feasibility around x. For active constraints (f_i = 0) it reduces to
grad(f_i)^T d <= 0. For inactive constraints (f_i < 0) it allows d to
increase f_i by up to |f_i(x)| — the current slack — before going infeasible.
This is why the method remains correct at strictly interior points.

If the LP optimal value < 0: d is a descent direction that keeps all
linearized constraints feasible. Take a step with backtracking line search
on f_0 (also checking true feasibility), and repeat.

If the LP optimal value >= 0: no improving linearized-feasible direction
exists. By LP duality this means the KKT conditions hold — we are optimal.
'''


def _lp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, n=None):
    """Solve min c^T d s.t. inequalities, equalities, -1 <= d_k <= 1."""
    bounds = [(-1.0, 1.0)] * len(c)
    kwargs = {'bounds': bounds, 'method': 'highs'}
    if A_ub is not None and A_ub.shape[0] > 0:
        kwargs['A_ub'], kwargs['b_ub'] = A_ub, b_ub
    if A_eq is not None and A_eq.shape[0] > 0:
        kwargs['A_eq'], kwargs['b_eq'] = A_eq, b_eq
    return linprog(c, **kwargs)


def _phase1_direction(x, f, Df, A):
    """
    Direction that most decreases phi(x) = sum max(f_i(x), 0).
    Returns (d, slope) where slope = grad_phi^T d  (negative = descent).
    """
    fx      = f(x)
    mask    = fx > 0
    if not np.any(mask):
        return np.zeros(len(x)), 0.0

    grad_phi = Df(x)[mask].sum(axis=0)          # sum of violated gradients

    A_eq = A if A.shape[0] > 0 else None
    b_eq = np.zeros(A.shape[0]) if A.shape[0] > 0 else None

    res = _lp(grad_phi, A_eq=A_eq, b_eq=b_eq)
    if res.status not in (0, 1):
        return np.zeros(len(x)), 0.0

    return np.array(res.x), float(res.fun)


def _phase2_direction(x, fx, grad_f0, Df, A):
    """
    Direction that most decreases f_0 subject to the linearized feasibility
    condition f_i(x) + grad(f_i)^T d <= 0 for all i.
    Returns (d, slope) where slope = grad_f0^T d  (negative = descent).
    """
    q   = grad_f0(x)
    Dfx = Df(x)           # (m, n)
    m   = Dfx.shape[0]

    # RHS: linearized feasibility — active constraints get 0, inactive get slack
    A_ub = Dfx if m > 0 else None
    b_ub = -fx  if m > 0 else None   # f_i(x) + Df_i*d <= 0  ↔  Df_i*d <= -f_i(x)

    A_eq = A if A.shape[0] > 0 else None
    b_eq = np.zeros(A.shape[0]) if A.shape[0] > 0 else None

    res = _lp(q, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
    if res.status not in (0, 1):
        return np.zeros(len(q)), 0.0

    return np.array(res.x), float(res.fun)


def _phi(x, f):
    return float(np.sum(np.maximum(f(x), 0.0)))


def solve3(x0, f0, grad_f0, f, Df, A, b, eps=1e-6, max_iter=500,
           alpha=0.01, beta=0.5):
    """
    Feasible direction method for convex programs.

    Args:
        x0      : (n,)   starting point (need not be feasible)
        f0      : callable x -> float    objective value
        grad_f0 : callable x -> (n,)     gradient of objective
        f       : callable x -> (m,)     inequality constraint values
        Df      : callable x -> (m,n)    Jacobian of inequalities
        A       : (p,n)  equality constraint matrix
        b       : (p,)   equality constraint RHS
        eps     : convergence tolerance
        max_iter: max iterations per phase
        alpha   : Armijo sufficient-decrease parameter
        beta    : backtracking reduction factor

    Returns:
        x : (n,) solution
    """
    x = x0.copy()

    # ------------------------------------------------------------------
    # Phase 1: drive phi(x) = sum max(f_i(x), 0) to zero
    # ------------------------------------------------------------------
    for _ in range(max_iter):
        phi = _phi(x, f)
        if phi < eps:
            break

        d, slope = _phase1_direction(x, f, Df, A)

        if slope >= -eps:
            raise ValueError(
                "Phase 1: no descent direction for phi. "
                "Problem is likely infeasible."
            )

        # Backtracking line search on phi (Armijo condition)
        s = 1.0
        while s > 1e-12 and _phi(x + s * d, f) > phi + alpha * s * slope:
            s *= beta

        x = x + s * d
    else:
        raise ValueError("Phase 1 did not converge within max_iter iterations.")

    # ------------------------------------------------------------------
    # Phase 2: minimize f_0 using linearized-feasibility direction LP
    # ------------------------------------------------------------------
    for _ in range(max_iter):
        fx_cur = f(x)
        d, slope = _phase2_direction(x, fx_cur, grad_f0, Df, A)

        if slope >= -eps:
            break   # no improving feasible direction — KKT conditions hold

        # Backtracking line search on f_0, also checking feasibility
        f0x   = f0(x)
        s     = 1.0
        while s > 1e-12:
            x_new = x + s * d
            if (np.all(f(x_new) <= eps) and
                    f0(x_new) <= f0x + alpha * s * slope):
                break
            s *= beta

        if s <= 1e-12:
            break   # step vanished — converged

        x = x + s * d

    return x
