import numpy as np
import sympy as sp

'''
Code works by satisfying the KKT conditions using newton's method.

This is essentially the primal-dual method described in Convex Optimization - Boyd

for min f_0
st: f <= 0
    Ax = b


KKT condtions are: 
1) Grad(L(x,u,v)) = 0  (stationarity)
2) f <= 0, Ax = b      (primal feasibility)
3) u >= 0              (dual feasibility)
4) u @ f = 0           (complementary slackness) 

Note, except for dual feasibility @ the primal inequalities, each of these is an equation.

If we assume dual feasibility and inequalities hold, and if we let kkt(x,u,v) = (Grad(L(x,u,v)), u @ f, Ax - b)
then if kkt(x,u,v) = 0, we have solved the KKT conditions and hence have found the solution to the optimization.

Hence, what we want to do is find a root of the kkt function. We can do this using Newton's method which 
simply repeats using a taylor approximation to solve the equation as in....

kkt(t+h) ~ kkt(t) + Dkkt(t)h

so, if we have some initial value of the kkt function, say kkt(t), we want to find the value h such that: kkt(t+h) = 0.
By the above inequality, this would be roughly the same as solving....

kkt(t) + Dkkt(t)h = 0
-> Dkkt(t)h = -kkt(t)

This is a linear equation whose solution is:

h = -D^(-1)kkt(t) * kkt(t)

hence, h is an appropriate guess for the root of the kkt function. It won't be exactly correct after one iteration,
but after many, it will find the root. This is the essense of how this all works.

The way we are able to ensure dual feasiblity and inequalities hold is by starting with an initial dual feasible
and strictly feasible solution (f < 0, we get this initial solution by solving the "phase 1 problem") and add constraints in the line search function to make sure that the
next solution stays feasible.

'''


def kkt_t(x, lam, nu, t, grad_f0, f, Df, A, b):
    """
    KKT residual vector for the primal-dual interior point method.

    Args:
        x   : (n,)   primal variable
        lam : (m,)   dual variable for inequality constraints
        nu  : (p,)   dual variable for equality constraints
        t   : float  barrier parameter
        grad_f0 : callable x -> (n,)   gradient of objective
        f       : callable x -> (m,)   inequality constraint values
        Df      : callable x -> (m,n)  Jacobian of inequality constraints
        A       : (p,n) equality constraint matrix
        b       : (p,)  equality constraint RHS

    Returns:
        r : (n+m+p,) residual vector stacked as [dual, centrality, primal]
    """
    fx = f(x)                          # (m,)
    dual       = grad_f0(x) + Df(x).T @ lam + A.T @ nu   # (n,)
    centrality = -np.diag(lam) @ fx - (1.0 / t) * np.ones(len(lam))  # (m,)
    primal     = A @ x - b                                            # (p,)
    return np.concatenate([dual, centrality, primal])


def D_kkt_t(x, lam, hess_f0, Hf, f, Df, A):
    """
    Jacobian of kkt_t with respect to (x, lam, nu).

    Args:
        x       : (n,)   primal variable
        lam     : (m,)   dual variable for inequality constraints
        hess_f0 : callable x -> (n,n)   Hessian of objective
        Hf      : callable x -> (m,n,n) Hessians of each inequality constraint (Hf(x)[i] = ∇²fᵢ(x))
        f       : callable x -> (m,)    inequality constraint values
        Df      : callable x -> (m,n)   Jacobian of inequality constraints
        A       : (p,n)  equality constraint matrix

    Returns:
        J : (n+m+p, n+m+p) Jacobian matrix
    """
    m, p = len(lam), A.shape[0]
    fx  = f(x)    # (m,)
    Dfx = Df(x)   # (m,n)
    Hfx = Hf(x)   # (m,n,n)

    # Hessian of Lagrangian: ∇²f₀ + Σᵢ λᵢ ∇²fᵢ
    hess_lag = hess_f0(x) + np.tensordot(lam, Hfx, axes=([0], [0]))  # (n,n)

    # Build block matrix row by row
    row_dual       = np.block([hess_lag,            Dfx.T,          A.T                      ])  # (n, n+m+p)
    row_centrality = np.block([-np.diag(lam) @ Dfx, -np.diag(fx),   np.zeros((m, p))         ])  # (m, n+m+p)
    row_primal     = np.block([A,                    np.zeros((p, m)), np.zeros((p, p))        ])  # (p, n+m+p)

    return np.vstack([row_dual, row_centrality, row_primal])


def parse(f0_str, ineq_strs, eq_strs):
    """
    Parse CLI strings into numpy callables for use with kkt_t and D_kkt_t.

    Args:
        f0_str    : string  objective function, e.g. "x1**2 + x2**2"
        ineq_strs : list of strings  inequality constraints f_i(x) <= 0, e.g. ["x1 + 1", "x2 + 1"]
        eq_strs   : list of strings  equality constraints, e.g. ["x1 + x2 == 5"]

    Returns:
        grad_f0 : callable x -> (n,)
        hess_f0 : callable x -> (n,n)
        f       : callable x -> (m,)
        Df      : callable x -> (m,n)
        Hf      : callable x -> (m,n,n)
        A       : (p,n) numpy array
        b       : (p,)  numpy array
        syms    : list of sympy symbols in the order corresponding to x
    """
    f0_expr    = sp.sympify(f0_str)
    ineq_exprs = [sp.sympify(s) for s in ineq_strs]
    # split on "==" or "=" before sympifying so Python's == isn't evaluated
    def _parse_eq(s):
        if '==' in s:
            lhs, rhs = s.split('==', 1)
        elif '=' in s:
            lhs, rhs = s.split('=', 1)
        else:
            return sp.sympify(s)
        return sp.sympify(lhs.strip()) - sp.sympify(rhs.strip())
    eq_exprs = [_parse_eq(s) for s in eq_strs]

    # collect and sort all variables for a consistent ordering
    all_exprs = [f0_expr] + ineq_exprs + eq_exprs
    syms = sorted(
        set().union(*[expr.free_symbols for expr in all_exprs]),
        key=lambda s: s.name
    )

    # lambdify a single sympy scalar expr into a Python callable (x: array) -> float
    def _scalar_fn(expr):
        fn = sp.lambdify(syms, expr, 'numpy')
        return lambda x: float(fn(*x))

    # --- objective ---
    grad_fns = [_scalar_fn(sp.diff(f0_expr, s)) for s in syms]
    def grad_f0(x):
        return np.array([fn(x) for fn in grad_fns])

    hess_fns = [[_scalar_fn(sp.diff(f0_expr, si, sj)) for sj in syms] for si in syms]
    def hess_f0(x):
        return np.array([[fn(x) for fn in row] for row in hess_fns])

    # --- inequality constraints ---
    f_fns = [_scalar_fn(fi) for fi in ineq_exprs]
    def f(x):
        return np.array([fn(x) for fn in f_fns])

    Df_fns = [[_scalar_fn(sp.diff(fi, s)) for s in syms] for fi in ineq_exprs]
    def Df(x):
        return np.array([[fn(x) for fn in row] for row in Df_fns])

    Hf_fns = [[[_scalar_fn(sp.diff(fi, si, sj)) for sj in syms] for si in syms] for fi in ineq_exprs]
    def Hf(x):
        return np.array([[[fn(x) for fn in row] for row in mat] for mat in Hf_fns])

    # --- equality constraints -> A, b ---
    if eq_exprs:
        A_sym, b_sym = sp.linear_eq_to_matrix(eq_exprs, syms)
        A = np.array(A_sym.tolist(), dtype=float)
        b = np.array(b_sym.tolist(), dtype=float).flatten()
    else:
        n = len(syms)
        A = np.zeros((0, n))
        b = np.zeros(0)

    return grad_f0, hess_f0, f, Df, Hf, A, b, syms


def newton_step(x, lam, nu, t, grad_f0, hess_f0, f, Df, Hf, A, b):
    """
    Compute one Newton step by solving D_kkt_t @ h = -kkt_t.

    Returns:
        dx   : (n,) step for primal variable
        dlam : (m,) step for inequality duals
        dnu  : (p,) step for equality duals
    """
    r = kkt_t(x, lam, nu, t, grad_f0, f, Df, A, b)
    J = D_kkt_t(x, lam, hess_f0, Hf, f, Df, A)
    h = np.linalg.solve(J, -r)

    n, m = len(x), len(lam)
    return h[:n], h[n:n+m], h[n+m:]


def compute_t(x, lam, f, mu):
    """
    Compute the barrier parameter t for the next interior point iteration.

        t = mu * m / eta,   eta = -f(x).T @ lam  (surrogate duality gap)

    Args:
        x   : (n,) current primal variable
        lam : (m,) current inequality duals
        f   : callable x -> (m,)
        mu  : float > 1, controls how aggressively t increases

    Returns:
        t   : float
        eta : float  surrogate duality gap (useful for the stopping criterion)
    """
    eta = -f(x) @ lam
    m   = len(lam)
    return mu * m / eta, eta


def backtrack(x, lam, nu, dx, dlam, dnu, t, grad_f0, f, Df, A, b, alpha=0.01, beta=0.5):
    """
    Backtracking line search for the primal-dual interior point method.

    Guarantees:
        lam + s*dlam > 0                        (dual strict feasibility)
        f(x + s*dx) < 0                         (primal strict feasibility)
        ||r(y + s*h)|| <= (1-alpha*s)||r(y)||   (Armijo sufficient decrease)

    Args:
        alpha : Armijo parameter (0, 0.5)
        beta  : backtracking factor (0, 1)

    Returns:
        s : step size
    """
    # 1. Largest step keeping lam > 0
    neg = dlam < 0
    s_max = float(np.min(-lam[neg] / dlam[neg])) if np.any(neg) else 1.0
    s = min(1.0, 0.99 * s_max)

    # 2. Backtrack until primal strict feasibility holds
    while np.any(f(x + s * dx) >= 0):
        s *= beta

    # 3. Armijo: sufficient decrease in ||kkt_t||
    r0_norm = np.linalg.norm(kkt_t(x, lam, nu, t, grad_f0, f, Df, A, b))
    while np.linalg.norm(kkt_t(x + s*dx, lam + s*dlam, nu + s*dnu, t, grad_f0, f, Df, A, b)) \
            > (1 - alpha * s) * r0_norm:
        s *= beta

    return s


def solve(x0, mu, grad_f0, hess_f0, f, Df, Hf, A, b, eps=1e-6):
    """
    Primal-dual interior point solver.

    Args:
        x0  : (n,) initial strictly feasible point (f(x0) < 0)
        mu  : float > 1, surrogate duality gap scaling factor
        eps : stopping tolerance on ||kkt_t||

    Returns:
        x, lam, nu at convergence
    """
    x   = x0.copy()
    lam = np.ones(len(f(x0)))
    nu  = np.zeros(A.shape[0])

    while True:
        t, _          = compute_t(x, lam, f, mu)
        dx, dlam, dnu = newton_step(x, lam, nu, t, grad_f0, hess_f0, f, Df, Hf, A, b)
        s             = backtrack(x, lam, nu, dx, dlam, dnu, t, grad_f0, f, Df, A, b)

        x   = x   + s * dx
        lam = lam + s * dlam
        nu  = nu  + s * dnu

        kkt = kkt_t(x, lam, nu, t, grad_f0, f, Df, A, b)
        if np.linalg.norm(kkt) < eps:
            break

    return x, lam, nu


def phase1(f, Df, Hf, A, b, x_init, mu=10.0, eps_reg=1e-8):
    """
    Find a strictly feasible starting point via Phase I.

    Runs the interior point method on:
        min  s + (eps_reg/2)||x||^2
        s.t. f_i(x) - s <= 0   for all i
             Ax = b

    Terminates as soon as f(x) < 0 rather than solving to full convergence.
    The eps_reg term prevents a singular Newton system for linear constraints.

    Args:
        x_init : (n,) initial point satisfying Ax_init = b
        mu     : surrogate duality gap scaling
        eps_reg: regularization weight

    Returns:
        x_feas : (n,) strictly feasible point with f(x_feas) < 0

    Raises:
        ValueError if 1000 iterations pass without finding a feasible point
    """
    n = len(x_init)
    m = len(f(x_init))

    # augmented variable y = [x, s]

    def grad_f0_p1(y):
        g      = np.zeros(n + 1)
        g[:n]  = eps_reg * y[:n]
        g[-1]  = 1.0
        return g

    def hess_f0_p1(y):
        H         = np.zeros((n + 1, n + 1))
        H[:n, :n] = eps_reg * np.eye(n)
        return H

    def f_p1(y):
        return f(y[:n]) - y[-1]

    def Df_p1(y):
        return np.hstack([Df(y[:n]), -np.ones((m, 1))])

    def Hf_p1(y):
        return np.pad(Hf(y[:n]), ((0, 0), (0, 1), (0, 1)))

    A_p1 = np.hstack([A, np.zeros((A.shape[0], 1))])
    s0   = float(np.max(f(x_init))) + 1.0
    y    = np.append(x_init, s0)
    lam  = np.ones(m)
    nu   = np.zeros(A.shape[0])

    for _ in range(1000):
        if np.all(f(y[:n]) < 0):
            return y[:n]

        t, _          = compute_t(y, lam, f_p1, mu)
        dy, dlam, dnu = newton_step(y, lam, nu, t, grad_f0_p1, hess_f0_p1, f_p1, Df_p1, Hf_p1, A_p1, b)
        s             = backtrack(y, lam, nu, dy, dlam, dnu, t, grad_f0_p1, f_p1, Df_p1, A_p1, b)

        y   = y   + s * dy
        lam = lam + s * dlam
        nu  = nu  + s * dnu

    raise ValueError("Phase I failed: could not find a feasible point after 1000 iterations")
