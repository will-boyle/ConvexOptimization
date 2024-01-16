# algorithms included here:
# 1) path following algorithm
# read about the algorithms at the bottom of the file

# overview of the path following algorithm 

# Starts with an arbitrary point solution e = (x,w,y,z) > 0 
# step 0: perform test for optimality 
# step 1: if not optimal compute new value of u
# step 2: determine step direction t using Newton's method
# step 3: calculate step size theta
# step 4: reset value of e to be e + theta * t
import numpy as np

def main(A, x, w, b, y, z, c):
    iteration = 0
    status = 0
    
    while status**2 != 1:
        iteration +=1 
        print("............................................iteration...", iteration)
        print("current value of by", b@y)
        print("current value of cx...", c @ x)
        print("duality gap...", c@x - b@y)
        print("current value of Ax + w", A @ x + w)
        print("curent value of A.Ty - z", A.T @ y - z)
        
        u = compute_value_of_barrier_parameter(x,w,y,z)
        delta_x, delta_w, delta_y, delta_z = compute_step_direction(A,x,w,y,z,b,c,u)
        step_size = compute_step_size(x,w,y,z,delta_x, delta_w, delta_y, delta_z)
        print(" current value of x is...", x)
        print("the x step direction is...", delta_x)
        print("the current value of w is...", w)
        print("the w step direction is...", delta_w)
        print("the y step directoin is...", delta_y)
        print("the z step directin is...", delta_z)
        print("the step size is...", step_size)

        # I don't know why but I think you have to do this...
        x_new = x + step_size * delta_x
        w_new = w + step_size * delta_w
        y_new = y + step_size * delta_y
        z_new = z + step_size * delta_z

        x = x_new
        w = w_new
        y = y_new
        z = z_new

        status = test_for_optimality(A,x,w,b,y,z,c)

def test_for_optimality(A, x, w, b, y, z, c):
    epsilon = .0001
    M = 1000000
    
    infinity_norm_of_x = np.max(x)
    infinity_norm_of_y = np.max(y)

    if ( infinity_norm_of_x > M ) or ( infinity_norm_of_y > M):
        status = -1 
        return status

    else:

        # need identity matrix for the primal slack vector w, which will be an m vector
        I_m = np.identity(len(b))
        p = b - A @ x - I_m @ w
        one_norm_of_p = np.sum(np.abs(p))

        # need identity matrix for the dual slack vector z, which will be an n vector
        I_n = np.identity(len(c))
        d = c - A.T @ y + I_n @ z
        one_norm_of_d = np.sum(np.abs(d))
    
        complementarity_test = x.T @ z + y.T @ w
        
        # if primal feasible and dual feasible and u-complemetarity holds then solution is optimal
        if ( one_norm_of_p < epsilon ) and ( one_norm_of_d < epsilon ) and ( complementarity_test < epsilon ):
            status = 1
        else:
            status = 0


        return status

def compute_value_of_barrier_parameter(x, w, y, z):
    delta = .1
    n_plus_m = len(x) + len(y)
    scalar = delta / n_plus_m

    u = x.T @ z + y.T @ w
    barrier_parameter_estimate = scalar * u
    return barrier_parameter_estimate

def compute_step_direction(A, x, w, y, z, b, c, u):
    '''creates the matrix associated with the central path, then computes the inverse
    of it to solve the system of equations.'''
    I_m = np.identity(len(b))
    zero_m_by_m = np.zeros([len(b), len(b)])
    zero_m_by_n = np.zeros([len(b), len(c)])
    
    # first row: A, I, 0, 0
    row_one = np.concatenate((A, I_m, zero_m_by_m, zero_m_by_n), axis = 1)
    
    zero_n_by_n = np.zeros([len(c), len(c)])
    zero_n_by_m = np.zeros([len(c), len(b)])
    I_n = np.identity(len(c))

    # second row: 0,0,A.T, -I
    row_two = np.concatenate((zero_n_by_n, zero_n_by_m, A.T, -1 * I_n), axis = 1)
    
    # I_n * z is a n by n times n by one matrix, returns n by n matrix because it's special
    # multiplication, I do this to create Z matrix from z vector
    # row three: Z,0,0,X
    row_three = np.concatenate((I_n * z, zero_n_by_m, zero_n_by_m, I_n * x), axis = 1)

    # row four: 0,Y,W,0
    row_four = np.concatenate((zero_m_by_n, I_m * y, I_m * w, zero_m_by_n), axis = 1)

    matrix = np.concatenate((row_one, row_two, row_three, row_four), axis = 0)
    
    ones_m = np.ones(len(b))
    ones_n = np.ones(len(c))

    p = b - A @ x - w
    d = c - A.T @ y + z
    primal_complementarity_condition = u * ones_n - (I_n * x) @ (I_n * z) @ ones_n
    dual_complementarity_condition = u * ones_m - (I_m * y) @ (I_m * w) @ ones_m
    
    right_hand_side = np.concatenate((p,d,primal_complementarity_condition, dual_complementarity_condition), axis = 0)

    solution = np.linalg.inv(matrix) @ right_hand_side
    
    m = len(b) 
    n = len(c)
 
    # have to break up the solution vector into its constituents
    # first m indices correspond to delta x etc. See page 312 of LP - vanderbei 
    delta_x = solution[0: n]
    delta_w = solution[n: n + m]
    delta_y = solution[n + m: n + 2*m]
    delta_z = solution[n + 2*m:]

    return delta_x, delta_w, delta_y, delta_z

# the actual computation should be such that the new solution is strictly positive
# however with the methodology below we could run into a divide by zero error..
# trying to work around this reality for now
def compute_step_size(x,w,y,z,delta_x, delta_w, delta_y, delta_z):
    scalar = .9

    candidates = [1]
    if len(delta_x[np.where(delta_x < 0)]) > 0:
        theta_x = min(-1*x[np.where(delta_x < 0)] / delta_x[np.where(delta_x < 0)])
        candidates.append(theta_x)
    if len(delta_w[np.where(delta_w < 0)]) > 0:
        theta_w = min(-1*w[np.where(delta_w < 0)] / delta_w[np.where(delta_w < 0)])
        candidates.append(theta_w)
    if len(delta_y[np.where(delta_y < 0)]) > 0:
        theta_y = min(-1*y[np.where(delta_y < 0)] / delta_y[np.where(delta_y < 0)])
        candidates.append(theta_y)
    if len(delta_z[np.where(delta_z < 0)]) > 0:
        theta_z = min(-1*z[np.where(delta_z < 0)] / delta_z[np.where(delta_z < 0)])
        candidates.append(theta_z)
    
    theta = min(candidates)
    step_size = scalar * theta

    return step_size

'''
                                        Step 0: the test for optimality

choose a small value for epsilon and a large value for M.
the optimal solution must satifsy the following: 
    1) p := b - ax - w ~ 0 (primal feasibility test)
    2) d := c - aTy + z ~ 0 (dual feasibility test)
    3) u := xTz + yTw ~ 0 (u complementarity test)

the way we effectuate this test is by choosing a small number epsilon and requiring:
    1) 1-norm(p) < epsilon
    2) 1-norm(d) < epsilon
    3) u < epsilon

We also must have a test for unboundedness, the test is just the following using a very large number for M
say 1000000:
    1) infinity_norm(x) > M
    2) infinity_norm(y) > M

If any of the above are two then we have that the problem is not subject to optimization and we terminate the 
algorithm.


                                        Step 1: compute value of u
let delta be an element of (0,1). I will use the value of .1

u = delta * ( xTz + yTw / n+m )

observe that if e is on the central path, the numerator in parenthesis would equal (m+n)*u. This is where the equation comes from.



                                        step 2: determine step direction using Newton's method.

recall that f(x+t) ~ f(x) + f'(x) * t
we want the LHS to be getting closer to zero so we want to find t where we know f, f':
    f'(x) * t = -f(x)

This is the above linear system that we solve iteratively, so after this step t, will be known and you try to find x+t+t2 or however you want to think about that until you have arrived at a root of the function. 

Now onto the function and its derivative. 

let e = (x, w, y , z) 

F(e) = (ax + w - b, aTy - z - c, XZe  - ue, YWe - ue)

F'(e) = [df1/dx, df1/dw..]
        [df2/dx, ...     ]

      = [A, I, 0, 0]
        [0, 0, AT, -I]
        [Z, 0, 0, X]
        [0, Y, W, 0]

so as the algorithm develops x,w,y,z are changing in the e and therefore f(e) and f'(e)

                                       step 3: determine the step size


recall that we require e > 0. It may happen that the solution given by newton's method above yields an answer whichis such that e + d <= 0. Notice the lax inequality. Hence, there is requirement to scale t so that this cannot happen. That is the purpose of this step. Observer that if d >= 0, then there is no need to scale it. Hence, we can simply assign 1 as the scale to it and that will be the step size. On the other hand, if entries are negative which they are likely to be, then e + d is endanger of not staying positive. 

Note: I use the theta and t interchangeably. I'm referring to step size each time. 

Let t_x = min { -xi/ d_xi, where d_xi is negative }
similarly define t_w, t_y, t_z.

let t' = min { t_x, t_w, t_y, t_z }

if we let this be our scaled value, then at least on of the values of x + td will be zero. Hence, we need to scale this numbereven further. Let r be an element of (0,1). Then t = t' * r will satisfy our requirements. r = .9 in my case. 

'''

