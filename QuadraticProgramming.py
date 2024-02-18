# Library to solve quadratic programming problems
# Algorithms contained
#     1) QP2 - solver for quadratic program with linear equality constraints, where rows of constraint
#       matrix are linearly independent. Algorithm requires an initial
#       solution. This algorithm referred to as QP2, since 
#       it is the second algorithm described in Quad. Prog. w Computer Programs by Michael Best. 

# NOTE in QP -1 corresponds to -2 here and 0 to -1. In regards to their meaning for J 
import numpy as np



def QP2(c, C, x, D, J):
    i = 0
    grad_i = c + C @ x
    test_for_optimality = 0

    while test_for_optimality == 0:
        i += 1
        Dinv = np.linalg.inv(D)
        print("D inv is", Dinv)
        test_for_optimality = 0

        print("")
        print("----------------")
        print("")
        print("iteration...", i)
        print("the current value of f is...", c @ x + .5 * x @ C @ x)

        k = QP2_step_one(grad_i, J, Dinv)
        if k == -1:
            print("algorithm has successfully terminated, solution is..", x)
            test_for_optimality = 1
            return c @ x + .5 * x @ C @ x

        if grad_i @ Dinv[:, k] == 0.0:
            print("algorithm has successfully terminated, solution is..", x)
            test_for_optimality = 1
            return c @ x + .5 * x @ C @ x

        # create search direction 
        if grad_i @ Dinv[:, k] > 0.0:
            s_i = Dinv[:, k] 
        else:
            s_i = -1 * Dinv[:, k]

        t = QP2_step_two(grad_i, s_i, C)

        print("search direction is...", s_i)
        print("t is...", t)
        
        # J, D are implicitly updated once the QP2_step_three function is called
        updates = QP2_step_three(x, t, s_i, c, C, J, k, D)
        
        x = updates[0]
        grad_i = updates[1]
        print("value of x is...", x)
        print("value of grad at x is...", grad_i)
    





# point of this function is to return k
# if J has no 0s, then this function will break. So need to deal with that test case in solver. 
def QP2_step_one(grad_i, J, Dinv):
    # determine index which will correspond to the new search direction. 
    # compute max { |g_i * Dinv_k|, where J_k = 0 }
    indices = np.where(J == -1.0)
    #print("indices",indices)

    products = []
    for index in indices[0]:
        #print("index")
        #print(index)
        # multiply gradient and Dinv[:, index] and take its absolute value
        # g_i * c_k, c_k represents col. k of Dinv.
        gick = float(np.abs(grad_i @ Dinv[:, index]))
        # add the above to the products list 
        products.append(gick)
    
    if len(products) == 0:
        return -1
    
    max_product = max(products)
    # I beleive .index returns the FIRST index which satisfies.. hence this quietly selects according
    # to bland's rule
    max_index = products.index(max_product)
    #print("max index is...", max_index)
    k = indices[0][max_index]

    return k


# purpose is to compute step size
def QP2_step_two(grad_i, s_i, C):
    gisi = grad_i @ s_i
    sCs = s_i @ C @ s_i
    t = gisi/sCs
    return t

def QP2_step_three(x_i, t, s_i, c, C, J, k, D):
    # update value of x
    x_i = x_i - t*s_i

    # update value of grad_i
    g_i = c + C @ x_i

    # update J
    J[k] = -2.0

    # create d, the vector which will replace row k of D
    d = 1/( (s_i @ C @ s_i)**(.5) ) * C @ s_i
    
    # update D
    D[k] = d


    return [x_i, g_i]
 
# QP3 contains all of the steps of QP2 but has other
# Assumptions are that you have a feasible solution already. 
# The matrix D is the matrix representing the active constraints on the 
# feasiable solution. J seems to be the indices of the active cosntraints
def QP3(A, x, b, c, C, D, J):

    i = 0
    test_for_optimality = 0
    while test_for_optimality != 1:
        i += 1
        print("------------------iteration {}-------------------------".format(i))
        print("value of f is...",  c @ x + .5 * x @ C @ x)

        gi = c + C @ x
        Dinv = np.linalg.inv(D)
        # step 1. If there is at least one index in J equal to 0, go to 1.1 else 1.2
     
        # see where indices where J = 0.0
        indices = np.where(J == -1.0)
    
        # see if there is an index where J = 0.0
        if len(indices[0]) != 0:
            # by the above such an index exists and you will go find k as in QP2 step 1
            # note in QP2 it was possible that k could be -1, this is not possible 
            # because of the above conditional test. 
            #  this is step 1.1 in QP3 
            k = QP2_step_one(gi, J, Dinv)
            if gi @ Dinv[:, k] == 0.0:
                k = QP3_step_1p2(J, gi, Dinv)
                if k == -1:
                    test_for_optimality = 1
                else:
                    s_i = Dinv[:, k]


            # create search direction 
            elif gi @ Dinv[:, k] > 0.0:
                s_i = Dinv[:, k]

            else:
                s_i = -1 * Dinv[:, k]
        else: 
            k = QP3_step_1p2(J, gi, Dinv)
            if k == -1:

                test_for_optimality = 1
            else:
                s_i = Dinv[:, k]

        step_2_results = QP3_step_2(gi, s_i, C, A, x, b)

        t = step_2_results[0]
        if t == -1:
            return "the problem is unbounded"
        else:
            l = step_2_results[1]
            if step_2_results[2] == 't1':
                updates = QP2_step_three(x, t, s_i, c, C, J, k, D)
                x = updates[0]
                gi = updates[1]
            else:
                updates = QP3_step_3p2(A, x, t, s_i, c, C, J, k, l, D)
                x = updates[0]
                gi = updates[1]

    print("algorithm has terminated f min is...",c @ x + .5 * x @ C @ x)
    print("argmin of f is...", x)

# gi is gradient of f at x at iteration i
def QP3_step_1p2(J, gi, Dinv):
    # indices where 1,..,m are in J
    indices = np.where(J >= 0.0)

    if len(indices[0]) != 0:
        products = []
    
        for index in indices[0]:
            gick = float(gi @ Dinv[:, index])
            # add the above to the products list 
            products.append(gick)
    
    
        max_product = max(products)
        if max_product <= 0.0:
            # this will signify that the solution is optimal
            return -1

        max_index = products.index(max_product)
        k = indices[0][max_index]
        return k

    else:
        # the solition is optimal
        return -1

# calculates maximum step size. returns >= 0 unless problem is unbounded. Also determines new active constraint ind.
def QP3_step_2(g_i, s_i, C, A, x, b):
    gisi = g_i @ s_i
    sCs = s_i @ C @ s_i
    if sCs == 0.0:
        t1  = 'inf'
    else:
        t1 = gisi/sCs

    Ax_minus_b = A @ x - b
    inactive_constraints = np.where(Ax_minus_b < 0.0)

    As_i = A @ s_i 
    limiting_constraints = np.where(As_i < 0.0)

    indices = np.intersect1d(inactive_constraints, limiting_constraints)

    if len(indices) == 0:
        t2 = 'inf'
        # there is no index which becomes active, so l = -1 to indicate this. 
        l = -1
    else:
        scalars = []
        for i in indices:
            scalars.append( (A[i,:] @ x - b[i]) / (A[i,:] @ s_i) )
        t2 = min(scalars)
        min_index = scalars.index(t2)
        l = indices[min_index]
    
    if (t1 == 'inf') and (t2 == 'inf'):
        # means the problem is unbounded. Since when it isn't t >= 0
        return -1, l
    elif t1 == 'inf':
        return t2, l, 't2'
    elif t2 == 'inf':
        return t1, l, 't1'
    else:
        if min(t1, t2) == t1:
            return t1, l, 't1'
        else:
            return t2, l, 't2'


def QP3_step_3p2(A, x_i, t, s_i, c, C, J, k, l, D):
    # update value of x
    x_i = x_i - t*s_i

    # update value of grad_i
    g_i = c + C @ x_i

    # update J
    J[k] = l
    indices = np.where(J == -2.0)
    J[indices] = -1.0

    
    # update D
    D[k] = A[l,:]


    return [x_i, g_i]

