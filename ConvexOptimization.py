import numpy as np
np.set_printoptions(precision=2, suppress=True)
import autograd.numpy as ag
from autograd import grad, jacobian

'''
Algorithms contained: 
    1) simplex method (two phase method)
    2) branch and bound (used to solve integer lps)
    3) primal dual method (for more general Convex problems .. assumes standard form .. from Boyd and Vandenberghe)


'''

def prompt_user_for_A_matrix():
    A = input("Enter A matrix. Values in rows separated by comma, begin next row with semi-colon ")
    rows = A.strip().split(";")
    A = [list(map(int, row.split(","))) for row in rows]
    A = np.array(A)
    return A

def prompt_user_for_b_vector():
    user_input = input("Enter the b vector (ex: '1,1'): ")
    ls = [float(x) for x in user_input.split(",")]
    b = np.array(ls)
    return b

def simplex_algorithm(c, A, x, b, b_vars, n_vars):
    B_cols = A[:, b_vars]
    B = np.concatenate([B_cols], axis = 1)

    N_cols = A[:, n_vars]
    N = np.concatenate([N_cols], axis = 1)

    flag = 0
    counter = 0
    
    while flag == 0:
        counter += 1

        test = test_for_optimality(B, N, c, x, b_vars, n_vars)
        if np.all(test) == True:
            flag = 1
            #print("basic vars", b_vars)
            print("OPTIMAL SOLUTION...{}".format(c @ x))
            print("ARGMIN...", x)
            print(f"the problem was solved in {counter} interations")
            #print("optimal variables", b_vars)
            #print("optimal basis", B)
            return [x, c @ x, b_vars, n_vars] 
        else: 

            # chosen according to Bland's rule.
            indices_where_test_fails = np.where(test == False)
            nonbasic_index_to_enter = n_vars[min(indices_where_test_fails)[0]]
   

            db = compute_direction_of_increase(B, N, nonbasic_index_to_enter, n_vars)[0] 
            dn = compute_direction_of_increase(B, N, nonbasic_index_to_enter, n_vars)[1]
            

            # gets the value of t for x + td, also gets index which is leaving the basis.
            compute_amount_of_increase_info = compute_amount_of_increase(x, b_vars, n_vars, db, dn)
            t = compute_amount_of_increase_info[0]
            #print("t", t)
            basic_index_to_leave = compute_amount_of_increase_info[1]
            
            tdb = t * db
            tdn = t * dn

            x[b_vars] += tdb
            x[n_vars] += tdn

            # update the basic variables
            new_bvars = []
            for index in b_vars:
                if index != basic_index_to_leave:
                    new_bvars.append(index)
            new_bvars.append(nonbasic_index_to_enter)
            b_vars = new_bvars

	
	    # update the nonbasic variables 
            new_nvars = []
            for index in n_vars:
                 if index != nonbasic_index_to_enter:
                     new_nvars.append(index)
            new_nvars.append(basic_index_to_leave)
            n_vars = new_nvars

            B_cols = A[:, b_vars]
            B = np.concatenate([B_cols], axis = 1)

            N_cols = A[:, n_vars]
            N = np.concatenate([N_cols], axis = 1)
 


# c,x are numpy array, b/n vars are integers in lists.
# in computing CbB^-1N, it is best to compute it as (CbB^-1) @ N,
def test_for_optimality(B, N, c, x, basic_vars, nonbasic_vars):
    # initialize an empty array to append values to
    Cn = np.array([])
    
    # need to select all the components of c which correspond to nonbasic vars
    # ie need to create Cn from c. 
    for index in nonbasic_vars:
        Cn = np.append(Cn, c[index])
    
    Cb = np.array([])

    # similarly, need to create Cb from c.
    for index in basic_vars:
        Cb = np.append(Cb, c[index])
   
    CbBinv = Cb @ np.linalg.inv(B)
    
    reduced_costs = Cn - CbBinv @ N

    #create zero vector for comparison
    zero_vector = np.zeros(len(Cn))

    test = reduced_costs >= zero_vector

    return test 




# j is the nonbasic index whose corresponding column is chosen to enter the basis
# db = -Binv @ N_j
# dn = I_j
def compute_direction_of_increase(B, N, j, nonbasic_vars):
    
    # find the index of nonbasic_vars where the value is equal to j. 
    # this would correspond to j* in the Dan Solow book. 
    j_star = nonbasic_vars.index(j)

    # now will compute db
    db = -1 * np.linalg.inv(B) @ N[:, j_star]

    # create an identity matrix of size len(nonbasic_vars)
    identity_matrix = np.identity(len(nonbasic_vars))
    
    # create dn
    dn = identity_matrix[:, j_star]
    
    return [db,dn]    




# something about this function seems to be wrong, it seems like when t = 0, sometimes it puts in a singular matrix as the basis. I don't think this is supposed to happen. 
# computes min { -x/d_i, where d_i is negative }
# determines which basis variable becomes nonbasis, determines k* in Dan Solow 
def compute_amount_of_increase(x, basic_vars, nonbasic_vars, db, dn):
    # need to determine the indices whre db is negative. 
    zero_vector = np.zeros(len(basic_vars))
    unbounded_test = db >= zero_vector

    if np.all(unbounded_test) == True: 
        print("the problem is unbounded, the direction of unboundedness is db: {0}, db = {1})".format(db, dn))     
        
    else: 
        indices_where_db_is_negative = np.where(unbounded_test == False)
       

        #algorithm assumes x >= 0, hence t >= 0 
        t = np.min(-1 *
np.divide(x[basic_vars][indices_where_db_is_negative],db[indices_where_db_is_negative]))
        xb_star = x[basic_vars] + t * db
        # the basic index of the leaving variable, denoted k*, chosen by
        # Bland's rule
        basic_vars_array = np.array(basic_vars)
        #if t != 0:
        k_star = int(np.min(basic_vars_array[indices_where_db_is_negative][np.where(xb_star[indices_where_db_is_negative] == 0)]))

        
        return [t, k_star]

def get_phase_one_matrix(A):
    # for an m X n matrix, we need m basis vectors
    I = np.identity(A.shape[0])

    # adds the m x m identity matrix at the end of the original matrix. 
    LP1_matrix = np.concatenate((A,I), axis=1)

    return LP1_matrix

def get_phase_one_objective_fn(A):
    # each column of original problem will have 0 associated
    zeros = np.zeros(A.shape[1])
    
    # each dummy variable has 1 associated, there are m 
    ones = np.ones(A.shape[0])
    
    # combine the zeros and ones into one vector
    phase_one_objective = np.concatenate((zeros,ones), axis = 0)

    return phase_one_objective

def get_phase_one_init_solution(A,b):
    # the first n values will be 0s
    zeros = np.zeros(A.shape[1])
    
    # just append b and this is the initial sol. 
    init_value = np.concatenate((zeros, b), axis = 0)

    return init_value

def get_phase_one_basic_variables(A):
    # just selecting last m indices starting from index n + 1
    basic_variables = list(range(A.shape[1], A.shape[1] + A.shape[0]))
    
    return basic_variables 

def get_phase_one_nonbasic_variables(A):
    # first n cols
    nonbasic_variables = list(range(0, A.shape[1]))
    
    return nonbasic_variables 

# input matrix is the phase one matrix, NOT the original matrix
def get_phase_two_matrix(A):
    # phase 2 matrix has one extra row and one extra column than p1 matrix
    
    # last constraint is (0,0...,1,1...,1) n zeros, m ones, 1 one for x,y,z 
    zeros = np.zeros(A.shape[1] - A.shape[0])
    ones = np.ones(A.shape[0])

    # concatenates the zeros and ones together to form one row vector 
    last_constraint_partial = np.concatenate((zeros, ones), axis=0)
    
    # will concatenate the rows of A with last_constraint_partial to get A
    # sans last column
    rows_of_A = A[ [ i for i in range(0, A.shape[0]) ] , : ]

    phase_two_matrix_partial = np.concatenate((rows_of_A, [last_constraint_partial]),axis = 0)
   
    # last column will be m zeros and one 1 
    zeros = np.zeros(A.shape[0])
    ones  = np.ones(1)

    last_column = np.concatenate((zeros, ones), axis = 0).reshape(-1,1)
    
    # concatenate the partial matrix with the last column to get the total matrix
    phase_two_matrix = np.concatenate((phase_two_matrix_partial,last_column),axis = 1)
    
    return phase_two_matrix

# input vector is the original A matrix and c vector, NOT the phase one A or c vector
def get_phase_two_objective_fn(A,c):
    # new c is original + m 0s for Y + one 0 for Z    
    # equivalently, c + m+1 0s
    zeros = np.zeros(A.shape[0]+1)
 
    phase_two_objective = np.concatenate((c, zeros), axis = 0)   
 
    return phase_two_objective

# input vector is the final solution to the phase 1 lp
def get_phase_two_init_solution(x):
    # same as final solution to phase 1 problem + 1 zero
    zeros = np.zeros(1)

    initial_solution = np.concatenate((x, zeros), axis = 0)


    return initial_solution

# input vector is original b, there is no phase one b that is different. 
def get_phase_two_b_vector(b):
    # you just add a zero to the original b vector, same logic as for p2 c vector
    zeros = np.zeros(1)

    phase_two_b = np.concatenate((b, zeros), axis = 0)

    return phase_two_b

# input list is going to be the output basic variables of solution to p1 prob.
# and phase 2 matrix. Nonbasic variables do not change.
def get_phase_two_basic_variables(A, basic_vars):
    basic_variables = basic_vars

    # this would be the last index of the phase two matrix
    basic_variables.append(A.shape[1] - 1)
    
    return basic_variables

def get_standard_form(c, A, inequality_types):
    '''need to be able to take a system that may be in a general form
    and turn the system into equality form using slack variables. Must also
    have the right hand side be >= 0'''
    I_m = np.identity(A.shape[0])
    x = []
    for inequality in inequality_types:
        if inequality == '<=':
            x.append(1)
        else:
            x.append(-1)
    x = np.array(x)
    # observe this is not mat mult, I am scaling each column. 
    I_x = x * I_m
    
    standard_form_matrix = np.concatenate((A, I_x), axis = 1)
    zero_m = np.zeros(A.shape[0])
    new_c = np.concatenate((c, zero_m), axis = 0)
    
    return [new_c, standard_form_matrix]

def solve_phase_one_problem(A, b):

    #phase = 'one'

    c = get_phase_one_objective_fn(A)

    phase_one_matrix = get_phase_one_matrix(A)

    x = get_phase_one_init_solution(A, b)
    
    b_vars = get_phase_one_basic_variables(A)
    
    n_vars = get_phase_one_nonbasic_variables(A)

    solution_info = simplex_algorithm(c, phase_one_matrix, x, b, b_vars, n_vars)
    x = solution_info[0]
    cx = solution_info[1]
    b_vars = solution_info[2]
    n_vars = solution_info[3]

    if cx != 0:
        print("problem is infeasible, phase one solution is:", cx)
        return [cx]
    else:
        return [cx, phase_one_matrix, x, b_vars, n_vars]



def solve_phase_two_problem(c, A, x, b, b_vars, n_vars):

    #phase = 'two'

    phase_two_objective_fn = get_phase_two_objective_fn(A,c)

    phase_two_matrix = get_phase_two_matrix(A)


    initial_solution = get_phase_two_init_solution(x)

    b = get_phase_two_b_vector(b)

    b_vars = get_phase_two_basic_variables(phase_two_matrix, b_vars)


    #print("............................... solving phase two problem...................................................................")  
    #print("")
    #print("")
    #print("")
    solution_info = simplex_algorithm(phase_two_objective_fn, phase_two_matrix, initial_solution, b, b_vars, n_vars)
    x = solution_info[0]
    cx = solution_info[1]
    return [x, cx]

def solve_lp(c, A, inequality_types, b):
    '''The first return value signals whether or not the problem was feasible 0 means no.'''
    
    c = get_standard_form(c, A, inequality_types)[0]
    A = get_standard_form(c, A, inequality_types)[1]

    phase_one_solution_info = solve_phase_one_problem(A, b)
    if phase_one_solution_info[0] != 0:
        return [0]

    phase_two_solution_info = solve_phase_two_problem(c, phase_one_solution_info[1], phase_one_solution_info[2], b, phase_one_solution_info[3], phase_one_solution_info[4])

    x = phase_two_solution_info[0]
    cx = phase_two_solution_info[1]
    return [1, cx, x]


def find_nonintegral_index(x):
    int_x = [ int(number) for number in x ]
    int_x = np.array(int_x)
    
    nonintegral_indices = np.where(int_x != x)
    if len(nonintegral_indices[0]) == 0:
        return []
    else: 
        return [np.min(nonintegral_indices)]

def get_left_branch_problem(A, x, inequality_types, b):
    I_n = np.identity(len(x[:A.shape[1]]))
    # because this function is called, we know there is a nonintegral value
    nonintegral_index = find_nonintegral_index(x[:A.shape[1]])[0]
    # the above index corresponds to the index of the identity matrix we will grab
    # it represents the left side of the constraint we must add.
    constraint_row = np.expand_dims(I_n[nonintegral_index], axis = 0)
    new_constraint_matrix = np.concatenate((A, constraint_row), axis = 0)

    # add <= to inequalities
    copy_of_inequalities = inequality_types.copy()
    copy_of_inequalities.append('<=')

    new_b = b[:]
    new_b = np.append(new_b, np.floor(x[nonintegral_index]))
    return new_constraint_matrix, copy_of_inequalities, new_b

def get_right_branch_problem(A, x, inequality_types, b):
    I_n = np.identity(len(x[:A.shape[1]]))
    # because this function is called, we know there is a nonintegral value
    nonintegral_index = find_nonintegral_index(x[:A.shape[1]])[0]
    # the above index corresponds to the index of the identity matrix we will grab
    # it represents the left side of the constraint we must add.
    constraint_row = np.expand_dims(I_n[nonintegral_index], axis = 0)

    new_constraint_matrix = np.concatenate((A, constraint_row), axis = 0)

    # add <= to inequalities
    copy_of_inequalities = inequality_types.copy()
    copy_of_inequalities.append('>=')

    new_b = b[:]
    new_b = np.append(new_b, np.ceil(x[nonintegral_index]))
    return new_constraint_matrix, copy_of_inequalities, new_b

# the creation of new c value when slacks are added needs to be made via a function
# missing pruning by bound condition
def branch_and_bound_algorithm(c, A, inequality_types, b):
    # of the problems whose feasible space is a partition of the original feasible space
    # if any of those problems have integer solution, they go here
    # the max of these is then the current best solution
    # contains both the value of cx and x. cx on odd indices.
    candidate_solution = [] 
    problems_to_solve = [[c, A, inequality_types, b]]
    # solve relaxed problem
    while len(problems_to_solve) != 0:
        #print("candidate solutions...", candidate_solution)
        # implements LIFO structure because of depth first preference
        # do I want to be pulling out the last problem in the list? 
        problem = problems_to_solve.pop(-1)
        #print("SOLVING PROBLEM.....")
        #print(problem)
        relaxed_solution = solve_lp(problem[0], problem[1], problem[2], problem[3])
        status = relaxed_solution[0]
        if status == 0:
            continue
        else:
            cx, x = relaxed_solution[1], relaxed_solution[2]

        # first make sure the problem is feasible, if not there is no integer solution because there is no solution. 
        if status == 1:
            # check whether or not the solution is an integer solution
            # if x is an integer solution, note I don't think it matters if slack/dummy variables take on fractional values. 
            if len(find_nonintegral_index(x[:A.shape[1]])) == 0:
                # if no candidate solution exists, add the integer solution
                if len(candidate_solution) == 0: 
                    candidate_solution.append(cx)
                    candidate_solution.append(x)
                # if an integer solution exists already, replace it with the new one only if new one is better.
                else: 
                    if cx < candidate_solution[0]:
                        del candidate_solution[0]
                        candidate_solution.append(cx)
                        candidate_solution.append(x)
            # if not an integer solution
            else:
                # add right branch problem (RBP) to be solved.
                RBP = get_right_branch_problem(problem[1], x, problem[2], problem[3])
                problems_to_solve.append([c, RBP[0], RBP[1], RBP[2]])
                # add left branch problem (LBP) to be solved.
                LBP = get_left_branch_problem(problem[1], x, problem[2], problem[3])
                problems_to_solve.append([ c, LBP[0], LBP[1], LBP[2]])
    cx = candidate_solution[0]
    x = candidate_solution[1]

    print("OPTIMAL SOLUTION: ", cx)
    print("ARGMIN: ", x)
    #print("the branch and bound algorithm has successfull terminated the optimal solution is....")
    return cx, x
def compute_function_value(function_str, x):
    """
    Compute the value of the function given as a string at a specific point.
    
    Args:
    - function_str: The function as a string (e.g., "x[0]**2 + x[1]**2")
    - x: The value of the vector at which to compute the function value (e.g., [2.0, 3.0])
    
    Returns:
    - The function value at the point x.
    """
    # Define the function dynamically using eval
    def func(x):
        return eval(function_str)

    # Ensure x is a NumPy array (Autograd works with NumPy arrays)
    x = ag.array(x)
    
    # Compute the function value at the given point
    function_value = func(x)
    
    return function_value

def compute_gradient(function_str, x):
    """
    Compute the gradient of a function given as a string with respect to a vector using autograd.
    
    Args:
    - function_str: The function as a string (e.g., "x[0]**2 + x[1]**2")
    - x: The value of the vector at which to compute the gradient (e.g., [2.0, 3.0])
    
    Returns:
    - The gradient as a numeric vector (list of scalar values)
    """
    # Define the function dynamically using eval
    def func(x):
        return eval(function_str)

    # Ensure x is a NumPy array (Autograd works with NumPy arrays)
    x = ag.array(x)
    
    # Use autograd's grad function to compute the gradient of the function
    gradient_fn = grad(func)
    
    # Compute the gradient at the given vector value
    gradient_at_value = gradient_fn(x)
    
    # Convert the gradient to a regular Python list (which contains scalar values)
    gradient_list = gradient_at_value.tolist()  # This works because gradient_at_value is a NumPy array
    
    return gradient_list

def compute_hessian(function_str, x):
    """
    Compute the Hessian (second-order derivative) of a function given as a string at a point using autograd.
    
    Args:
    - function_str: The function as a string (e.g., "x[0]**2 + x[1]**2")
    - x: The point at which to compute the Hessian (e.g., [2.0, 3.0])
    
    Returns:
    - The Hessian matrix as a NumPy array.
    """
    # Define the function dynamically using eval
    def func(x):
        return eval(function_str)

    # Ensure x is a NumPy array (Autograd works with NumPy arrays)
    x = ag.array(x)
    
    # First compute the gradient of the function
    gradient_fn = grad(func)
    
    # Now compute the Hessian by applying the jacobian of the gradient
    hessian_fn = jacobian(gradient_fn)
    
    # Compute the Hessian at the given vector value
    hessian_at_value = hessian_fn(x)
    
    return hessian_at_value

def get_derivative_matrix(inequality_constraints, x):
    '''
    This is used to compute Df(x) matrix on page 609
    '''
    grads = []
    for i,exp in enumerate(inequality_constraints):
        grads.append(compute_gradient(exp, x))
    Df = np.array(grads)
    return Df

def get_fx_matrix(inequality_constraints, x):
    '''
    This is the f(X) matrix defined on p. 609
    '''
    inequalities = []
    for i,exp in enumerate(inequality_constraints):
        inequalities.append(compute_function_value(exp, x))
    fx = np.array(inequalities)
    return fx

def get_rt_vector(objective_grad_at_x, Df, fx, A, x, b2, ineq_dual_values, eq_dual_values, t):
    '''
    This is the rt(x,lambda,v) vector shown on p. 609
    '''
    t = t
    r_dual = objective_grad_at_x + Df.T @ ineq_dual_values + A.T @ eq_dual_values
    #print(f"r_dual .. {r_dual}")
    r_cent = -1 * np.diag(ineq_dual_values) @ fx - (1/t) * np.ones(len(ineq_dual_values))
    #print(f"r_cent .. {r_cent}")
    r_pri = A @ x - b2
    #print(f"r_pri .. {r_pri}")
    rt = np.concat((r_dual, r_cent, r_pri))
    return rt

def get_primal_dual_matrix(objective_hessian_at_x, inequality_constraints, x, Df, fx, A, ineq_dual_values):
    '''
    This is the matrix for the system of equations defined on page 610
    '''
    sm = objective_hessian_at_x.copy()
    for i,exp in enumerate(inequality_constraints):
        #print(inequality_constraints[i])
        #print(ineq_dual_values[i])
        sm += ineq_dual_values[i] * compute_hessian(inequality_constraints[i], x)
    row1 = np.concatenate((sm, Df.T, A.T), axis = 1)

    row21 = np.diag(ineq_dual_values) @ Df
    row22 = -1 * np.diag(fx) 
    row23 = np.zeros((len(ineq_dual_values), A.T.shape[1]))
    row2 = np.concat((row21, row22, row23), axis = 1)

    row31 = A
    row32 = np.zeros((A.shape[0], len(ineq_dual_values)))
    row33 = np.zeros((A.shape[0], A.T.shape[1]))
    row3 = np.concat((row31, row32, row33), axis = 1)

    primal_dual_matrix = np.concat((row1, row2, row3), axis = 0)
    return primal_dual_matrix

def get_t_for_primal_dual_method(inequality_constraints, x, ineq_dual_values):
    '''
    This is step 1 of the algorithm outlined on page 612 of Boyd (primal dual interior point method)
    '''
    u = 10 

    fx = get_fx_matrix(inequality_constraints, x)

    n = -1 * fx.T @ ineq_dual_values
    m = len(inequality_constraints)

    t = u * (m/n)
    return t

def get_updated_values(objective_function, inequality_constraints, A, b2, x, ineq_dual_values, eq_dual_values, t):
    '''
    This is step 2 & step 3 of the algorithm outlined on page 612 of Boyd (primal dual interior point method)
    I put this in one function because the data that goes into the third step is already calculated in the second step and I didn't want to have to return that data to plug it into the third step. 
    So just made another function to deal with it that has access to the infomation defined in the outer function. 
    '''
    objective_grad_at_x = compute_gradient(objective_function, x)
    objective_hessian_at_x = compute_hessian(objective_function, x)
    Df = get_derivative_matrix(inequality_constraints, x)
    fx = get_fx_matrix(inequality_constraints, x)
    
    
    # This is the system of equations that is solved to get the primal dual direction
    primal_dual_matrix = get_primal_dual_matrix(objective_hessian_at_x, inequality_constraints, x, Df, fx, A, ineq_dual_values)
    r_t = get_rt_vector(objective_grad_at_x, Df, fx, A, x, b2, ineq_dual_values, eq_dual_values, t)
    rhs = -1 * r_t
    primal_dual_direction = np.linalg.inv(primal_dual_matrix) @ rhs

    def get_primal_dual_step_length():
        '''
        This is step three of the algorithm
        '''
        a = .1
        b = .5


        # This block of code is where s is initially defined. All that may happen from here is that it will get scaled by b
        delta_lambda = primal_dual_direction[len(x): len(x) + len(ineq_dual_values)] #np.array([-1.5,-1.]) 
        indices_where_delta_lambda_less_than_zero = list(np.where(delta_lambda < 0)[0])
        num_indices = len(indices_where_delta_lambda_less_than_zero)
        s_prime = [1.]
        if num_indices > 0:
            div = -1 * ineq_dual_values[indices_where_delta_lambda_less_than_zero] / delta_lambda[indices_where_delta_lambda_less_than_zero]
            s_prime.append(float(min(list(div))))
        s_max = min(s_prime)
        s = .99 * s_max

        
        # These start out as false because otherwise the code that goes in the loop needs to be out of the loop as well
        norm_condition = False
        strictly_feasible_condition = False
        current_state = np.concat((x, ineq_dual_values, eq_dual_values))
        while not ( strictly_feasible_condition and norm_condition ):
            s = b * s
            next_state = current_state + s * primal_dual_direction
            
            # recalculating rt for the new vector x + s * primal_dual_direction
            objective_grad_at_next_state = compute_gradient(objective_function, next_state[:len(x)]) 
            ineq_dual_values_at_next_state = next_state[len(x): len(x) + len(ineq_dual_values)]
            eq_dual_values_at_next_state = next_state[len(x) + len(ineq_dual_values):]
            Df = get_derivative_matrix(inequality_constraints, next_state[:len(x)])
            fx = get_fx_matrix(inequality_constraints, next_state[:len(x)])
            r_t_next_state = get_rt_vector(objective_grad_at_next_state, Df, fx, A, next_state[:len(x)], b2, ineq_dual_values_at_next_state, eq_dual_values_at_next_state, t)
            r_t_norm = np.linalg.norm(r_t)
            r_t_next_state_norm = np.linalg.norm(r_t_next_state)

            # update flags to see if another loop is required
            norm_condition = r_t_next_state_norm <= (1 - a*s) * r_t_norm
            #print(f"norm condition is .. {norm_condition}")
            strictly_feasible_condition = ( fx < 0 ).all()
            #print(f"striclty feasible condition is..{strictly_feasible_condition}")
        return [next_state, r_t_next_state]
        
    info = get_primal_dual_step_length()
    next_state, r_t_next_state = info[0], info[1]
    return [next_state, r_t_next_state]

def primal_dual_solver(objective_function, A, x, b2, inequality_constraints, ineq_dual_values, eq_dual_values):
    '''
    this is the solver that uses the primal dual interior point method described on page 612 of Boyd
    '''
    feasible_e = .0001
    e = .0001
    
    primal_feas_condition = False
    dual_feas_condition = False
    duality_gap_condition = False

    iterations = 0 
    while not ( primal_feas_condition and dual_feas_condition and duality_gap_condition ):
        # print current value of f for reference
        iterations += 1
        if iterations > 500:
            print("500 iteration limit reached.. perhaps you gave an infeasible problem or otherwise did not put your problem in standard form?")
            break
        #print(f"iters... {iterations}")
        ##print(f"current value of x is .. {x}")
        objective_value = compute_function_value(objective_function, x)
        #print(f"objective is .. {objective_value:.2f}")
        # step 1 determine t
        t = get_t_for_primal_dual_method(inequality_constraints, x, ineq_dual_values)
        #print(t)
        # step 2 compute primal dual direction and also backtrack to find s and return new value of x
        info = get_updated_values(objective_function, inequality_constraints, A, b2, x, ineq_dual_values, eq_dual_values, t)
        next_state = info[0]
        x = next_state[:len(x)]
        ineq_dual_values = next_state[len(x): len(x) + len(ineq_dual_values)]
        eq_dual_values = next_state[len(x) + len(ineq_dual_values):]
        # do this until these conditions are all true
        r_t_next_state = info[1]


        r_pri_norm = np.linalg.norm(r_t_next_state[:len(x)])
        r_dual_norm = np.linalg.norm(r_t_next_state[len(x): len(x) + len(ineq_dual_values)])
        duality_gap = -1*get_fx_matrix(inequality_constraints, x).T @ ineq_dual_values

        primal_feas_condition = r_pri_norm <= feasible_e
        dual_feas_condition = r_dual_norm <= feasible_e
        duality_gap_condition = duality_gap <= e
    print(f"current value of x is .. {x}")
    print(f"objective is .. {objective_value:.2f}")
    
