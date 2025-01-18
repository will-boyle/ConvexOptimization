import numpy as np
np.set_printoptions(precision=2, suppress=True)
import autograd.numpy as ag
from autograd import grad, jacobian

'''
Algorithms contained: 
    1) simplex method (two phase method)
    2) branch and bound (used to solve integer lps)
    3) primal dual method (for more general Convex problems .. assumes standard form .. from Boyd and Vandenberghe). Requires inequaltiies to be met strictly at each iteration I believe
       # has my own tweaks to how the algorithm works as well.

'''

       
def prompt_user_for_inequalities(dimension_of_x):
    '''
    Prompt user of inequalities, if none, creates dummy inequalities
    '''
    # example of a constraints: ["x[0]**2 + x[1]**2 - 1000", "2*x[0]**2 + 2*x[1]**2 - 1000"]
    inequality_constraints = input("Enter your inequality constraints, these are assumed to be <= 0 so you just enter the LHS. \nEx: x[0]**2 + x[1]**2 -10, x[0]**4 + 2*x[1]**2 -10.\nIf you do not have any inequality constraints, type n ") 
    if inequality_constraints != 'n':
        inequality_constraints = inequality_constraints.split(",")
    else: 
        inequality_constraints = create_dummy_inequalities(dimension_of_x)
    return inequality_constraints


def prompt_user_for_A_matrix():
    A = input("Enter A matrix. Values in rows separated by comma, begin next row with semi-colon.\nIf there is no A matrix, type n ")
    if A == 'n':
        return A
    else:
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
            #print("OPTIMAL SOLUTION...{}".format(c @ x))
            #print("ARGMIN...", x)
            #print(f"the problem was solved in {counter} interations")
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
    print(f"OPTIMAL SOLUTION...{cx}")
    print(f"ARGMIN... {x}")
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

#- - - - - - - - - - - - - - - - - - - - - - CODE BELOW HERE FOR CVX PRIMAL DUAL - - - - - - - - - - - - 

def create_dummy_inequalities(num_variables):
    '''This is the ugliest way I could think of to make dummy inequalities, 
    so naturally that is what I opted for.
    The idea here is that we need inequalities for the algorithm to work as is, so I need to create 
    and inequality that is equivalent to having no indequality. I have opted for 0 * x <= 0. However, if 
    I do that it won't be strictly feasible which creates issues, hence the -10 term. 

    TLDR
    used to create dummy inequalities for when user does not supply any. 
    '''
    terms = [f"0.01*x[{i}]+" for i in range(num_variables)]
    terms.append("-10")
    result_string = " + ".join(terms)
    return [result_string]

def create_inequalities_for_phase_one(inequalities, dim_of_x):
    '''
    This is for the primal dual solver. Basically, I am going to create a slack variable to make the original problem strictly feasible in an extra dimension.
    Need to do that if I want to start my initial solution as the zero vector for the algorithm. 
    '''
    new_inequalities = []
    for ineq in inequalities:
        new_inequalities.append(ineq + f" -1*x[{dim_of_x}]")

    return new_inequalities

def create_objective_for_phase_one(dim_of_x):
    '''
    This creates the objective function for the phase one optimization problem:
    min dummy_vars ** 2
    st...
    '''
    str = ""
    #str += f"1*x[{dim_of_x}]**2"
    # if you choose - x**2 then no longer a convex problem, perhaps should be x[0] + 2x[1]... + x[n+1]**1
    for i in range(dim_of_x):
        str += f"{i}*x[{i}]**2+"

    str += f"1*x[{dim_of_x}]**2"
    print(str)
    return str


def create_x_for_phase_one(dim_of_x):
    '''
    Need to make a slack variable basically x1 ... - huge slack var <= 0 will be strictly feasible because of this. 
    I have each x_i to be different so that I don't get a singular matrix. 
    '''
    new_x = np.ones(dim_of_x + 1)
    for i, el in enumerate(new_x):
        # the goal of this is to make sure the initial solution has different values, otherwise the primal dual matrix is liable to be singular by having columns which repeat
        new_x[i] = i
    
    new_x[-1] = 10000
    return new_x

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
    rt = np.concatenate((r_dual, r_cent, r_pri))
    return rt

def get_rt_vector2(objective_grad_at_x, Df, fx, x, ineq_dual_values, t):
    '''
    This is the rt(x,lambda,v) vector shown on p. 609. 
    The difference between this one and the other is that this one is for problems with no equality constraints.
    '''
    t = t
    r_dual = objective_grad_at_x + Df.T @ ineq_dual_values
    #print(f"r_dual .. {r_dual}")
    r_cent = -1 * np.diag(ineq_dual_values) @ fx - (1/t) * np.ones(len(ineq_dual_values))
    #print(f"r_cent .. {r_cent}")
    rt = np.concatenate((r_dual, r_cent))
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
    row2 = np.concatenate((row21, row22, row23), axis = 1)

    row31 = A
    row32 = np.zeros((A.shape[0], len(ineq_dual_values)))
    row33 = np.zeros((A.shape[0], A.T.shape[1]))
    row3 = np.concatenate((row31, row32, row33), axis = 1)

    primal_dual_matrix = np.concatenate((row1, row2, row3), axis = 0)
    return primal_dual_matrix

def get_primal_dual_matrix2(objective_hessian_at_x, inequality_constraints, x, Df, fx, ineq_dual_values):
    '''
    This is the matrix for the system of equations defined on page 610
    This is the same as the other method except is for problems without an A matrix.
    '''
    sm = objective_hessian_at_x.copy()
    for i,exp in enumerate(inequality_constraints):
        #print(inequality_constraints[i])
        #print(ineq_dual_values[i])
        sm += ineq_dual_values[i] * compute_hessian(inequality_constraints[i], x)
    row1 = np.concatenate((sm, Df.T), axis = 1)
    row21 = np.diag(ineq_dual_values) @ Df
    row22 = -1 * np.diag(fx) 
    row2 = np.concatenate((row21, row22), axis = 1)

    primal_dual_matrix = np.concatenate((row1, row2), axis = 0)
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
        current_state = np.concatenate((x, ineq_dual_values, eq_dual_values))
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

def get_updated_values2(objective_function, inequality_constraints, x, ineq_dual_values, t):
    '''
    This is step 2 & step 3 of the algorithm outlined on page 612 of Boyd (primal dual interior point method)
    I put this in one function because the data that goes into the third step is already calculated in the second step and I didn't want to have to return that data to plug it into the third step. 
    So just made another function to deal with it that has access to the infomation defined in the outer function. 

    This is the same as the above except meant for problems without equality constraints. This was the least elegant implementation so that is what I opted for. 
    '''
    objective_grad_at_x = compute_gradient(objective_function, x)
    objective_hessian_at_x = compute_hessian(objective_function, x)
    Df = get_derivative_matrix(inequality_constraints, x)
    fx = get_fx_matrix(inequality_constraints, x)
    
    
    # This is the system of equations that is solved to get the primal dual direction
    primal_dual_matrix = get_primal_dual_matrix2(objective_hessian_at_x, inequality_constraints, x, Df, fx, ineq_dual_values)
    r_t = get_rt_vector2(objective_grad_at_x, Df, fx, x, ineq_dual_values, t)
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
        current_state = np.concatenate((x, ineq_dual_values))
        while not ( strictly_feasible_condition and norm_condition ):
            s = b * s
            next_state = current_state + s * primal_dual_direction
            
            # recalculating rt for the new vector x + s * primal_dual_direction
            objective_grad_at_next_state = compute_gradient(objective_function, next_state[:len(x)]) 
            ineq_dual_values_at_next_state = next_state[len(x): len(x) + len(ineq_dual_values)]
            Df = get_derivative_matrix(inequality_constraints, next_state[:len(x)])
            fx = get_fx_matrix(inequality_constraints, next_state[:len(x)])
            r_t_next_state = get_rt_vector2(objective_grad_at_next_state, Df, fx, next_state[:len(x)], ineq_dual_values_at_next_state, t)
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
    feasible_e = .000001
    e = .000001
    
    primal_feas_condition = False
    dual_feas_condition = False
    duality_gap_condition = False

    iterations = 0 
    while not ( primal_feas_condition and dual_feas_condition and duality_gap_condition ):
        # print current value of f for reference
        iterations += 1
        limit = 50
        if iterations > limit:
            print(f"{limit} iteration limit reached.. perhaps you gave an infeasible problem or otherwise did not put your problem in standard form?")
            break
        #print(f"iters... {iterations}")
        #print(f"current value of x is .. {x}")
        objective_value = compute_function_value(objective_function, x)
        #print(f"objective is .. {objective_value:.2f}")
        # step 1 determine t
        t = get_t_for_primal_dual_method(inequality_constraints, x, ineq_dual_values)
        #print(f"value of t is .. {t}")
        # step 2 compute primal dual direction and also backtrack to find s and return new value of x
        info = get_updated_values(objective_function, inequality_constraints, A, b2, x, ineq_dual_values, eq_dual_values, t)
        next_state = info[0]
        x = next_state[:len(x)]
        #print(f"new value of x is .. {x}")
        ineq_dual_values = next_state[len(x): len(x) + len(ineq_dual_values)]
        #print("hello...")
        #print(f"ineq dual values {float(ineq_dual_values):.7f}")
        eq_dual_values = next_state[len(x) + len(ineq_dual_values):]
        #print(f"eq dual values {eq_dual_values}")
        # do this until these conditions are all true
        r_t_next_state = info[1]
        #print(f"rt next state is... {r_t_next_state}")
        #print(f"r pri is.. {r_t_next_state[-len(x):]}")
        r_pri_norm = np.linalg.norm(r_t_next_state[-len(x):])
        r_dual_norm = np.linalg.norm(r_t_next_state[ -(len(x) + len(ineq_dual_values)) : -len(x)])
        duality_gap = -1*get_fx_matrix(inequality_constraints, x).T @ ineq_dual_values
        #print(f"r pri norm: {r_pri_norm}, r dual norm: {r_dual_norm}, duality gap: {duality_gap}")
        #print(f"Ax - b .. {A @ x - b2}")
        primal_feas_condition = r_pri_norm <= feasible_e
        dual_feas_condition = r_dual_norm <= e
        duality_gap_condition = duality_gap <= e
    print(f"ARGMIN IS .. {x}")
    print(f"OPTIMAL SOLUTION IS .. {objective_value:.2f}")
    return [x, ineq_dual_values, eq_dual_values]

def primal_dual_solver2(objective_function, x, inequality_constraints, ineq_dual_values):
    '''
    this is the solver that uses the primal dual interior point method described on page 612 of Boyd

    The same code as above except does not consider equality constraints
    '''
    feasible_e = .000001
    e = .000001
    
    dual_feas_condition = False
    duality_gap_condition = False

    iterations = 0 
    while not ( dual_feas_condition and duality_gap_condition ):
        # print current value of f for reference
        iterations += 1
        limit = 100
        if iterations > limit:
            print(f"{limit} iteration limit reached.. perhaps you gave an infeasible problem or otherwise did not put your problem in standard form?")
            break
        #print(f"iters... {iterations}")
        #print(f"current value of x is .. {x}")
        objective_value = compute_function_value(objective_function, x)
        #print(f"objective is .. {objective_value:.2f}")
        # step 1 determine t
        t = get_t_for_primal_dual_method(inequality_constraints, x, ineq_dual_values)
        #print(f"value of t is .. {t}")
        # step 2 compute primal dual direction and also backtrack to find s and return new value of x
        info = get_updated_values2(objective_function, inequality_constraints, x, ineq_dual_values, t)
        next_state = info[0]
        x = next_state[:len(x)]
        #print(f"new value of x is .. {x}")
        ineq_dual_values = next_state[len(x): len(x) + len(ineq_dual_values)]
        #print("hello...")
        #print(f"ineq dual values {float(ineq_dual_values):.7f}")
        #print(f"eq dual values {eq_dual_values}")
        # do this until these conditions are all true
        r_t_next_state = info[1]
        #print(f"rt next state is... {r_t_next_state}")
        #print(f"r pri is.. {r_t_next_state[-len(x):]}")
        r_dual_norm = np.linalg.norm(r_t_next_state[ -(len(x) + len(ineq_dual_values)) : -len(x)])
        duality_gap = -1*get_fx_matrix(inequality_constraints, x).T @ ineq_dual_values
        #print(f"r dual norm: {r_dual_norm}, duality gap: {duality_gap}")
        dual_feas_condition = r_dual_norm <= e
        duality_gap_condition = duality_gap <= e
    print(f"ARGMIN IS .. {x}")
    print(f"OPTIMAL SOLUTION IS .. {objective_value:.2f}")
    return [x, ineq_dual_values]

def solve_convex_problem_without_equality_constraints(dim_of_x, objective_function):
    # there is always inequalities passed to solver, they are either user created or dummy inequalities. 
    # this branch is assuming that there is no A matrix. So there are two cases, either we have inequalities or we do not. If ienqualities, we need to solve phase one, else we don't 
    # example of a constraints: ["x[0]**2 + x[1]**2 - 1000", "2*x[0]**2 + 2*x[1]**2 - 1000"]
    inequality_constraints = prompt_user_for_inequalities(dim_of_x)

    phase_one_objective_function = create_objective_for_phase_one(dim_of_x)
    x = create_x_for_phase_one(dim_of_x)
    phase_one_inequality_constraints = create_inequalities_for_phase_one(inequality_constraints, dim_of_x)
    ineq_dual_values = np.full(len(phase_one_inequality_constraints), 1.)

    phase_one_solution = primal_dual_solver2(phase_one_objective_function, x, phase_one_inequality_constraints, ineq_dual_values)
    feasible_solution = phase_one_solution[0][:-1] # last index is dummy variable
    feasible_solution_ineq_dual_values = phase_one_solution[1]
        
    phase_two_solution = primal_dual_solver2(objective_function, feasible_solution, inequality_constraints, feasible_solution_ineq_dual_values)

def solve_convex_problem_with_equality_constraints(dim_of_x, objective_function, A):
    # there are always inequalities passed to solver. Either user defined or dummy. 
    # worked when I entered an A matrix and inequalities
    # failed when I entered A matrix and no inequalities
    # fails whenever there are no inequalities
    b = prompt_user_for_b_vector()
    inequality_constraints = prompt_user_for_inequalities(dim_of_x)
    phase_one_objective_function = create_objective_for_phase_one(dim_of_x)
    x = create_x_for_phase_one(dim_of_x)
    phase_one_inequality_constraints = create_inequalities_for_phase_one(inequality_constraints, dim_of_x)
    ineq_dual_values = np.full(len(phase_one_inequality_constraints), 1.)
    eq_dual_values = np.zeros(A.shape[0])
    # you don't need to feed in A matrix, we are just trying to find a feasible solution first for the inequalities, algo can't work without that. 
    phase_one_solution = primal_dual_solver2(phase_one_objective_function, x, phase_one_inequality_constraints, ineq_dual_values)
    feasible_solution = phase_one_solution[0][:-1] # last index is dummy variable
    feasible_solution_ineq_dual_values = phase_one_solution[1]
        
    phase_two_solution = primal_dual_solver(objective_function, A, feasible_solution, b, inequality_constraints, feasible_solution_ineq_dual_values, eq_dual_values)

# Library to solve quadratic programming problems
# Algorithms contained
#     1) QP2 - solver for quadratic program with linear equality constraints, where rows of constraint
#       matrix are linearly independent. Algorithm requires an initial
#       solution. This algorithm referred to as QP2, since 
#       it is the second algorithm described in Quad. Prog. w Computer Programs by Michael Best. 

# NOTE in QP -1 corresponds to -2 here and 0 to -1. In regards to their meaning for J 



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
            print("algorithm has successfully terminated by grad, solution is..", x)
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
