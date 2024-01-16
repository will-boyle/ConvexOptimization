# learned: when t = 0, the solution B^-1b = x does not apply to new basis
# basically, when t = 0, it means that B^-1b !>= and so isn't feasible.
# this is why it is necessary to always compute x* = x + t * d. Instead of
# with the new inverse matrix.  

# because the phase two problem has a lot of zeros in the last row, 
# you need to make sure the column vector associated with the z variable is in the basis I think...
# otherwise if you choose basic variables that aren't slack or dummy or the z variable, then the last row is all zeros and 
# the matrix will be nonsingular. 

'''
Algorithms contained: 
    1) simplex method (two phase method)
    2) branch and bound (used to solve integer lps)


'''
import numpy as np


'''
def simplex_algorithm(phase, c, A, x, b, b_vars, n_vars):

    B_cols = A[:, b_vars]
    B = np.concatenate([B_cols], axis = 1)

    N_cols = A[:, n_vars]
    N = np.concatenate([N_cols], axis = 1)

  
    flag = 0
    counter = 0
    
    while flag == 0:
        counter += 1
        print("  THIS IS THE CORRECT FILE")
        print("....................................................................................................iteration.....", counter)
        print("current value of cx", c @ x)
        print("value of Ax is......", A @ x)
    
        test = test_for_optimality(B, N, c, x, b_vars, n_vars)
        if np.all(test) == True:
            flag = 1
            print("the optimal solution is...",x)
            print("the algorithm has successfully terminated, the optimal solution is...{}".format(c @ x))
            print("optimal variables", b_vars)
            print("optimal basis", B)
            return [x, c @ x, b_vars, n_vars] 
        else: 
	    # chosen according to Bland's rule.
            indices_where_test_fails = np.where(test == False) 
            nonbasic_index_to_enter = n_vars[min(indices_where_test_fails)[0]]
   
            compute_direction_of_increase(B, N, nonbasic_index_to_enter, n_vars)   
	
            db = compute_direction_of_increase(B, N, nonbasic_index_to_enter, n_vars)[0] 
            dn = compute_direction_of_increase(B, N, nonbasic_index_to_enter, n_vars)[1]
	
            t = compute_amount_of_increase(phase, x, b_vars, n_vars, db, dn)[0]
            basic_index_to_leave = compute_amount_of_increase(phase, x, b_vars, n_vars, db, dn)[1]
            
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
'''


# for whatever reason, it doesn't seem to print the correct basis or basic variables... but does print the correct optimal value of cx and x. Something to look at when you get a minute
# this logic is being redone to handle phase 1 or phase 2 problem
# for the phase 2 problem, the matrix column assoicated with the z constriant must be in the basis. 
def simplex_algorithm(c, A, x, b, b_vars, n_vars):
    B_cols = A[:, b_vars]
    B = np.concatenate([B_cols], axis = 1)

    N_cols = A[:, n_vars]
    N = np.concatenate([N_cols], axis = 1)

    print("A matrix")
    print(A)
    flag = 0
    counter = 0
    
    while flag == 0:
        

        counter += 1
        print("iteration.....", counter)
        print("basic vars", b_vars)
        print("matrix", B)
        print("current value of x",x)
        print(A @ x)


        test = test_for_optimality(B, N, c, x, b_vars, n_vars)
        if np.all(test) == True:
            flag = 1
            print("the optimal solution is...",x)
            print("the algorithm has successfully terminated, the optimal solution is...{}".format(c @ x))
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
            print("t", t)
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

# developing below, trying to handle phase two issue leading to singular matrix...



'''
# computes min { -x/d_i, where d_i is negative }
# determines which basis variable becomes nonbasis, determines k* in Dan Solow 
def compute_amount_of_increase(phase, x, basic_vars, nonbasic_vars, db, dn):

    print("current value of x...",x)
    print(" ")
    print(" ")
    print(" "
    print(" ")

    # need to determine the indices whre db is negative.
    if phase == 'one':
        zero_vector = np.zeros(len(basic_vars))
        unbounded_test = db >= zero_vector
        indices = [ i for i, value in enumerate(basic_vars) ]
        print("indices", indices)

    else: 
        zero_vector = np.zeros(len(basic_vars) - 1)
        z_index = basic_vars.index(len(x) - 1)
        print("basic variables", basic_vars)
        print("nonbasic variables", nonbasic_vars)
        # z index in basic indices list
        #print("z index", z_index)
        print("db", db)
        indices = [ i for i, value in enumerate(basic_vars) if i != z_index]
        print("indices............", indices)
        unbounded_test = db[indices] >= zero_vector
        print("unbounded test....",unbounded_test)
    
    if np.all(unbounded_test) == True: 
        print("the problem is unbounded, the direction of unboundedness is db: {0}, dn = {1})".format(db, dn))     
        
    else:
        indices_where_db_is_negative = np.where(unbounded_test == False)

        #algorithm assumes x >= 0, hence t >= 0 
        t = np.min(-1 *
np.divide(x[basic_vars][indices][indices_where_db_is_negative],db[indices][indices_where_db_is_negative]))
        print("indices where db is negative", indices_where_db_is_negative)
        print("basic indices under consideration", x[basic_vars][indices][indices_where_db_is_negative])
        print("division.........", np.divide(x[basic_vars][indices][indices_where_db_is_negative],db[indices][indices_where_db_is_negative]))
       



        print("")
        print("basic values of x....", x[basic_vars])
        print("the basic values of x except for the z variable", x[basic_vars][indices])
        print("the indices of the above where they are negative", x[basic_vars][indices][indices_where_db_is_negative])
        print("")


        print("current value of basic variables", x[basic_vars][indices])
        xb_star = x[basic_vars] + t * db
        #print("new values of these basic variables", xb_star)
        # the basic index of the leaving variable, denoted k*, chosen by
        # Bland's rule
        if t != 0:
            k_star = np.min(np.where(xb_star[indices] == 0))
        else:
            k_star = np.min(np.where(xb_star[indices] == x[basic_vars][indices]))
        

        b = np.array(basic_vars)
        print(b[indices])
        print(k_star)
        k_star = b[indices][k_star]
        print(x[k_star])
        print("value of t............",t)

        return [t, k_star]
'''



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
            # was this... k_star = int(np.min(basic_vars_array[np.where(xb_star == 0)]))

        #else:
            #k_star = int(np.min(basic_vars_array[np.where(xb_star == x[basic_vars])]))
        print("")
        print("")
        print("")
        print("")
        print("")
        print("the basic variables are...", basic_vars)
        print("the value of the basic variables are", x[basic_vars])
        print("db vector is", db)
        print("xb_star", xb_star)
        print("the basic variable leaving the basis is...", k_star)
        
        print(x[basic_vars][np.where(xb_star == x[basic_vars])])
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


    print("............................... solving phase two problem...................................................................")  
    print("")
    print("")
    print("")
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
        print("candidate solutions...", candidate_solution)
        # implements LIFO structure because of depth first preference
        # do I want to be pulling out the last problem in the list? 
        problem = problems_to_solve.pop(-1)
        print("SOLVING PROBLEM.....")
        print(problem)
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
    print("the branch and bound algorithm has successfull terminated the optimal solution is....")
    return cx, x



    # if the problem is infeasible, we can prune this node and make it a leaf. 

    # if not, find first index which has integer component
    # create partition problem
    

# algorithm needs to be able to solve the relaxed problem. 
# needs to be able to test whether or not the solution is an integer solution
# if yes, then this is the optimal solution
# else, there exists an index where x is not integral
# create two LPs from this each using the same LP but adding the constraint
# x_j <= floor(x_j) or x_j = ceil(x_J). (this is a partition, there are no integer values between
# if it happens that adding one of the constraints creates an infeasible problem,
# then we can ignore this node and move on to the next node.

# three ways to prune. 1) if solution is integral, do not need to create any more nodes for 
# problem and therefore you have a leaf. 2) is solition is <= current best integer solution, then
# no reason to look for more nodes, hence a you have a leaf. 3) if solution is infeasible, 
# no need to keep searching.

# it seems to me that the rule for branching does not matter, so choose the first 
# index such that the solution is not integral. 
