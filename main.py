import numpy as np
import ConvexOptimization as coop


print(".........................WELCOME TO THE OPTIMIZATION COMMAND LINE TOOL............................")
print("--------------------------------------------------------------------------------------------------")

print("The types of problems you can solve are listed below:")

choices = ["LP", "ILP  (integer constrained LP)", "CVX (convex problem with convex objective, <= 0 inequalities, and a linear equality system"]
for choice in choices:
    print(choice)



problem_type = input("Which type of problem are you trying to solve? ")

if ( problem_type == 'LP' ) or ( problem_type == 'ILP' ):

    A = coop.prompt_user_for_A_matrix()
    b = coop.prompt_user_for_b_vector()
    c = input("Enter the objective function (ex: '1,1' for x1 + x2): ")
    c = [float(x) for x in c.split(",")]
    c = np.array(c)
    inequality_types = input("Please enter the types inequalites related to each constraint (ex: '==, <=, >=' ) etc. ")
    inequality_types = inequality_types.split(",")


    if problem_type == 'LP':
        coop.solve_lp(c, A, inequality_types, b)
    elif problem_type == 'ILP':
        coop.branch_and_bound_algorithm(c, A, inequality_types, b)

elif problem_type == 'CVX':

    dimension_of_x = int(input("how many variables does your problem have? ex: 2 "))
    x = np.zeros(int(dimension_of_x))

    objective_function = input("Enter you objective function ( ex: x[0]**4 + x[1]**2 ) ")    #"x[0]**4 + x[1]**2"  # Function as a string


    # example of a constraints: ["x[0]**2 + x[1]**2 - 1000", "2*x[0]**2 + 2*x[1]**2 - 1000"]
    inequality_constraints = input("Enter your inequality constraints, these are assumed to be <= 0 so you just enter the LHS. \nEx: x[0]**2 + x[1]**2 -10, x[0]**4 + 2*x[1]**2 -10.\nIf you do not have any inequality constraints, type n ") 
    if inequality_constraints != 'n':
        inequality_constraints = inequality_constraints.split(",")
    else: 
        inequality_constraints = coop.create_dummy_inequalieties(dimension_of_x)
        print(inequality_constraints)
    
    A = coop.prompt_user_for_A_matrix()
    if A != 'n':
        b2 = coop.prompt_user_for_b_vector()    
    else:
        # idea is that 0 x = 0, so if I make a dummy A matrix that is all zeros I don't have to change algo. However, if a row is all zeros then matrix is singular, so need A to approximate zeros without being all zeros.
        A = np.full( (1, len(x) ), .00001 )
        b2 = np.array(0.0)

     

    

    ineq_dual_values = np.full(len(inequality_constraints), .00001)
    eq_dual_values = np.zeros(A.shape[0])

    coop.primal_dual_solver(objective_function, A, x, b2, inequality_constraints, ineq_dual_values, eq_dual_values)
