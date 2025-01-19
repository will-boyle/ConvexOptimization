import numpy as np
import ConvexOptimization as coop


print(".........................WELCOME TO THE OPTIMIZATION COMMAND LINE TOOL............................")
print("--------------------------------------------------------------------------------------------------")
print("if you see many solutions, the last one is always the actual solution. Usually in solving the problem you solve many subproblems. Will fix later")

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
    
    dim_of_x = int(input("how many variables does your problem have? ex: 2 "))
    objective_function = input("Enter you objective function ( ex: x[0]**4 + x[1]**2 ) ")    #"x[0]**4 + x[1]**2"  # Function as a string
    

    A = coop.prompt_user_for_A_matrix()
    if not(isinstance(A, np.ndarray)):
        coop.solve_convex_problem_without_equality_constraints(dim_of_x, objective_function)
    else:
        coop.solve_convex_problem_with_equality_constraints(dim_of_x, objective_function, A)
