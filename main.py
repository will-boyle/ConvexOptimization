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

    objective_function = input("Enter you objective function ( ex: x[0]**4 + x[1]**2 ) ")    #"x[0]**4 + x[1]**2"  # Function as a string


    # example of a constraints: ["x[0]**2 + x[1]**2 - 1000", "2*x[0]**2 + 2*x[1]**2 - 1000"]
    inequality_constraints = input("Enter your inequality constraints, these are assumed to be <= 0 so you just enter the LHS. Ex: x[0]**2 + x[1]**2 - 1000, 2*x[0]**2 + 2*x[1]**2 - 1000 ") 
    inequality_constraints = inequality_constraints.split(",")

    A = coop.prompt_user_for_A_matrix()

    b2 = coop.prompt_user_for_b_vector() 

    dimension_of_x = input("how many variables does your problem have? ex: 2 ")
    x = np.zeros(int(dimension_of_x))

    ineq_dual_values = np.array([0.1,0.1])
    eq_dual_values = np.array([0.1,0.1])

    coop.primal_dual_solver(objective_function, A, x, b2, inequality_constraints, ineq_dual_values, eq_dual_values)
