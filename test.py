
import numpy as np
import ConvexOptimization as coop

# the purpose of these tests is to understand why the cvx algo isn't solving LPs
# note the every LP has inequality constraints (the nonzero constraints) so we can ignore case where it doesn't. 

# problem 2.10 
#'''
dim_of_x = 4
objective_function = "-6*x[0] - 8*x[1] - 5*x[2] - 9*x[3]"
A = np.array([1,1,1,1])
b = 1
inequalities = ["-x[0],-x[1],-x[2],-x[3]"]
#'''


#A = coop.prompt_user_for_A_matrix()
#if not(isinstance(A, np.ndarray)):
if True:
    # there is always inequalities passed to solver, they are either user created or dummy inequalities. 
    # this branch is assuming that there is no A matrix. So there are two cases, either we have inequalities or we do not. If ienqualities, we need to solve phase one, else we don't 
    # example of a constraints: ["x[0]**2 + x[1]**2 - 1000", "2*x[0]**2 + 2*x[1]**2 - 1000"]
    inequality_constraints = inequalities

    phase_one_objective_function = coop.create_objective_for_phase_one(dim_of_x)
    x = coop.create_x_for_phase_one(dim_of_x)
    phase_one_inequality_constraints = coop.create_inequalities_for_phase_one(inequality_constraints, dim_of_x)
    ineq_dual_values = np.full(len(phase_one_inequality_constraints), 1.)

    phase_one_solution = coop.primal_dual_solver2(phase_one_objective_function, x, phase_one_inequality_constraints, ineq_dual_values)
    feasible_solution = phase_one_solution[0][:-1] # last index is dummy variable
    feasible_solution_ineq_dual_values = phase_one_solution[1]
        
    phase_two_solution = coop.primal_dual_solver2(objective_function, feasible_solution, inequality_constraints, feasible_solution_ineq_dual_values)
else:
    # there are always inequalities passed to solver. Either user defined or dummy. 
    # worked when I entered an A matrix and inequalities
    # failed when I entered A matrix and no inequalities
    # fails whenever there are no inequalities
    b = coop.prompt_user_for_b_vector()
    inequality_constraints = inequalities
    phase_one_objective_function = coop.create_objective_for_phase_one(dim_of_x)
    x = coop.create_x_for_phase_one(dim_of_x)
    phase_one_inequality_constraints = coop.create_inequalities_for_phase_one(inequality_constraints, dim_of_x)
    ineq_dual_values = np.full(len(phase_one_inequality_constraints), 1.)
    eq_dual_values = np.zeros(A.shape[0])
    # you don't need to feed in A matrix, we are just trying to find a feasible solution first for the inequalities, algo can't work without that. 
    phase_one_solution = coop.primal_dual_solver2(phase_one_objective_function, x, phase_one_inequality_constraints, ineq_dual_values)
    feasible_solution = phase_one_solution[0][:-1] # last index is dummy variable
    feasible_solution_ineq_dual_values = phase_one_solution[1]
        
    phase_two_solution = coop.primal_dual_solver(objective_function, A, feasible_solution, b, inequality_constraints, feasible_solution_ineq_dual_values, eq_dual_values)
