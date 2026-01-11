import numpy as np

epsilon = 1e-7

# Add artificial variables to find a feasible basis
# We are minimizing sum(a) s.t. Ax + Ia = b
def compute_feasable_solution(lp, A, b):
    num_constraints = lp.num_rows

    # Append a m*m identity matrix to A
    A_p1 = np.hstack([A, np.eye(num_constraints)])
    # We are minimizing sum(a)
    c_p1 = np.concatenate([np.zeros(lp.num_columns), np.ones(num_constraints)])
    # Set initial basis to the indices of artificial variables
    basis_p1 = np.arange(lp.num_columns, lp.num_columns + num_constraints)
    
    # Solve Phase I (minimizing sum of artificial variables)
    result = solve_with_basis(A_p1, b, c_p1, basis_p1, True) 
    basis = result['basis']
    return basis

def solve(lp):
    # Maximisation problem
    c = np.array(lp.objective)
    
    # Decide whether this is a minimisation problem
    minimize = lp.sense == 'minimize'
    if not minimize:
        c = -c

    # Init matrix A and rhs vector b
    A = np.zeros((lp.num_rows, lp.num_columns))
    for i, constraint in enumerate(lp.constraints):
        for j, coefficient in constraint['coefficients'].items():
            j = int(j)
            A[i, j] = float(coefficient)
    b = np.array([c['rhs'] for c in lp.constraints])

    # Basis
    basis = None
    if lp.has_basis:
        basis = np.array(lp.basis)
    else:
        basis = compute_feasable_solution(lp, A, b)

    return solve_with_basis(A, b, c, basis, minimize)


def solve_with_basis(A, b, c, basis, minimize=True):
    max_iterations = 10000

    for _ in range(max_iterations):
        # Line 2
        N = np.setdiff1d(np.arange(A.shape[1]), basis)

        # Line 3
        A_basis = A[:, basis]
        x_basis = np.linalg.solve(A_basis, b)

        # Line 4
        c_basis = c[basis]
        y = np.linalg.solve(A_basis.T, c_basis)
        A_nonbasic = A[:, N]
        c_nonbasic = c[N]
        c_bar_N = c_nonbasic - np.dot(A_nonbasic.T, y)

        # Line 5
        if np.all(c_bar_N >= -epsilon):
            x = np.zeros(A.shape[1])
            x[basis] = x_basis

            if minimize:
                dual = y
            else: # Maximize
                dual = -y
            return {
                "status": "optimal",
                "primal": x,
                "dual": dual,
                "ray": None,
                "farkas": None,
                "basis": basis
            }

        # Line 6
        entering_candidates = N[c_bar_N < -epsilon]
        k = np.min(entering_candidates)

        # Line 7
        A_k = A[:, k]
        d_B = np.linalg.solve(A_basis, -A_k)
        d = np.zeros(A.shape[1])
        d[basis] = d_B
        d[k] = 1.0

        # Line 8
        if np.all(d_B >= -epsilon):
            return {
                "status": "unbounded",
                "primal": None,
                "dual": None,
                "ray": d,
                "farkas": None,
                "basis": basis
            }

        # Line 9
        j_mask = d_B < -epsilon
        ratios = -x_basis[j_mask] / d_B[j_mask]

        # Line 10
        candidate_indices_in_basis = np.where(j_mask)[0][np.argmin(ratios)]
        l = basis[candidate_indices_in_basis]

        # Line 11
        new_basis = np.append(basis[basis != l], k)
        basis = np.array(sorted(new_basis), dtype=int)

    return {
        "status": "limit reached",
        "primal": None,
        "dual": None,
        "ray": None,
        "farkas": None,
        "basis": basis}