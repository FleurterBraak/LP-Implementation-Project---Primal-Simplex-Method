#In deze file kunnen we lekker klooien zonder dat we werkende codes verpesten jeej

from lp import LP
import numpy as np
import json


def solve(lp):
    max_iterations = 1000
    epsilon = 1e-7

    # Maximisation problem
    c = np.array(lp.objective)
    if lp.sense == 'maximize':
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
        # TODO
        pass
    print(f"Basis: {basis}")

    for _ in range(max_iterations):
        # Line 2
        N = np.setdiff1d(np.arange(lp.num_columns), basis)

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
            x = np.zeros(lp.num_columns)
            x[basis] = x_basis
            return {
                "status": "optimal",
                "primal": x,
                "dual": None,
                "ray": None,
                "farkas": None,
                "basis": basis}

        # Line 6
        entering_candidates = N[c_bar_N < -epsilon]
        k = np.min(entering_candidates)

        # Line 7
        A_k = A[:, k]
        d_B = np.linalg.solve(A_basis, -A_k)
        d = np.zeros(lp.num_columns)
        d[basis] = d_B
        d[k] = 1.0

        # Line 8
        if np.all(d_B >= -epsilon):
            return {
                "status": "unbounded",
                "primal": x,
                "dual": None,
                "ray": d,
                "farkas": None,
                "basis": basis}

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
        "primal": x,
        "dual": None,
        "ray": None,
        "farkas": None,
        "basis": basis}

EXAMPLE_FILE = "examples/BT-Example-3.6-std.json"
with open(EXAMPLE_FILE) as f:
    data = json.load(f)
    lp = LP(data)
    print(lp.constraints)
    result = solve(lp)
    print(f"status: {result['status']}")
    print(f'primal: {result['primal']}')
    print(f"dual: {result['dual']}")
    print(f"ray: {result['ray']}")
    print(f"farkas: {result['farkas']}")
    print(f"basis: {result['basis']}")
    print(f"objective value: {np.dot(np.array(lp.objective), result['primal'])}")
    #print(f"Outcome: {outcome}")
    #if outcome.lower() == "optimal":
        #objective_value = np.dot(np.array(lp.objective), x)
        #print(f"Value: {objective_value}")

