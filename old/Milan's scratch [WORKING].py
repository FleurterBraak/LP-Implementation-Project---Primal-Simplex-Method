#In deze file kunnen we lekker klooien zonder dat we werkende codes verpesten jeej

import json as json
import numpy as np

class LP:

    def __init__(self, data):
        with open(data) as file:
            lp = json.load(file)

        self.sense = lp['sense']
        self.c = np.array(lp['objective'])
        self.signs = np.array(lp['signs'])

        self.number_rows = len(lp['constraints'])
        self.number_columns = len(self.c)
        self.A = np.zeros((self.number_rows, self.number_columns))
        self.b = np.zeros(self.number_rows)

        for i, constraint in enumerate(lp['constraints']):
            for j, coefficient in constraint['coefficients'].items():
                j = int(j)
                self.A[i,j] = float(coefficient)
            self.b[i] = float(constraint['rhs'])

        self.basis = np.array(lp['basis'])
        nonbasic = []
        for j in range(self.number_columns):
            if j not in self.basis:
              nonbasic.append(j)
        self.nonbasic = np.array(nonbasic)

    def solve(self):
        max_iterations = 1000
        epsilon = 1e-7

        # Maximisation problem
        c = self.c
        if self.sense.lower() == 'maximize':
            c = -self.c
        
        for _ in range(max_iterations):
            # Line 2
            N = np.array(self.nonbasic)

            # Line 3
            A_basis = self.A[:, self.basis]
            x_basis = np.linalg.solve(A_basis, self.b)
            
            # Line 4
            c_basis = c[self.basis]
            y = np.linalg.solve(A_basis.T, c_basis)
            A_nonbasic = self.A[:, N]
            c_nonbasic = c[N]
            c_bar_N = c_nonbasic - np.dot(A_nonbasic.T, y)

            # Line 5
            if np.all(c_bar_N >= -epsilon):
                x = np.zeros(self.number_columns)
                x[self.basis] = x_basis
                return "optimal", x, self.basis

            # Line 6
            # Bland's Rule 
            entering_candidates = N[c_bar_N < -epsilon]
            k = np.min(entering_candidates)

            # Line 7
            A_k = self.A[:, k]
            d_B = np.linalg.solve(A_basis, -A_k)
            d = np.zeros(self.number_columns)
            d[self.basis] = d_B
            d[k] = 1.0

            # Line 8
            if np.all(d_B >= -epsilon):
                return "unbounded", x, d

            # Line 9
            j_mask = d_B < -epsilon
            ratios = -x_basis[j_mask] / d_B[j_mask]
    
            # Line 10
            l_idx = np.where(j_mask)[0][np.argmin(ratios)]
            l = self.basis[l_idx]

            # Line 11
            new_basis = []
            for j in self.basis:
                if j != l:
                    new_basis.append(j)
            new_basis.append(k)

            self.basis = np.array(sorted(new_basis), dtype=int)
            self.nonbasic=[]
            for j in range(self.number_columns):
                if j not in self.basis:
                    self.nonbasic.append(j)
        
        return "limit reached", x, self.basis



data = LP('BT-example-3.8-std.json')
outcome, x, basis_direction = data.solve()
print(f"Outcome: {outcome}")
if outcome.lower() == "optimal":
    objective_value = np.dot(data.c, x)
    print(f"Value: {objective_value}")