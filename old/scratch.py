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
        max_iterations = 100
        for _ in range(max_iterations):
            N = self.nonbasic
            A_basis = self.A[:, self.basis]
            x_basis = np.linalg.solve(A_basis,self.b)
            c_basis = self.c[self.basis]
            y = np.linalg.solve(A_basis.T, c_basis)
            A_nonbasic = self.A[:, N]
            c_nonbasic = self.c[N]
            c_bar_N = c_nonbasic - np.dot(A_nonbasic.T, y)

            x = np.zeros(self.number_columns)
            x[self.basis] = x_basis
            if np.all(c_bar_N >=0):
                return "optimal", x, self.basis

            k_star = np.argmin(c_bar_N)
            k = N[k_star]
            A_k= self.A[:, k]
            d_B = np.linalg.solve(A_basis, -A_k)
            d = np.zeros(self.number_columns)
            d[self.basis] = d_B
            d[k] = 1.0

            if np.all(d_B >= 0):
                return "unbounded", x, d

            j = d_B < 0
            theta_star = np.min(-x_basis[j] / d_B[j])

            l_star = np.abs(x_basis + d_B * theta_star) < 1e-10
            l = self.basis[np.where(j & l_star)[0][0]]

            x = x + theta_star * d

            new_basis=[]
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



data = LP('BT-example-3.5-std.json')
outcome, x, basis_direction = data.solve()
objective_value = np.dot(data.c, x)
print(objective_value)
#A_basis = data.solve()
#print(data.A, data.b, data.c, data.signs, data.basis, data.nonbasic, A_basis)