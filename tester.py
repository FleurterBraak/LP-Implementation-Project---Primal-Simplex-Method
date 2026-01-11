from simplex import solve
from lp import LP
import numpy as np
import json

EXAMPLE_FILE = "examples/orig-basis-scagr25.json"
with open(EXAMPLE_FILE) as f:
    data = json.load(f)
    lp = LP(data)
    result = solve(lp)
    print(f"status: {result['status']}")
    print(f'primal: {result['primal']}')
    print(f"dual: {result['dual']}")
    print(f"ray: {result['ray']}")
    print(f"farkas: {result['farkas']}")
    print(f"basis: {result['basis']}")
    print(f"objective value: {np.dot(np.array(lp.objective), result['primal'])}")
    print(f"objective dual value: {np.dot(np.array([c['rhs'] for c in lp.constraints]), result['dual'])}")