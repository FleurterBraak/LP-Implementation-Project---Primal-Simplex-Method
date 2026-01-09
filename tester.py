from simplex import solve
from lp import LP
import numpy as np
import json

EXAMPLE_FILE = "examples/BT-Example-3.6-std.json"
with open(EXAMPLE_FILE) as f:
    data = json.load(f)
    lp = LP(data)
    print(lp.constraints)
    outcome, x, dir = solve(lp)
    print(f"Outcome: {outcome}")
    if outcome.lower() == "optimal":
        objective_value = np.dot(np.array(lp.objective), x)
        print(f"Value: {objective_value}")
