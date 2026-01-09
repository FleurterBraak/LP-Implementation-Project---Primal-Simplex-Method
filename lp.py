class LP:
  '''Class for representing arbitrary linear programs (LPs).

  The LP is internally represented as a dictionary containing:
  - objective coefficients
  - constraints
  - variable signs
  - optimization sense (minimize or maximize)
  - optional basis information
  '''

  def __init__(self, data):
    '''Initialize an LP instance.

    Parameters
    ----------
    data : dict or str
        Either:
        - A dictionary representing the LP structure directly.
        - A JSON string encoding such a dictionary.

    Notes
    -----
    - If a dictionary is provided, ownership is transferred to the LP object.
    - If a JSON string is provided, it is parsed and coefficient keys are
      converted from strings to integers, with coefficients converted to floats.
    '''

    if isinstance(data, dict):
      self._data = data
    elif isinstance(data, str):
      import json
      self._data = json.loads(data)

      # In JSON, dict keys are always strings. Hence, we replace them by integers and convert the coefficients to floats.
      for constraint in self._data['constraints']:
        constraint['coefficients'] = { int(j):float(coef) for j,coef in constraint['coefficients'].items() }
    else:
      raise Exception(f'An LP can only be created from a dict or from a string (in JSON format), but not from {type(data)}.')

  def copy(self):
    '''Create a deep copy of the LP.

    Returns
    -------
    LP
        A new LP instance with duplicated internal data.
    '''

    import copy

    return LP(copy.deepcopy(self._data))

  def to_json(self):
    '''Serialize the LP into a JSON-formatted string.

    Returns
    -------
    str
        A JSON representation of the LP with indentation.

    Raises
    ------
    Exception
        If unsupported data types are encountered during serialization.
    '''

    import json
    import numpy

    def serialize(data):
      if isinstance(data, dict):
        result = {}
        for key,value in data.items():
          result[serialize(key)] = serialize(value)
        return result
      elif isinstance(data, list):
        return [ serialize(value) for value in data ]
      elif isinstance(data, (float,int,str)):
        return data
      elif isinstance(data, (numpy.int16, numpy.int32, numpy.int64)):
        return int(data)
      elif isinstance(data, (numpy.float32, numpy.float64, numpy.float128)):
        return float(data)
      else:
        raise Exception(f'Invalid data type <{type(data)}> of <{data}> encountered in LP.to_json() call.')

    return json.dumps(serialize(self._data), indent=2)

  def save(self, file_name):
    '''Save the LP to a file in JSON format.

    Parameters
    ----------
    file_name : str
        Path to the output file.
    '''

    with open(file_name, 'w') as fp:
      fp.write(self.to_json())
      fp.write('\n')

  @property
  def num_columns(self):
    '''Return the number of variables (columns) in the LP.'''
    return len(self._data['objective'])

  @property
  def num_rows(self):
    '''Return the number of constraints (rows) in the LP.'''
    return len(self._data['constraints'])

  @property
  def sense(self):
    '''Return the optimization sense ("minimize" or "maximize").'''
    return self._data['sense']

  @sense.setter
  def sense(self, new_sense):
    '''Set the optimization sense.

    Parameters
    ----------
    new_sense : str
        Either "minimize" or "maximize".
    '''

    self._data['sense'] = new_sense

  @property
  def objective(self):
    '''Return the objective coefficient vector.'''
    return self._data['objective']

  @objective.setter
  def objective(self, new_objective):
    '''Set the objective coefficient vector.'''
    self._data['objective'] = new_objective

  @property
  def signs(self):
    '''Return the variable sign restrictions.

    Each entry is:
    - +1 for a nonnegative variable
    - -1 for a nonpositive variable
    -  0 for a free variable
    '''

    return self._data['signs']

  @signs.setter
  def signs(self, new_signs):
    '''Set the variable sign restrictions.'''
    self._data['signs'] = new_signs

  @property
  def constraints(self):
    '''Return the list of constraints.'''
    return self._data['constraints']

  @constraints.setter
  def constraints(self, new_constraints):
    '''Set the list of constraints.'''
    self._data['constraints'] = new_constraints

  @property
  def basis(self):
    '''Return the basis indices, if present.'''
    return self._data['basis']

  @basis.setter
  def basis(self, new_basis):
    '''Set the basis indices.'''
    self._data['basis'] = new_basis

  @property
  def has_basis(self):
    '''Returns True if and only if a basis is present.'''
    return 'basis' in self._data

  def remove_basis(self):
    '''Remove the basis information from the LP, if it exists.'''
    if 'basis' in self._data:
      del self._data['basis']

  def __repr__(self):
    '''Return a human-readable string representation of the LP.

    Returns
    -------
    str
        A formatted textual description of the LP.
    '''

    output = [] # List of strings that will finally be concatenated.

    # minimize or maximize
    output.append(f'{self.sense} ({self.num_rows}-by-{self.num_columns})')

    # Objective vector
    for j,c_j in enumerate(self.objective):
      if c_j > 0.0:
        output.append(f' + {c_j}*x_{j}')
      elif c_j < 0.0:
        output.append(f' - {-c_j}*x_{j}')
    output.append('\nsubject to\n')

    # Constraints
    for constraint in self.constraints:
      for j,c_j in constraint['coefficients'].items():
        if c_j > 0.0:
          output.append(f' + {c_j}*x_{j}')
        elif c_j < 0.0:
          output.append(f' - {-c_j}*x_{j}')
      output.append(f' {constraint["relation"]}= {constraint["rhs"]}\n')

    # Variables
    nonneg_vars = [ j for j,sign in enumerate(self.signs) if sign == 1 ]
    nonpos_vars = [ j for j,sign in enumerate(self.signs) if sign == -1 ]
    free_vars = [ j for j,sign in enumerate(self.signs) if sign not in [-1, 1] ]
    if nonneg_vars:
      output.append('nonnegative variables: ' + ','.join(f'x_{j}' for j in nonneg_vars) + '\n')
    if nonpos_vars:
      output.append('nonpositive variables: ' + ','.join(f'x_{j}' for j in nonpos_vars) + '\n')
    if free_vars:
      output.append('free variables: ' + ','.join(f'x_{j}' for j in free_vars) + '\n')
    if 'basis' in self._data:
      output.append('basis: ' + ','.join(f'x_{j}' for j in self._data['basis']) + '\n')

    # Concatenate all strings and remove the final newline.
    return (''.join(output))[:-1]

  def check_consistent(self):
    '''Check whether the LP data structure is internally consistent.

    This method verifies:
    - Required keys are present.
    - Objective and sign vector lengths match.
    - The optimization sense is valid.
    - Variable signs are valid.
    - Constraint structure, relations, indices, and coefficient types are valid.

    Returns
    -------
    bool
        True if the LP is consistent.

    Raises
    ------
    Exception
        If any inconsistency or structural error is detected.
    '''

    for key in ('signs', 'objective', 'constraints', 'sense'):
      if key not in self._data:
        raise Exception(f'LP is inconsistent: {key} missing!')
    if len(self.signs) != len(self.objective):
      raise Exception(f'LP is inconsistent: {len(self.signs)} variable signs, but {len(self.objective)} objective coefficients.')
    if self.sense not in ('minimize', 'maximize'):
      raise Exception(f'LP is inconsistent: self.sense is "{self.sense}", but "minimize" or "maximize" is expected.')
    for j,sign in enumerate(self.signs):
      if sign not in (-1, 0, +1):
        raise Exception(f'LP is inconsistent: sign of x_{j} is {sign}, but -1, 0 or +1 is expected.')
    for i,cons in enumerate(self.constraints):
      if 'relation' not in cons:
        raise Exception(f'LP is inconsistent: constraint #{i} has no "relation" key.')
      if 'rhs' not in cons:
        raise Exception(f'LP is inconsistent: constraint #{i} has no "rhs" key.')
      if 'coefficients' not in cons:
        raise Exception(f'LP is inconsistent: constraint #{i} has no "coefficients" key.')
      if cons['relation'] not in ('<', '>', '='):
        raise Exception(f'LP is inconsistent: constraint #{i} has relation "{cons["relation"]}", but "<", ">" or "=" is expected.')
      for j,A_ij in cons['coefficients'].items():
        if not isinstance(j, int):
          raise Exception(f'LP is inconsistent: constraint #{i} has entry for non-integer column {j}.')
        if j < 0:
          raise Exception(f'LP is inconsistent: constraint #{i} has entry for column {j}.')
        if j >= self.num_columns:
          raise Exception(f'LP is inconsistent: constraint #{i} has entry for column {j}, but there are only {self.num_columns}.')
        if not isinstance(A_ij, float):
          raise Exception(f'LP is inconsistent: constraint #{i} has non-float coefficient {A_ij} for column {j}.')

    return True

  def primal_value(self, primal):
    '''Compute the objective value of a given primal solution.

    Parameters
    ----------
    primal : sequence of float
        Primal variable values.

    Returns
    -------
    float
        The objective value corresponding to the primal solution.
    '''
    return sum( primal[j] * obj for j,obj in enumerate(self.objective) )

  def primal_is_feasible(self, primal):
    '''Check feasibility of a primal solution.

    The method verifies:
    - Variable sign constraints.
    - Satisfaction of all primal constraints within a tolerance.

    Parameters
    ----------
    primal : sequence of float
        Primal variable values.

    Returns
    -------
    tuple
        (True, max_violation) if feasible,
        (False, error_message) if infeasible.

    Raises
    ------
    Exception
        If the primal vector has incorrect dimension.
    '''

    PRIMAL_EPSILON = 1.0e-6

    if len(primal) != self.num_columns:
      raise Exception(f'Given primal vector has {len(primal)} entries, but {self.num_columns} are expected.')

    max_violation = 0.0
    for j,sign in enumerate(self.signs):
      if sign == 1:
        if -primal[j] > PRIMAL_EPSILON:
          return (False, f'Nonnegativity for x_{j} = {primal[j]} is violated.')
        else:
          max_violation = max(max_violation, -primal[j])
      elif sign == -1:
        if primal[j] > PRIMAL_EPSILON:
          return (False, f'Nonpositivity for x_{j} = {primal[j]} is violated.')
        else:
          max_violation = max(max_violation, primal[j])
    for i,cons in enumerate(self.constraints):
      activity = 0.0
      for j,coef in cons['coefficients'].items():
        activity += coef * primal[j]
      if cons['relation'] == '=':
        if abs(cons['rhs'] - activity) > PRIMAL_EPSILON:
          return (False, f'Constraint #{i} evaluates to {activity} == {cons["rhs"]}.')
        else:
          max_violation = max(max_violation, abs(cons['rhs'] - activity))
      elif cons['relation'] == '<':
        if activity > cons['rhs'] + PRIMAL_EPSILON:
          return (False, f'Constraint #{i} evaluates to {activity} <= {cons["rhs"]}.')
        else:
          max_violation = max(max_violation, activity - cons['rhs'])
      elif cons['relation'] == '>':
        if activity < cons['rhs'] - PRIMAL_EPSILON:
          return (False, f'Constraint #{i} evaluates to {activity} >= {cons["rhs"]}.')
        else:
          max_violation = max(max_violation, cons['rhs'] - activity)

    return (True, max_violation)

  def dual_is_feasible(self, dual):
    '''Check feasibility of a dual solution.

    The method verifies:
    - Dual sign conditions implied by constraint relations.
    - Satisfaction of all dual constraints within a tolerance.

    Parameters
    ----------
    dual : sequence of float
        Dual variable values (one per constraint).

    Returns
    -------
    tuple
        (True, max_violation) if feasible,
        (False, error_message) if infeasible.

    Raises
    ------
    Exception
        If the dual vector has incorrect dimension.
    '''

    DUAL_EPSILON = 1.0e-6

    if len(dual) != self.num_rows:
      raise Exception(f'Given dual vector has {len(dual)} entries, but {self.num_rows} are expected.')

    max_violation = 0.0
    lhs = [ 0.0 for _ in range(self.num_columns) ]
    less_than_equal = 1.0 if self.sense == 'maximize' else -1.0
    for i,cons in enumerate(self.constraints):
      if cons['relation'] == '<':
        if less_than_equal * dual[i] < -DUAL_EPSILON: # <=: negative is forbidden for maximization
          return (False, f'Non{"negativity" if less_than_equal > 0.0 else "positivity"} for y_{i} = {dual[i]} (<=-constraint for {self.sense[:-1]}ation) is violated.')
        else:
          max_violation = max(max_violation, -less_than_equal * dual[i])
      elif cons['relation'] == '>':
        if less_than_equal * dual[i] > DUAL_EPSILON: # >=: positive is forbidden for maximization
          return (False, f'Non{"negativity" if less_than_equal < 0.0 else "positivity"} for y_{i} = {dual[i]} (>=-constraint for {self.sense[:-1]}ation) is violated.')
        else:
          max_violation = max(max_violation, less_than_equal * dual[i])

      for j,coef in cons['coefficients'].items():
        lhs[j] += coef * dual[i]

    for j,c_j in enumerate(self.objective):
      if self.signs[j] * less_than_equal < 0: # minimize and nonnegative or maximize and nonpositive -> <= c_j
        if lhs[j] > c_j + DUAL_EPSILON:
          return (False, f'Dual constraint for variable x_{j} has activity {lhs[j]}, which should be <= {c_j}.')
        else:
          max_violation = max(max_violation, lhs[j] - c_j)
      if self.signs[j] * less_than_equal > 0: # maximize and nonnegative or minimize and nonpositive -> >= c_j
        if lhs[j] < c_j - DUAL_EPSILON:
          return (False, f'Dual constraint for variable x_{j} has activity {lhs[j]}, which should be >= {c_j}.')
        else:
          max_violation = max(max_violation, c_j - lhs[j])

    return (True, max_violation)

  def ray_is_unbounded(self, ray):
    '''Check whether a given ray certifies unboundedness of the LP.

    The method verifies:
    - Variable sign constraints for the ray.
    - Satisfaction of recession cone constraints.
    - That the ray improves the objective in the correct direction.

    Parameters
    ----------
    ray : sequence of float
        Direction vector representing an unbounded ray.

    Returns
    -------
    tuple
        (True, max_violation) if the ray certifies unboundedness,
        (False, error_message) otherwise.

    Raises
    ------
    Exception
        If the ray vector has incorrect dimension.
    '''

    EPSILON = 1.0e-6

    if len(ray) != self.num_columns:
      raise Exception(f'Given ray vector has {len(ray)} entries, but {self.num_columns} are expected.')

    max_violation = 0.0
    for j,sign in enumerate(self.signs):
      if sign == 1:
        if -ray[j] > EPSILON:
          return (False, f'Nonnegativity for x_{j} = {ray[j]} is violated.')
        else:
          max_violation = max(max_violation, -ray[j])
      elif sign == -1:
        if ray[j] > EPSILON:
          return (False, f'Nonpositivity for x_{j} = {ray[j]} is violated.')
        else:
          max_violation = max(max_violation, ray[j])
    for i,cons in enumerate(self.constraints):
      activity = 0.0
      for j,coef in cons['coefficients'].items():
        activity += coef * ray[j]
      if cons['relation'] == '=':
        if abs(activity) > EPSILON:
          return (False, f'Recession cone constraint #{i} evaluates to {activity} == 0.0.')
        else:
          max_violation = max(max_violation, abs(activity))
      elif cons['relation'] == '<':
        if activity > EPSILON:
          return (False, f'Recession cone constraint #{i} evaluates to {activity} <= 0.0.')
        else:
          max_violation = max(max_violation, activity)
      elif cons['relation'] == '>':
        if activity < -EPSILON:
          return (False, f'Recession cone constraint #{i} evaluates to {activity} >= 0.0.')
        else:
          max_violation = max(max_violation, -activity)

    delta = 0.0
    for j,obj in enumerate(self.objective):
      delta += ray[j] * obj

    if self.sense == 'minimize' and delta > -EPSILON:
      return (False, f'Unbounded ray for minimization increases objetictive value by {delta}')
    elif self.sense == 'maximize' and delta < EPSILON:
      return (False, f'Unbounded ray for maximization increases objetictive value by {delta}')

    return (True, max_violation)

  def farkas_is_correct(self, farkas):
    '''Check correctness of a Farkas certificate for infeasibility.

    The method verifies:
    - Correct sign conditions on the Farkas multipliers.
    - Satisfaction of variable sign implications.
    - Strict negativity of the resulting right-hand side.

    Parameters
    ----------
    farkas : sequence of float
        Farkas multipliers (one per constraint).

    Returns
    -------
    tuple
        (True, max_violation) if the certificate is valid,
        (False, error_message) otherwise.

    Raises
    ------
    Exception
        If the Farkas vector has incorrect dimension.
    '''

    EPSILON = 1.0e-6

    if len(farkas) != self.num_rows:
      raise Exception(f'Given Farkas vector has {len(farkas)} entries, but {self.num_rows} are expected.')
    
    max_violation = 0.0
    lhs = [ 0.0 for _ in range(self.num_columns) ]
    rhs = 0.0
    for i,cons in enumerate(self.constraints):
      if cons['relation'] == '<':
        if farkas[i] < -EPSILON: # <=: negative is forbidden
          return (False, f'Nonnegativity for z_{i} = {farkas[i]} (<=-constraint) is violated.')
        else:
          max_violation = max(max_violation, -farkas[i])
      elif cons['relation'] == '>':
        if farkas[i] > EPSILON: # >=: positive is forbidden
          return (False, f'Nonpositivity for z_{i} = {farkas[i]} (>=-constraint) is violated.')
        else:
          max_violation = max(max_violation, farkas[i])
      for j,coef in cons['coefficients'].items():
        lhs[j] += coef * farkas[i]
      rhs += cons['rhs'] * farkas[i]

    for j,sign in enumerate(self.signs):
      if sign == 0:
        if abs(lhs[j]) > EPSILON:
          return (False, f'Nonzero Farkas coefficient {lhs[j]} for x_{j} in equation.')
        else:
          max_violation = max(max_violation, abs(lhs[j]))
      if sign == 1:
        if lhs[j] < -EPSILON:
          return (False, f'Nonpositive Farkas coefficient {lhs[j]} for x_{j} in <=-constraint.')
        else:
          max_violation = max(max_violation, -lhs[j])
      if sign == -1:
        if lhs[j] > EPSILON:
          return (False, f'Nonnegative Farkas coefficient {lhs[j]} for x_{j} in >=-constraint.')
        else:
          max_violation = max(max_violation, lhs[j])
    if rhs >= -EPSILON:
      return (False, f'Nonnegative or too slightly negative right-hand side {rhs}.')
    else:
      max_violation = max(max_violation, rhs - EPSILON)

    return (True, max_violation)

  def farkas_rhs(self, farkas):
    '''Compute the right-hand side value induced by a Farkas vector.

    Parameters
    ----------
    farkas : sequence of float
        Farkas multipliers.

    Returns
    -------
    float
        The weighted sum of constraint right-hand sides.
    '''

    return sum( cons['rhs'] * farkas[i] for i,cons in enumerate(self.constraints) )

  def primal_dual_are_optimal(self, primal, dual):
    '''Check optimality via strong duality.

    The method compares primal and dual objective values
    and verifies equality within a tolerance.

    Parameters
    ----------
    primal : sequence of float
        Primal variable values.
    dual : sequence of float
        Dual variable values.

    Returns
    -------
    tuple
        (True, absolute_gap) if optimal,
        (False, error_message) otherwise.
    '''

    EPSILON = 1.0e-6

    primal_objective = sum( c_j * primal[j] for j,c_j in enumerate(self.objective) )
    dual_objective = sum( cons['rhs'] * dual[i] for i,cons in enumerate(self.constraints) )
    if abs(primal_objective - dual_objective) < EPSILON:
      return (True, abs(primal_objective - dual_objective))
    else:
      return (False, f'Primal and dual objective values are {primal_objective} and {dual_objective}.')

  def is_in_standard_form(self):
    '''Check whether the LP is in standard form.

    Standard form requirements:
    - All variables are nonnegative.
    - All constraints are equalities.
    - The objective is minimization.

    Returns
    -------
    bool
        True if and only if the LP is in standard form.
    '''

    # Are all variables nonnegative?
    if self.signs != [1] * self.num_columns:
      return False

    # Are all constraints equations?
    for cons in self.constraints:
      if cons['relation'] != '=':
        return False

    # Do we minimize?
    return self.sense == 'minimize'

