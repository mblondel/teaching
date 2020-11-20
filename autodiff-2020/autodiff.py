# Copyright 2020 Mathieu Blondel.
# BSD license.

import numpy as np

# Basic functions and their VJPs.

def dot(x, W):
  return np.dot(W, x)


def dot_make_vjp(x, W):
  def vjp(u):
    return W.T.dot(u), np.outer(u, x)
  return vjp


dot.make_vjp = dot_make_vjp


def relu(x):
  return np.maximum(x, 0)


def relu_make_vjp(x):
  gprime = np.zeros(len(x))
  gprime[x >= 0] = 1

  def vjp(u):
    return u * gprime,  # The comma is important.

  return vjp


relu.make_vjp = relu_make_vjp


def squared_loss(y_pred, y):
  # The code requires every output to be an array.
  return np.array([0.5 * np.sum((y - y_pred) ** 2)])


def squared_loss_make_vjp(y_pred, y):
  diff = y_pred - y

  def vjp(u):
    return diff * u, -diff * u

  return vjp


squared_loss.make_vjp = squared_loss_make_vjp


def add(a, b):
  return a + b


def add_make_vjp(a, b):
  gprime = np.ones(len(a))

  def vjp(u):
    return u * gprime, u * gprime

  return vjp


add.make_vjp = add_make_vjp


def mul(a, b):
  return a * b


def mul_make_vjp(a, b):
  gprime_a = b
  gprime_b = a

  def vjp(u):
    return u * gprime_a, u * gprime_b

  return vjp


mul.make_vjp = mul_make_vjp


def exp(x):
  return np.exp(x)


def exp_make_vjp(x):
  gprime = exp(x)

  def vjp(u):
    return u * gprime,

  return vjp


exp.make_vjp = exp_make_vjp


def sqrt(x):
  return np.sqrt(x)


def sqrt_make_vjp(x):
  gprime = 1. / (2 * sqrt(x))

  def vjp(u):
    return u * gprime,

  return vjp


sqrt.make_vjp = sqrt_make_vjp


# Feedforward.

def call_func(x, func, param):
  """Make sure the function is called with the correct number of arguments."""

  if param is None:
    # Unary function
    return func(x)
  else:
    # Binary function
    return func(x, param)


def evaluate_chain(x, funcs, params, return_all=False):
  """
  Evaluate a chain of functions.

  Args:
    x: initial input to the chain.
    funcs: a list of functions of the form func(x) or func(x, param).
    params: a list of parameters, with len(params) = len(funcs).
            If a function doesn't have parameters, use None.

  Returns:
    value
  """
  if len(funcs) != len(params):
    raise ValueError("len(funcs) and len(params) should be equal.")

  xs = [x]

  for k in range(len(funcs)):
    xs.append(call_func(xs[k], funcs[k], params[k]))

  if return_all:
    return xs
  else:
    return xs[-1]


def forward_diff_chain(x, funcs, params):
  """
  Forward-mode differentiation.

  Args:
    x: initial input to the chain.
    funcs: a list of functions of the form func(x) or func(x, param).
    params: a list of parameters, with len(params) = len(funcs).
            If a function doesn't have parameters, use None.

  Returns:
    value, Jacobian w.r.t. x.
  """
  if len(funcs) != len(params):
    raise ValueError("len(funcs) and len(params) should be equal.")

  n = len(x)  # Input size
  eps = 1e-6  # Precision

  # We need a list as the shape of V can change.
  V = list(np.eye(n))

  for k in range(len(funcs)):
    func = lambda x: call_func(x, funcs[k], params[k])

    for j in range(n):
      # We compute JVPs by finite difference for convenience as they require
      # only two calls to the function.
      V[j] = (func(x + eps * V[j]) - func(x - eps * V[j])) / (2 * eps)

    x = func(x)

  return x, np.array(V).T


def call_vjp(x, func, param, u):
  """Make sure the vjp is called with the correct number of arguments."""
  if param is None:
    vjp = func.make_vjp(x)
    vjp_x, = vjp(u)
    vjp_param = None
  else:
    vjp = func.make_vjp(x, param)
    vjp_x, vjp_param = vjp(u)
  return vjp_x, vjp_param


def backward_diff_chain(x, funcs, params):
  """
  Reverse-mode differentiation of a chain of computations.

  Args:
    x: initial input to the chain.
    funcs: a list of functions of the form func(x) or func(x, param).
    params: a list of parameters, with len(params) = len(funcs).
            If a function doesn't have parameters, use None.

  Returns:
    value, Jacobian w.r.t. x, Jacobians w.r.t. params.
  """
  # Evaluate the feedforward model and store intermediate computations,
  # as they will be needed during the backward pass.
  xs = evaluate_chain(x, funcs, params, return_all=True)

  m = xs[-1].shape[0]  # Output size
  K = len(funcs)  # Number of functions.

  # We need a list as the shape of U can change.
  U = list(np.eye(m))

  # List that will contain the Jacobian of each function w.r.t. parameters.
  J = [None] * K

  for k in reversed(range(K)):
    jac = []

    for i in range(m):
      vjp_x, vjp_param = call_vjp(xs[k], funcs[k], params[k], U[i])
      jac.append(vjp_param)
      U[i] = vjp_x

    J[k] = np.array(jac)

  return xs[-1], np.array(U), J


# Direct acyclic graphs.

class Node(object):

  def __init__(self, value=None, func=None, parents=None, name=""):
    # Value stored in the node.
    self.value = value
    # Function producing the node.
    self.func = func
    # Inputs to the function.
    self.parents = [] if parents is None else parents
    # Unique name of the node (for debugging and hashing).
    self.name = name
    # Gradient / Jacobian.
    self.grad = 0
    if not name:
      raise ValueError("Each node must have a unique name.")

  def __hash__(self):
    return hash(self.name)

  def __repr__(self):
    return "Node(%s)" % self.name


def dfs(node, visited):
  visited.add(node)
  for parent in node.parents:
    if not parent in visited:
      # Yield parent nodes first.
      yield from dfs(parent, visited)
  # And current node later.
  yield node


def topological_sort(end_node):
  """
  Topological sorting.

  Args:
    end_node: in.

  Returns:
    sorted_nodes
  """
  visited = set()
  sorted_nodes = []

  # All non-visited nodes reachable from end_node.
  for node in dfs(end_node, visited):
    sorted_nodes.append(node)

  return sorted_nodes


def evaluate_dag(sorted_nodes):
  for node in sorted_nodes:
    if node.value is None:
      values = [p.value for p in node.parents]
      node.value = node.func(*values)
  return sorted_nodes[-1].value


def backward_diff_dag(sorted_nodes):
  value = evaluate_dag(sorted_nodes)
  m = value.shape[0]  # Output size

  # Initialize recursion.
  sorted_nodes[-1].grad = np.eye(m)

  for node_k in reversed(sorted_nodes):
    if not node_k.parents:
      # We reached a node without parents.
      continue

    # Values of the parent nodes.
    values = [p.value for p in node_k.parents]

    # Iterate over outputs.
    for i in range(m):
      # A list of size len(values) containing the vjps.
      vjps = node_k.func.make_vjp(*values)(node_k.grad[i])

      for node_j, vjp in zip(node_k.parents, vjps):
        node_j.grad += vjp

  return sorted_nodes
