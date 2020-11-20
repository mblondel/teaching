import numpy as np

from nose.tools import assert_equal, assert_almost_equal
from numpy.testing import assert_array_almost_equal, assert_array_equal

from autodiff import dot, relu, mul, add, exp, sqrt, squared_loss
from autodiff import evaluate_chain, forward_diff_chain, backward_diff_chain
from autodiff import Node, evaluate_dag, backward_diff_dag, topological_sort

# Finite difference utilities.

def num_jvp(f, x, v, eps=1e-6):
  """
  Args:
    f: a function returning an array.
    x: an array.
    v: an array (same shape as x).

  Returns:
    numerical_jvp
  """
  if not np.array_equal(x.shape, v.shape):
    raise ValueError("x and v should have the same shape.")

  return (f(x + eps * v) - f(x - eps * v)) / (2 * eps)


def num_jacobian(f, x, eps=1e-6):
  """
  Args:
    f: a function returning an array.
    x: an array (only 1d and 2d arrays supported).

  Returns:
    numerical_jacobian
  """
  def e(i):
    ret = np.zeros_like(x)
    ret[i] = 1
    return ret

  def E(i, j):
    ret = np.zeros_like(x)
    ret[i, j] = 1
    return ret

  if len(x.shape) == 1:
    return np.array([num_jvp(f, x, e(i), eps=eps) for i in range(len(x))]).T
  elif len(x.shape) == 2:
    return np.array([[num_jvp(f, x, E(i, j), eps=eps) \
                     for i in range(x.shape[0])] \
                     for j in range(x.shape[1])]).T
  else:
    raise NotImplementedError


def num_vjp(f, x, u, eps=1e-6):
  """
  Args:
    f: a function returning an array.
    x: an array (only 1d and 2d arrays supported).

  Returns:
    numerical_vjp
  """
  J = num_jacobian(f, x, eps=eps)
  if len(J.shape) == 2:
    return J.T.dot(u)
  elif len(J.shape) == 3:
    shape = J.shape[1:]
    J = J.reshape(J.shape[0], -1)
    return u.dot(J).reshape(shape)
  else:
    raise NotImplementedError


def test_num_jacobian():
  a, b = 3, 2
  rng = np.random.RandomState(0)
  W = rng.randn(a, b)
  x = rng.randn(b)
  v = rng.randn(b)
  u = rng.randn(a)

  # A function from R^2 to R^3
  f = lambda x: np.dot(W, x)

  vjp_f = lambda u: num_vjp(f, x, u)

  jvp = num_jvp(f, x, v)
  assert_equal(jvp.shape, (a,))
  assert_array_almost_equal(num_vjp(vjp_f, u, v), jvp)

  J = num_jacobian(f, x)
  assert_equal(J.shape, (a, b))

  vjp = num_vjp(f, x, u)
  assert_equal(vjp.shape, (b,))

  # A function from R^{3, 2} to R^3
  g = lambda W: np.dot(W, x)

  J = num_jacobian(g, W)
  assert_equal(J.shape, (a, a, b))

  V = rng.randn(*W.shape)
  jvp = num_jvp(g, W, V)
  assert_equal(jvp.shape, (a, ))

  vjp = num_vjp(g, W, u)
  assert_equal(vjp.shape, (a, b))


# Actual tests begin here.

def test_dot_vjp():
  a, b = 3, 2
  rng = np.random.RandomState(0)
  W = rng.randn(a, b)
  x = rng.randn(b)
  v = rng.randn(b)
  u = rng.randn(a)
  V = rng.randn(a, b)

  f = lambda x: dot(x, W)
  g = lambda W: dot(x, W)

  vjp_x_num = num_vjp(f, x, u)
  vjp_W_num = num_vjp(g, W, u)
  vjp_x, vjp_W = dot.make_vjp(x, W)(u)
  assert_array_almost_equal(vjp_x, vjp_x_num)
  assert_array_almost_equal(vjp_W_num, vjp_W)


def test_unary_vjp():
  b = 5
  rng = np.random.RandomState(0)
  x = rng.rand(b)
  u = rng.rand(b)

  for func in (exp, relu, sqrt):
    vjp_x_num = num_vjp(func, x, u)
    vjp_x, = func.make_vjp(x)(u)
    assert_array_almost_equal(vjp_x_num, vjp_x)


def test_binary_vjp():
  b = 5
  rng = np.random.RandomState(0)
  x = rng.rand(b)
  y = rng.rand(b)
  u = rng.rand(b)

  for func in (add, mul):
    f = lambda x: func(x, y)
    vjp_x_num = num_vjp(f, x, u)

    f = lambda y: func(x, y)
    vjp_y_num = num_vjp(f, y, u)

    vjp_x, vjp_y = func.make_vjp(x, y)(u)
    assert_array_almost_equal(vjp_x_num, vjp_x)
    assert_array_almost_equal(vjp_y_num, vjp_y)


def test_squared_loss_vjp():
  rng = np.random.RandomState(0)
  y_pred = rng.randn(2)
  y = rng.randn(*y_pred.shape)
  u = rng.randn(1)

  f = lambda y_pred: squared_loss(y_pred, y)
  vjp_y_pred_num = num_vjp(f, y_pred, u)

  g = lambda y: squared_loss(y_pred, y)
  vjp_y_num = num_vjp(g, y, u)

  vjp_y_pred, vjp_y = squared_loss.make_vjp(y_pred, y)(u)

  assert_array_almost_equal(vjp_y_pred_num, -vjp_y_num)
  assert_array_almost_equal(vjp_y_pred_num, vjp_y_pred)
  assert_array_almost_equal(vjp_y_num, vjp_y)


def create_feed_forward(n, y, seed=None):
  rng = np.random.RandomState(seed)

  funcs = [
    dot,
    relu,
    dot,
    relu,
    dot,
    squared_loss
  ]

  params = [
    rng.randn(3, n),
    None,
    rng.randn(4, 3),
    None,
    rng.randn(1, 4),
    y
  ]

  return funcs, params


def test_feed_forward():
  rng = np.random.RandomState(0)
  x = rng.randn(2)
  y = 1.5

  funcs, params = create_feed_forward(n=len(x), y=y, seed=0)
  W1, _, W3, _, W5, y = params

  # Test evaluate_chain.
  value = evaluate_chain(x, funcs, params)
  value2 = 0.5 * (y - W5.dot(relu(W3.dot(relu(W1.dot(x)))))) ** 2
  assert_array_almost_equal(value, value2)

  # Test derivatives wrt x.
  f = lambda x: evaluate_chain(x, funcs, params)
  num_jac = num_jacobian(f, x)

  value, jac = forward_diff_chain(x, funcs, params)
  assert_array_almost_equal(jac, num_jac)

  value2, jac2, param_jacs = backward_diff_chain(x, funcs, params)

  assert_almost_equal(value, value2)
  assert_array_almost_equal(jac, jac2)

  # Test derivatives wrt W5
  def f(W5):
    params = [W1, None, W3, None, W5, y]
    return evaluate_chain(x, funcs, params)

  W5 = params[4]
  num_jac = num_jacobian(f, W5)
  assert_array_almost_equal(num_jac, param_jacs[4])

  # Test derivatives wrt W3
  def f(W3):
    params = [W1, None, W3, None, W5, y]
    return evaluate_chain(x, funcs, params)

  W3 = params[2]
  num_jac = num_jacobian(f, W3)
  assert_array_almost_equal(num_jac, param_jacs[2])

  # Test derivatives wrt W1
  def f(W1):
    params = [W1, None, W3, None, W5, y]
    return evaluate_chain(x, funcs, params)

  W1 = params[0]
  num_jac = num_jacobian(f, W1)
  assert_array_almost_equal(num_jac, param_jacs[0])


def test_diff_opt():
  from scipy.linalg import solve

  rng = np.random.RandomState(0)
  n_samples, n_features = 5, 3
  A = rng.randn(n_samples, n_features)
  b = rng.randn(5)
  theta = 1.5

  AA = np.dot(A.T, A)
  I = np.eye(A.shape[1])
  Ab = np.dot(A.T, b)
  x_star = solve(AA + theta * I, Ab)
  grad = AA.dot(x_star) - Ab + theta * x_star
  assert_array_almost_equal(grad, np.zeros_like(grad))

  def grad_x(x):
    return AA.dot(x) - Ab + theta * x

  H = AA + theta * I
  assert_array_almost_equal(num_jacobian(grad_x, x_star), H)

  def grad_theta(theta):
    return AA.dot(x_star) - Ab + theta[0] * x_star

  theta_vec = np.array([theta])
  J2 = x_star.reshape(-1, 1)
  assert_array_almost_equal(num_jacobian(grad_theta, theta_vec), J2)

  def f(theta):
    return solve(AA + theta[0] * I, Ab)

  assert_array_almost_equal(num_jacobian(f, theta_vec), solve(H, -J2))


def create_dag(x):
  if len(x) != 2:
    raise ValueError

  x1 = Node(value=np.array([x[0]]), name="x1")
  x2 = Node(value=np.array([x[1]]), name="x2")
  x3 = Node(func=exp, parents=[x1], name="x3")
  x4 = Node(func=mul, parents=[x2, x3], name="x4")
  x5 = Node(func=add, parents=[x1, x4], name="x5")
  x6 = Node(func=sqrt, parents=[x5], name="x6")
  x7 = Node(func=mul, parents=[x4, x6], name="x7")
  return x7


def test_dag():
  x7 = create_dag([0.5, 1.3])
  sorted_nodes = topological_sort(x7)
  node_names = [node.name for node in sorted_nodes]
  names = ["x2", "x1", "x3", "x4", "x5", "x6", "x7"]
  assert_array_equal(node_names, names)

  value = evaluate_dag(sorted_nodes)

  def f(x):
    return x[1] * np.exp(x[0]) * np.sqrt(x[0] + x[1] * np.exp(x[0]))

  x = np.array([0.5, 1.3])
  value2 = f(x)
  assert_array_almost_equal(value, value2)

  num_jac = num_jacobian(f, x)
  backward_diff_dag(sorted_nodes)
  # x2 is before x1 in the topological order
  jac = np.concatenate([sorted_nodes[1].grad, sorted_nodes[0].grad])
  assert_array_almost_equal(num_jac, jac)
