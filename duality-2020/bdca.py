# Copyright 2020 Mathieu Blondel.
# BSD license.

# Slides:
# http://mblondel.org/teaching/autodiff-2020.pdf

# The code is optimized for readability, not speed.

import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer


def primal_dual_link_squared_l2(X, Y, lam, beta):
  """
  Computes primal-dual link in the squared L2 regularization case.

  Args:
    X: feature matrix (n_samples x n_features)
    Y: label matrix (n_samples x n_classes)
    lam: lambda value (regularization strength)
    beta: dual variables (n_samples x n_classes)

  Returns:
    W = nabla G^*(X^T (Y - beta))
  """
  return np.dot(X.T, (Y - beta)) / lam


def prox_squared_loss(eta, tau):
  """
  Prox operator associated with the squared loss.

  Args:
    eta: input
    tau: multiplication factor
  """
  return eta / (tau + 1)


def bdca_squared_loss(X, Y, lam, n_epochs=100):
  """
  BDCA for the squared loss with squared L2 regularization (ridge regression).

  Args:
    X: feature matrix (n_samples x n_features)
    Y: label matrix (n_samples x n_classes)
    lam: lambda value (regularization)
    n_epochs: number of iterations to perform

  Returns:
    W
  """
  n_samples, n_classes = Y.shape

  # Initialization.
  beta = np.ones((n_samples, n_classes)) / n_classes
  W = primal_dual_link_squared_l2(X, Y, lam, beta)  # n_features x n_classes

  # Pre-compute squared norms.
  sqnorms = np.sum(X ** 2, axis=1)

  # Loop over epochs.
  for it in range(n_epochs):

    # Loop over samples / blocks.
    for i in range(len(X)):
      if sqnorms[i] == 0:
        continue

      sigma_i = sqnorms[i] / lam
      u_i = np.dot(W.T, X[i]) + sigma_i * beta[i]
      beta[i] = prox_squared_loss(u_i / sigma_i, 1./ sigma_i)

      # We recompute W from scratch for simplicity.
      # Since only beta[i] changed, we can update W more efficiently.
      W = primal_dual_link_squared_l2(X, Y, lam, beta)

    # Compute the duality gap once per epoch.
    Y_pred = np.dot(X, W)

    primal_obj = 0.5 * np.sum((Y - Y_pred) ** 2)  # loss term
    primal_obj += 0.5 * lam * np.vdot(W, W) # regul term

    dual_obj = -0.5 * np.sum(beta ** 2) + 0.5 * np.sum(Y ** 2)  # loss term
    V = np.dot(X.T, (Y - beta))
    # Equivalent to 0.5 * lam * np.vdot(W, W)
    dual_obj -= 0.5 / lam * np.vdot(V, V)  # regul term

    gap = primal_obj - dual_obj
    print(gap)

  return W


def projection_simplex(v, z=1):
  """
  Project v onto the probability simplex.
  """
  n_features = v.shape[0]
  u = np.sort(v)[::-1]
  cssv = np.cumsum(u) - z
  ind = np.arange(n_features) + 1
  cond = u - cssv / ind > 0
  rho = ind[cond][-1]
  theta = cssv[cond][-1] / float(rho)
  w = np.maximum(v - theta, 0)
  return w


def prox_multiclass_hinge_loss(eta, tau, v):
  """
  Prox operator associated with the multiclass hinge loss.

  Args:
    eta: input
    tau: multiplication factor
    v: input (see slides for definition)
  """
  return projection_simplex(eta + tau * v)


def bdca_multiclass_hinge_loss(X, Y, lam, n_epochs=100):
  """
  BDCA for the multiclass hinge loss with squared L2 regularization.

  Args:
    X: feature matrix (n_samples x n_features)
    Y: label matrix (n_samples x n_classes)
    lam: lambda value (regularization)
    n_epochs: number of iterations to perform

  Returns:
    W
  """
  n_samples, n_classes = Y.shape

  # Initialization.
  beta = np.ones((n_samples, n_classes)) / n_classes
  W = primal_dual_link_squared_l2(X, Y, lam, beta)  # n_features x n_classes

  # Pre-compute squared norms.
  sqnorms = np.sum(X ** 2, axis=1)

  # Loop over epochs.
  for it in range(n_epochs):

    # Loop over samples / blocks.
    for i in range(len(X)):
      if sqnorms[i] == 0:
        continue

      sigma_i = sqnorms[i] / lam
      u_i = np.dot(W.T, X[i]) + sigma_i * beta[i]
      v_i = 1 - Y[i]
      beta[i] = prox_multiclass_hinge_loss(u_i / sigma_i, 1./ sigma_i, v_i)

      # We recompute W from scratch for simplicity.
      # Since only beta[i] changed, we can update W more efficiently.
      W = primal_dual_link_squared_l2(X, Y, lam, beta)

    # Compute the duality gap once per epoch.
    Y_pred = np.dot(X, W)

    primal_obj = 0
    dual_obj = 0
    for i in range(len(X)):
      v_i = 1 - Y[i]
      # loss terms
      primal_obj += np.max(Y_pred[i] + v_i) - np.dot(Y_pred[i], Y[i])
      dual_obj -= (-np.dot(beta[i], v_i) + np.dot(Y[i], v_i))

    # regul terms
    primal_obj += 0.5 * lam * np.vdot(W, W)
    V = np.dot(X.T, (Y - beta))
    # Equivalent to 0.5 * lam * np.vdot(W, W)
    dual_obj -= 0.5 / lam * np.vdot(V, V)

    gap = primal_obj - dual_obj
    print(gap)

  return W


def main():
  X, y = load_digits(return_X_y=True)

  # Transform labels to a one-hot representation.
  # Y has shape (n_samples, n_classes).
  Y = LabelBinarizer().fit_transform(y)

  # Shuffle the samples.
  rng = np.random.RandomState(0)
  perm = rng.permutation(len(X))
  X = X[perm]
  Y = Y[perm]

  #bdca_squared_loss(X, Y, lam=1e3, n_epochs=100)
  bdca_multiclass_hinge_loss(X, Y, lam=1e3, n_epochs=100)


if __name__ == '__main__':
  main()
