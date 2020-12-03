# Copyright 2020 Mathieu Blondel.
# BSD license.

# Slides:
# http://mblondel.org/teaching/beyond-gd-2020.pdf

# The code is optimized for readability, not speed.

import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer


def _primal_dual_link_squared_l2(X, Y, lam, beta):
  return np.dot(X.T, (Y - beta)) / lam


def _duality_gap(X, Y, W, beta, lam):
  # Compute the duality gap.
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

  return primal_obj - dual_obj


def _gradient(X, Y, W):
  n_samples = X.shape[0]
  n_classes = W.shape[1]

  # Same as below but using a for loop.
  #G = np.zeros((n_samples, n_classes))
  #for i in range(n_samples):
    #G[i] -= (Y[i] - 1)
    #G[i] += np.dot(W.T, X[i])

  G = -(Y - 1).astype(float)
  G += np.dot(X, W)

  # We want to maximize, hence the sign.
  return -G


def _lmo(G):
  n_samples = G.shape[0]
  S = np.zeros_like(G)

  # Same as below but using a for loop.
  #for i in range(n_samples):
    #j = np.argmin(G[i])
    #S[i, j] = 1

  ind = np.arange(n_samples)
  S[ind, np.argmin(G, axis=1)] = 1

  return S


def frank_wolfe_multiclass_svm(X, Y, lam, n_epochs=100):
  n_samples, n_classes = Y.shape

  # Initialization.
  beta = np.ones((n_samples, n_classes)) / n_classes
  W = _primal_dual_link_squared_l2(X, Y, lam, beta)  # n_features x n_classes

  # Loop over epochs.
  for it in range(n_epochs):

    G = _gradient(X, Y, W)
    S = _lmo(G)

    gamma = 2 / (2 + it)
    beta = (1 - gamma) * beta + gamma * S
    W = _primal_dual_link_squared_l2(X, Y, lam, beta)
    print(_duality_gap(X, Y, W, beta, lam))


def mirror_descent_multiclass_svm(X, Y, lam, n_epochs=100):
  n_samples, n_classes = Y.shape

  # Initialization.
  beta = np.ones((n_samples, n_classes)) / n_classes
  W = _primal_dual_link_squared_l2(X, Y, lam, beta)  # n_features x n_classes

  eta = 1e-2

  # Loop over epochs.
  for it in range(1, n_epochs + 1):

    G = _gradient(X, Y, W)
    eta_t = eta / np.sqrt(it)

    for i in range(n_samples):
      tmp = beta[i] * np.exp(-eta_t * G[i])
      beta[i] = tmp / np.sum(tmp)

    W = _primal_dual_link_squared_l2(X, Y, lam, beta)
    print(_duality_gap(X, Y, W, beta, lam))


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

  frank_wolfe_multiclass_svm(X, Y, lam=1e3, n_epochs=30)
  print()

  mirror_descent_multiclass_svm(X, Y, lam=1e3, n_epochs=30)
  print()


if __name__ == '__main__':
  main()
