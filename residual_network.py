import mxnet as mx
from MXLayers import *
from hyper_network import generated_convolution_weight

def _normalized_convolution(X, kernel_shape, n_kernels, stride, pad, weight=None, bias=None):
  network = convolution(X, kernel_shape, n_kernels, stride, pad, weight=weight, bias=bias)
  network = batch_normalization(network)
  network = ReLU(network)
  return network

def _transit(X, n_kernels):
  P = _normalized_convolution(X, (3, 3), n_kernels, (2, 2), (1, 1))
  P = _normalized_convolution(P, (3, 3), n_kernels, (1, 1), (1, 1))
  Q = pooling(X, 'average', (2, 2), (2, 2))
  Q = convolution(Q, (1, 1), n_kernels)
  return P + Q

def _recur(X, n_kernels, weight=None, bias=None):
  residual = _normalized_convolution(X, (3, 3), n_kernels, (1, 1), (1, 1), weight=weight, bias=bias)
  residual = _normalized_convolution(residual, (3, 3), n_kernels, (1, 1), (1, 1), weight=weight, bias=bias)
  return X + residual

def triple_state_residual_network(n, mode):
  N_z = 64
  d = N_z
  WIDTH, HEIGHT = 3, 3

  network = variable('data')

  FILTER_IN, FILTER_OUT = 16, 16
  network = _normalized_convolution(network, (3, 3), FILTER_IN, (1, 1), (1, 1))
  if mode == 'normal': weight = None
  else: weight = generated_convolution_weight(N_z, d, FILTER_IN, FILTER_OUT, WIDTH, HEIGHT)
  for i in range(n): network = _recur(network, FILTER_OUT, weight=weight)

  FILTER_IN, FILTER_OUT = 32, 32
  network = _transit(network, FILTER_IN)
  if mode == 'normal': weight = None
  else: weight = generated_convolution_weight(N_z, d, FILTER_IN, FILTER_OUT, WIDTH, HEIGHT)
  for i in range(n - 1): network = _recur(network, FILTER_OUT, weight=weight)

  FILTER_IN, FILTER_OUT = 64, 64
  network = _transit(network, 64)
  if mode == 'normal': weight = None
  else: weight = generated_convolution_weight(N_z, d, FILTER_IN, FILTER_OUT, WIDTH, HEIGHT)
  for i in range(n - 1): network = _recur(network, 64, weight=weight)

  network = pooling(network, (8, 8), (1, 1))
  network = flatten(network)
  network = fully_connected(network, 10)
  network = softmax(network)
  return network
