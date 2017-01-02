import mxnet as mx
from MXLayers import *

class ResidualNetworkScheduler:
  def __init__(self):
    self.lr = 0.1
  def __call__(self, iteration):
    if iteration == 32000:
      self.lr *= 0.1
    if iteration == 48000:
      self.lr *= 0.1
    return self.lr

class ResidualNetwork():
  def __init__(self, n, activation, initializer, double_kernel=True):
    self.n, self.activation, self.initializer = n, activation, initializer
    self.double_kernel = double_kernel

  def __call__(self, data_shape, intermediate_results=False):
    if intermediate_results:
      results = []

    def normalized_convolution(inputs, kernel_shape, kernel_number, stride, pad, activation=None):
      outputs = convolution(inputs, kernel_shape, kernel_number, stride, pad)
      if intermediate_results == 'pre-activation':
        results.append(outputs)
      if activation:
        outputs = activate(outputs, activation, data_shape)
      if intermediate_results == 'post-activation':
        results.append(outputs)
      return outputs

    def residual(inputs, kernel_number, activation=None, project=False):
      if project:
        left = normalized_convolution(inputs, (3, 3), kernel_number, (2, 2), (1, 1), activation)
        left = normalized_convolution(left, (3, 3), kernel_number, (1, 1), (1, 1), activation)
        right = pooling(inputs, 'avg', (2, 2), (2, 2))
        right = convolution(right, (1, 1), kernel_number)
        outputs = left + right

      else:
        transformed = normalized_convolution(inputs, (3, 3), kernel_number, (1, 1), (1, 1), activation)
        if self.double_kernel:
          transformed = normalized_convolution(transformed, (3, 3), kernel_number, (1, 1), (1, 1), activation)
        outputs = inputs + transformed

      return outputs
     
    n = self.n
    activation = self.activation

    inputs = variable('data')
    outputs = normalized_convolution(inputs, (3, 3), 16, (1, 1), (1, 1), activation)

    for i in range(n):
      outputs = residual(outputs, 16, activation)

    outputs = residual(outputs, 32, activation, project=True)

    for i in range(n-1):
      outputs = residual(outputs, 32, activation)
    outputs = residual(outputs, 64, activation, project=True)

    for i in range(n-1):
      outputs = residual(outputs, 64, activation)

    outputs = pooling(outputs, 'avg', (8, 8))
    outputs = flatten(outputs)
    outputs = fully_connected(outputs, 10)

    if intermediate_results:
      return mx.symbol.Group(results)
    else:
      return softmax(outputs), self.initializer
