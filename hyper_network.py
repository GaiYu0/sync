import mxnet as mx
from mx_layers import *

def generated_convolution_weight(*args):
  # implementing the method proposed in "HyperNetworks"
  return _generated_convolution_weight._(*args)

class _generated_convolution_weight:
  _n = 0
  @staticmethod
  def _(N_z, d, filter_in, filter_out, width, height):
    # TODO layer name
    n = generated_convolution_weight._n
    weight = variable('convolution_embedding%d_weight' % n)
    generated_convolution_weight._n += 1
    weight = fully_connected(weight, filter_in * d)
    weight = reshape(weight, (filter_in, d))
    weight = fully_connected(weight, filter_out * width * height)
    weight = reshape(weight, (filter_in, filter_out, width, height))
    weight = swap_axes(weight, 0, 1)
    return weight
