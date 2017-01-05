import logging

class MannualScheduler:
  def __init__(self, lr):
    self.lr = lr
  def __call__(self, *args):
    return self.lr

class FactorScheduler:
  def __init__(self, lr, factor, interval, minimum_lr=0):
    self.lr = lr
    self._minimum_lr = minimum_lr
    self._factor = factor
    self._interval = interval
    self._previous_iteration = None
  def __call__(self, iteration):
    if iteration != self._previous_iteration:
      if iteration != 0 and iteration % self._interval == 0 and self.lr * self._factor > self._minimum_lr:
        self.lr *= self._factor
        logging.info('iteration %d learning rate set to %f' % (iteration, self.lr))
      self._previous_iteration = iteration
    return self.lr

class AtEpochScheduler:
  def __init__(self, lr_table):
    self._lr_table = lr_table
    # epoch starts from 0
    self.lr = lr
    self._factor = factor
    self._epochs_to_decay = epochs_to_decay + [float('inf')]
    self._epoch_index = 0
    self._batches = batches
    self._previous_iteration = None
  def __call__(self, iteration):
    if iteration != self._previous_iteration:
      if iteration / self._batches == self._epochs_to_decay[self._epoch_index]:
        self.lr *= self._factor
        self._epoch_index += 1
      self._previous_iteration = iteration
    return self.lr
