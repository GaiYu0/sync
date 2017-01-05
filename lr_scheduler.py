import logging

class Scheduler(object):
  def __init__(self, lr, n, batch_size):
    self.lr = lr
    self._n_batches = n // batch_size
  def __call__(self, iteration):
    raise NotImplementedError()
  def _current_epoch(self, iteration):
    return iteration % self._n_batches

class MannualScheduler:
  def __init__(self, lr):
    self.lr = lr
  def __call__(self, *args):
    return self.lr

class FactorScheduler(object):
  def __init__(self, lr, factor, n, batch_size, minimum_lr=0):
    super(FactorScheduler, self).__init__(lr, n, batch_size)
    self._factor = factor
    self._minimum_lr = minimum_lr
    self._previous_iteration = None
  def __call__(self, iteration):
    if iteration != self._previous_iteration:
      if (iteration + 1) % self._n_batches == 0 and self.lr * self._factor > self._minimum_lr:
        self.lr *= self._factor
        logging.info('iteration %d learning rate set to %f' % (iteration, self.lr))
      self._previous_iteration = iteration
    return self.lr

class AtEpochScheduler(object):
  def __init__(self, initial_lr, lr_table, n, batch_size):
    super(AtEpochScheduler, self).__init__(initial_lr, n, batch_size)
    self._lr_table = lr_table
  def __call__(self, iteration):
    self.lr = self._lr_table.get(self._current_epoch(), self.lr)
    return self.lr
