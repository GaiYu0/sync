import mxnet as mx
import math
import numpy as np
import re
import logging

from lr_scheduler import MannualScheduler

class AccuracyFilter(logging.Filter):
  def __init__(self, solver):
    self._solver = solver
    self._parameters, self._auxiliary_states = None, None

  def _allocate_memory(self):
    from mxnet import cpu
    from mxnet.ndarray import zeros
    parameters = self._solver._model.arg_params
    self._parameters = {key : zeros(value.shape, cpu()) for key, value in parameters.items()}
    states = self._solver._model.aux_params
    self._auxiliary_states = {key : zeros(value.shape, cpu()) for key, value in states.items()}

  def _restore(self):
    for key, value in self._parameters.items():
      value.copyto(self._solver._model.arg_params[key])
    for key, value in self._auxiliary_states.items():
      value.copyto(self._solver._model.aux_params[key])
    
  def _store(self):
    if self._parameters is None or self._auxiliary_states is None:
      self._allocate_memory()
    for key, value in self._solver._model.arg_params.items():
      value.copyto(self._parameters[key])
    for key, value in self._solver._model.aux_params.items():
      value.copyto(self._auxiliary_states[key])

  def filter(self, record):
    message = record.getMessage()
    epoch = self._solver._progress['epoch']
    if 'Validation' in message:
      result = float(message.split('=')[-1])
      if 'cross-entropy' in message:
        self._solver._progress['validation_loss'].append(result)
        print 'epoch %d validation loss: %f' % (epoch, result)
      elif 'accuracy' in message:
        table = self._solver._progress['validation_accuracy']
        table.append(result)
        print 'epoch %d validation accuracy: %f' % (epoch, result)
        if result > max(table[:-1]):
          self._store()
          print 'epoch %d parameters stored' % epoch
      else: raise Exception('unrecognized message: %s' % message)
      return False
    else: return True

class BlockFilter(logging.Filter):
  # TODO 'Running performance tests'
  _excerpts = \
    'Auto-select kvstore type', \
    '[Deprecation Warning]', \
    'Resetting Data Iterator', \
    'Running performance tests', \
    'Start training with'

  def filter(self, record):
    message = record.getMessage()
    return all(excerpt not in message for excerpt in BlockFilter._excerpts)

class LearningRateFilter(logging.Filter):
  def filter(self, record):
    message = record.getMessage()
    if 'learning rate' in message:
      lr = float(message.split(' ')[-1])
      # TODO
      return False
    else: return True

class TimeFilter(logging.Filter):
  def __init__(self, solver):
    self._solver = solver
  def filter(self, record):
    message = record.getMessage()
    if 'Time' in message:
      epoch = self._solver._progress['epoch']
      time = float(message.split('=')[-1])
      print 'epoch %d time: %f seconds' % (epoch, time)
      return False
    else: return True

class MXSolver():
# TODO monitor
  def __init__(self,
    batch_functions    = [],
    batch_size         = None,
    data               = None,
    devices            = 'cpu',
    epochs             = None,
    epoch_functions    = [],
    symbol             = None,
    initializer        = None,
    optimizer_settings = None,
    setting_file       = None
  ):
    self._batch_size = batch_size
    self._data = data
    if devices is 'cpu': self._devices = mx.cpu()
    else: self._devices = [mx.gpu(index) for index in devices]
    self._epochs = epochs
    self._file = setting_file

    self._batch_functions = []
    if isinstance(batch_functions, list): self._batch_functions.extend(batch_functions)
    elif callable(batch_functions): self._batch_functions.append(batch_functions)

    self._epoch_functions = [
      self._training_statistics,
      self._update_settings,
      self._update_progress
    ]
    if isinstance(epoch_functions, list): self._epoch_functions.extend(epoch_functions)
    elif callable(epoch_functions): self._epoch_functions.append(epoch_functions)

    lr = optimizer_settings['initial_lr']
    self._lr_scheduler = optimizer_settings.get('lr_scheduler', MannualScheduler(lr))
    optimizer = getattr(mx.optimizer, optimizer_settings['optimizer'])(
      learning_rate = lr,
      lr_scheduler  = self._lr_scheduler,
      rescale_grad  = 1.0 / self._batch_size,
      wd            = optimizer_settings.get('weight_decay', 0),
    )

    logging.basicConfig(level=logging.NOTSET)
    logger = logging.getLogger()
    logger.addFilter(BlockFilter())

    self._model = mx.model.FeedForward(
      ctx         = self._devices,
      initializer = initializer,
      num_epoch   = self._epochs,
      optimizer   = optimizer,
      symbol      = symbol
    )

  def _training_statistics(self, *args):
    results = dict(zip(*self._metrics.get()))
    self._progress['training_loss'].append(results['cross-entropy'])
    self._progress['training_accuracy'].append(results['accuracy'])
    epoch = self._progress['epoch']
    print 'epoch %d training loss %f' % (epoch, results['cross-entropy'])
    print 'epoch %d training accuracy %f' % (epoch, results['accuracy'])

  def _load_settings(self):
    if self._file is None:
      return {}
    with open(self._file, 'r') as source:
      settings = source.read()
    settings = settings.split('\n')
    settings = dict([setting.split('=') for setting in settings])
    return settings

  def _update_progress(self, epoch, *args):
    self._progress['epoch'] = epoch
    
  def _update_settings(self, *args):
    settings = self._load_settings()
    if isinstance(self._lr_scheduler, MannualScheduler):
      pass
      # TODO

      '''
      for key, value in settings.items():
        if key == 'lr':
          lr = float(settings['lr'])
          if self.solver.scheduler.lr != lr:
            self.solver.scheduler.lr = lr
            print 'epoch {:<3} learning rate {}'.format(epoch, lr)
      '''

  # TODO data as an argument of train
  def train(self):
    self._progress = {
      'epoch'               : 0,
      'training_loss'       : [],
      'training_accuracy'   : [],
      'validation_loss'     : [],
      'validation_accuracy' : [0]
    }

    logger = logging.getLogger()
    accuracy_filter = AccuracyFilter(self)
    logger.addFilter(accuracy_filter)
    logger.addFilter(LearningRateFilter())
    logger.addFilter(TimeFilter(self))

    from mxnet.metric import Accuracy, CompositeEvalMetric, CrossEntropy
    self._metrics = CompositeEvalMetric(metrics=[Accuracy(), CrossEntropy()])

    training_X, training_Y, validation_X, validation_Y, test_X, test_Y = self._data

    self._model.fit(
      X                  = training_X,
      y                  = training_Y,
      eval_data          = (validation_X, validation_Y),
      eval_metric        = self._metrics,
      batch_end_callback = self._batch_functions,
      epoch_end_callback = self._epoch_functions,
      logger             = logger
    )

    accuracy_filter._restore()
    test_data = mx.io.NDArrayIter(test_X, test_Y, batch_size=self._batch_size)
    test_accuracy= self._model.score(test_data)
    print 'test accuracy %f' % test_accuracy

    return test_accuracy, self._progress
