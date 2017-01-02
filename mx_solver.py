import mxnet as mx
import math
import numpy as np
import re
import logging

from LRScheduler import MannualScheduler

class AccuracyFilter(logging.Filter):
  def __init__(self, solver, validation_data, validation_labels, validation_batch_size, progress):
    self.model = solver.model
    self.progress = progress
    self.memory_allocated = False
    self.arg_params, self.aux_params = None, None
    self.validation_iterator = mx.io.NDArrayIter(
      validation_data,
      validation_labels,
      validation_batch_size
    )
    self.data_shape = (validation_batch_size,) + validation_data.shape[1:]
    self.verbose = solver.verbose
  def filter(self, record):
    message = record.getMessage()
    if 'Validation' in message:
      result = float(message.split('=')[-1])
      if 'cross-entropy' in message:
        validation_loss = self.progress['validation_loss']
        validation_loss.append(result)
        if self.verbose:
          print 'epoch {:<3} validation loss\t{}'.format(
            self.progress['epoch'],
            validation_loss[-1]
          )
      elif 'accuracy' in message:
        validation_accuracy = self.progress['validation_accuracy']
        validation_accuracy.append(result)
        print 'epoch {:<3} validation accuracy\t{}'.format(
          self.progress['epoch'],
          validation_accuracy[-1]
        )
        if validation_accuracy[-1] > max(validation_accuracy[:-1]):
          if not self.memory_allocated:
            self.arg_params = {
              key : mx.nd.zeros(value.shape, mx.cpu()) for key, value in self.model.arg_params.items()
            }
            self.aux_params = {
              key : mx.nd.zeros(value.shape, mx.cpu()) for key, value in self.model.aux_params.items()
            }
            self.memory_allocated = True
          for key, value in self.model.arg_params.items():
            value.copyto(self.arg_params[key])
          for key, value in self.model.aux_params.items():
            value.copyto(self.aux_params[key])
          print 'epoch {:<3} checkpointed'.format(self.progress['epoch'])
      else:
        print 'unrecognized message: %s' % message
        raise Exception()
      return False
    else:
      return True

class BlockFilter(logging.Filter):
  def __init__(self):
    self.targets = (
      'Auto-select kvstore type',
      'Start training with',
      'Resetting Data Iterator'
    )
  def filter(self, record):
    message = record.getMessage()
    return all(target not in message for target in self.targets)

class LRDecayFilter(logging.Filter):
  def __init__(self, settings):
    self.optimizer_settings = settings
  def filter(self, record):
    message = record.getMessage()
    if 'Change learning rate to' in message:
      lr = float(message.split(' ')[-1])
      if lr != self.optimizer_settings['lr']:
        print 'learning rate decayed to %f' % lr
      return False
    else:
      return True

class TimeFilter(logging.Filter):
  def __init__(self, progress):
    self.progress = progress
  def filter(self, record):
    message = record.getMessage()
    if 'Time cost' in message:
      time_consumed = float(message.split('=')[-1])
      print 'epoch {:<3} {} seconds consumed'.format(self.progress['epoch'], time_consumed)
      return False
    else:
      return True

class EpochCallback:
  def __init__(self, solver, metric, callback, progress):
    self.solver = solver
    self.metric = metric
    self.callbacks = callback
    self.progress = progress
  def __call__(self, epoch, *args):
    self.progress['epoch'] = epoch

    training_loss = self.progress['training_loss']
    training_accuracy = self.progress['training_accuracy']

    result = dict(zip(*self.metric.get()))

    training_loss.append(result['cross-entropy'])
    training_accuracy.append(result['accuracy'])

    if self.solver.verbose:
      print 'epoch {:<3} training loss\t{}'.format(
        epoch,
        training_loss[-1]
      )
      print 'epoch {:<3} training accuracy\t{}'.format(
        epoch,
        training_accuracy[-1]
      )

    for callback in self.callbacks:
      callback(epoch, *args)

    # read learning rate from file
    print 'epoch {:<3} current learning rate: {}'.format(
      epoch, self.solver.scheduler.lr
    )
    if isinstance(self.solver.scheduler, MannualScheduler):
      with open(self.solver.file, 'r') as source:
        settings = source.read()
        settings = settings.replace('\n', '')
        settings = settings.split(';')[:-1]
        settings = [setting.split('=') for setting in settings]
        if any(len(setting) != 2 for setting in settings):
          raise Exception('uninterpretable configuration')
        settings = dict(settings)
        for key, value in settings.items():
          if key == 'lr':
            lr = float(settings['lr'])
            if self.solver.scheduler.lr != lr:
              self.solver.scheduler.lr = lr
              print 'epoch {:<3} learning rate {}'.format(epoch, lr)
    return

class MXSolver():
  def __init__(self, model, **kwargs):
    self.batch_size      = kwargs['batch_size']
    self.data            = kwargs['data']
    self.devices         = [mx.gpu(index) for index in kwargs['devices']]
    self.epoch           = kwargs['epoch']
    self.file            = kwargs.get('file', 'lr')
    self.verbose         = kwargs.get('verbose', False)

    self.batch_end_callback = kwargs.pop('batch_end_callback', [])

    if 'epoch_end_callback' not in kwargs:
      self.callbacks = []
    else:
      if isinstance(kwargs['epoch_end_callback'], list):
        self.callbacks = kwargs['epoch_end_callback']
      else:
        self.callbacks = [kwargs['epoch_end_callback']]

    self.optimizer_settings = kwargs.pop('optimizer_settings', {})
    if self.optimizer_settings['optimizer'] == 'SGD':
      self.optimizer_settings['optimizer'] = 'ccSGD'
    __optimizer_settings = {key : value for key, value in self.optimizer_settings.items()}
    batch_count = math.ceil(self.data[0].shape[0] / self.batch_size)

    if 'scheduler' in self.optimizer_settings:
      if self.optimizer_settings['scheduler'] == 'mannual':
        self.scheduler = MannualScheduler(self.optimizer_settings['lr'])
      else:
        self.scheduler = self.optimizer_settings['scheduler']
      self.optimizer_settings.pop('scheduler')
    else:
      self.scheduler = mx.lr_scheduler.FactorScheduler(
        step   = self.optimizer_settings.pop('lr_decay_interval') * batch_count,
        factor = self.optimizer_settings.pop('lr_decay_factor')
      )

    self.optimizer = getattr(
      mx.optimizer,
      self.optimizer_settings.pop('optimizer')
    )(
      learning_rate = self.optimizer_settings.pop('lr'),
      lr_scheduler  = self.scheduler,
      rescale_grad  = 1.0 / self.batch_size,
#     rescale_grad  = 1.0,
      wd            = self.optimizer_settings.pop('weight_decay', 0),
      **self.optimizer_settings
    )
    self.optimizer_settings = __optimizer_settings

    batch_shape = (self.batch_size,) + self.data[0].shape[1:]

    symbol, initializer = model(batch_shape)

    if self.verbose:
      args = symbol.list_arguments()
      arg_shapes, _, _ = symbol.infer_shape(data=self.data[0].shape)
      print '############################'
      print '# NETWORK PARAMETERS START #'
      print '############################'
      for arg, shape in zip(args, arg_shapes):
        if arg != 'data' and arg != 'softmax_label':
          print '{:<30}\t{}'.format(arg, shape)
      print '##########################'
      print '# NETWORK PARAMETERS END #'
      print '##########################'
      print
      print '#########################'
      print '# SOLVER SETTINGS START #'
      print '#########################'
      print 'scheduler', self.scheduler.__class__
      print '########################'
      print '# SOLVER SETTINGS ENDS #'
      print '########################'
      print 

    self.model = mx.model.FeedForward(
      ctx           = self.devices,
      initializer   = initializer,
      num_epoch     = self.epoch,
      optimizer     = self.optimizer,
      symbol        = symbol
    )

    print 'model constructed'

  def train(self):
    progress = {
      'epoch'               : 0,
      'training_loss'       : [],
      'training_accuracy'   : [],
      'validation_loss'     : [],
      'validation_accuracy' : [0]
    }

    logging.basicConfig(level=logging.NOTSET)
    logger = logging.getLogger()
    accuracy_filter = AccuracyFilter(
      self,
      self.data[2],
      self.data[3],
      self.batch_size,
      progress
    )
    logger.addFilter(accuracy_filter)
    logger.addFilter(BlockFilter())
    logger.addFilter(LRDecayFilter(self.optimizer_settings))
    logger.addFilter(TimeFilter(progress))

    metric = mx.metric.CompositeEvalMetric(
      metrics = [
        mx.metric.Accuracy(),
        mx.metric.CrossEntropy()
      ]
    )

    self.model.fit(
      X                  = self.data[0],
      y                  = self.data[1],
      eval_data          = (self.data[2], self.data[3]),
      eval_metric        = metric,
      batch_end_callback = self.batch_end_callback,
      epoch_end_callback = EpochCallback(self, metric, self.callbacks, progress),
      logger             = logger
    )

    arg_params, aux_params = accuracy_filter.arg_params, accuracy_filter.aux_params
    for key in self.model.arg_params:
      arg_params[key].copyto(self.model.arg_params[key])
    for key in self.model.aux_params:
      aux_params[key].copyto(self.model.aux_params[key])

    test_data = mx.io.NDArrayIter(self.data[4], self.data[5], batch_size=self.batch_size)
    test_accuracy= self.model.score(test_data)

    optimal_accuracy = max(progress['validation_accuracy'])
    epoch = progress['validation_accuracy'].index(optimal_accuracy)

    print 'optimal validation accuracy %f (epoch %d)' % (optimal_accuracy, epoch)
    print 'test accuracy %f' % (test_accuracy)

    return test_accuracy, progress
