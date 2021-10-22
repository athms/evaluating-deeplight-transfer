#!/usr/bin/env python
import modules
from ._dropout import Dropout
from ._lstm import LSTM_bidirectional
import tensorflow as tf
import numpy as np


def feature_extractor(
  nx: int, ny: int, nz: int,
  batch_size: int,
  conv_keep_probs: float = np.ones(3)):
  """Setup 2D-DeepLight convolutional feature extractor,
  as specified in Thomas et al., 2021"""
  return [modules.Convolution(input_depth=1,
                              output_depth=8,
                              kernel_size=3,
                              batch_size=batch_size*nz,
                              input_dim=(nx, ny),
                              stride_size=1,
                              act='relu',
                              pad='SAME',
                              weights_init=tf.glorot_normal_initializer()),
          Dropout(keep_prob=conv_keep_probs[0], noise_shape=[batch_size*nz, 1, 1, 8]),

          modules.Convolution(input_depth=8,
                              output_depth=8,
                              kernel_size=3,
                              stride_size=1,
                              act='relu',
                              pad='SAME',
                              weights_init=tf.glorot_normal_initializer()),
          Dropout(keep_prob=conv_keep_probs[0], noise_shape=[batch_size*nz, 1, 1, 8]),

          modules.Convolution(input_depth=8,
                              output_depth=16,
                              kernel_size=3,
                              stride_size=2,
                              act='relu',
                              pad='SAME',
                              weights_init=tf.glorot_normal_initializer()),
          Dropout(keep_prob=conv_keep_probs[0], noise_shape=[batch_size*nz, 1, 1, 16]),

          modules.Convolution(input_depth=16,
                              output_depth=16,
                              kernel_size=3,
                              stride_size=1,
                              act='relu',
                              pad='SAME',
                              weights_init=tf.glorot_normal_initializer()),
          Dropout(keep_prob=conv_keep_probs[0], noise_shape=[batch_size*nz, 1, 1, 16]),


          modules.Convolution(input_depth=16,
                              output_depth=32,
                              kernel_size=3,
                              stride_size=2,
                              act='relu',
                              pad='SAME',
                              weights_init=tf.glorot_normal_initializer()),
          Dropout(keep_prob=conv_keep_probs[1], noise_shape=[batch_size*nz, 1, 1, 32]),

          modules.Convolution(input_depth=32,
                              output_depth=32,
                              kernel_size=3,
                              stride_size=1,
                              act='relu',
                              pad='SAME',
                              weights_init=tf.glorot_normal_initializer()),
          Dropout(keep_prob=conv_keep_probs[1], noise_shape=[batch_size*nz, 1, 1, 32]),

          modules.Convolution(input_depth=32,
                              output_depth=32,
                              kernel_size=3,
                              stride_size=2,
                              act='relu',
                              pad='SAME',
                              weights_init=tf.glorot_normal_initializer()),
          Dropout(keep_prob=conv_keep_probs[1], noise_shape=[batch_size*nz, 1, 1, 32]),

          modules.Convolution(input_depth=32,
                              output_depth=32,
                              kernel_size=3,
                              stride_size=1,
                              act='relu',
                              pad='SAME',
                              weights_init=tf.glorot_normal_initializer()),
          Dropout(keep_prob=conv_keep_probs[1], noise_shape=[batch_size*nz, 1, 1, 32]),


          modules.Convolution(input_depth=32,
                              output_depth=64,
                              kernel_size=3,
                              stride_size=2,
                              act='relu',
                              pad='SAME',
                              weights_init=tf.glorot_normal_initializer()),
          Dropout(keep_prob=conv_keep_probs[2], noise_shape=[batch_size*nz, 1, 1, 64]),

          modules.Convolution(input_depth=64,
                              output_depth=64,
                              kernel_size=3,
                              stride_size=1,
                              act='relu',
                              pad='SAME',
                              weights_init=tf.glorot_normal_initializer()),
          Dropout(keep_prob=conv_keep_probs[2], noise_shape=[batch_size*nz, 1, 1, 64]),

          modules.Convolution(input_depth=64,
                              output_depth=64,
                              kernel_size=3,
                              stride_size=2,
                              act='relu',
                              pad='SAME',
                              weights_init=tf.glorot_normal_initializer()),
          Dropout(keep_prob=conv_keep_probs[2], noise_shape=[batch_size*nz, 1, 1, 64]),

          modules.Convolution(input_depth=64,
                              output_depth=64,
                              kernel_size=3,
                              stride_size=1,
                              act='relu',
                              pad='SAME',
                              weights_init=tf.glorot_normal_initializer()),
          Dropout(keep_prob=conv_keep_probs[2], noise_shape=[batch_size*nz, 1, 1, 64])]


def lstm_bidirectional(
  nz: int,
  batch_size: int,
  n_classes: int,
  keep_prob: float = 1.):
  """Setup 2D-DeepLight bidirectional LSTM unit,
  as specified in Thomas et al., 2021"""
  return [LSTM_bidirectional(input_dim=4*3*64,
                             dim=64,
                             sequence_length=nz,
                             batch_size=batch_size,
                             weights_init=tf.glorot_normal_initializer()),
          Dropout(keep_prob=keep_prob)]


def output_unit(
  n_classes: int,
  return_logits: bool = True):
  """Setup 2D-DeepLight output unit,
  as specified in Thomas et al., 2021"""
  if return_logits:
    return [modules.Linear(n_classes, weights_init=tf.glorot_normal_initializer())]
  else:
    return [modules.Linear(n_classes, weights_init=tf.glorot_normal_initializer()),
            modules.Softmax(n_classes, weights_init=tf.glorot_normal_initializer())]



def make_architecture(
  input_shape: int,
  n_classes: int,
  batch_size: int,
  conv_keep_probs: float = np.ones(3),
  keep_prob: float = 1.,
  return_logits: bool = True):
  """Setup 2D-DeepLight architecture,
  as specified in Thomas et al., 2021"""
  nz, ny, nx = input_shape
  return modules.Sequential(
    feature_extractor(
      nx=nx, ny=ny, nz=nz,
      batch_size=batch_size,
      conv_keep_probs=conv_keep_probs
      ) +
    lstm_bidirectional(
      nz=nz,
      batch_size=batch_size,
       n_classes=n_classes, keep_prob=keep_prob) +
    output_unit(
      n_classes=n_classes,
      return_logits=return_logits
    )
  )