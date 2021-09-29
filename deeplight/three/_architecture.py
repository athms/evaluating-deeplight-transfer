#!/usr/bin/env python
import numpy as np


def _init_model(
  keras,
  input_shape: int,
  n_classes: int,
  batch_size: int,
  return_logits: bool = True):
  """Setup 3D-DeepLight architecture,
  as specified in Thomas et al., 2021"""
  
  model = keras.models.Sequential()

  model.add(keras.layers.InputLayer(input_shape=input_shape))


  model.add(keras.layers.Conv3D(8, 3, strides=1,
    padding='same',
    data_format='channels_last',
    activation='relu'))
  model.add(keras.layers.SpatialDropout3D(0.2))
  model.add(keras.layers.BatchNormalization())

  model.add(keras.layers.Conv3D(8, 3, strides=1,
    padding='same',
    data_format='channels_last',
    activation='relu'))
  model.add(keras.layers.SpatialDropout3D(0.2))
  model.add(keras.layers.BatchNormalization())


  model.add(keras.layers.Conv3D(8, 3, strides=2,
    padding='same',
    data_format='channels_last',
    activation='relu'))
  model.add(keras.layers.SpatialDropout3D(0.2))
  model.add(keras.layers.BatchNormalization())

  model.add(keras.layers.Conv3D(8, 3, strides=1,
    padding='same',
    data_format='channels_last',
    activation='relu'))
  model.add(keras.layers.SpatialDropout3D(0.2))
  model.add(keras.layers.BatchNormalization())


  model.add(keras.layers.Conv3D(16, 3, strides=2,
    padding='same',
    data_format='channels_last',
    activation='relu'))
  model.add(keras.layers.SpatialDropout3D(0.2))
  model.add(keras.layers.BatchNormalization())

  model.add(keras.layers.Conv3D(16, 3, strides=1,
    padding='same',
    data_format='channels_last',
    activation='relu'))
  model.add(keras.layers.SpatialDropout3D(0.2))
  model.add(keras.layers.BatchNormalization())


  model.add(keras.layers.Conv3D(32, 3, strides=2,
    padding='same',
    data_format='channels_last',
    activation='relu'))
  model.add(keras.layers.SpatialDropout3D(0.2))
  model.add(keras.layers.BatchNormalization())

  model.add(keras.layers.Conv3D(32, 3, strides=1,
    padding='same',
    data_format='channels_last',
    activation='relu'))
  model.add(keras.layers.SpatialDropout3D(0.2))
  model.add(keras.layers.BatchNormalization())


  model.add(keras.layers.Conv3D(64, 3, strides=2,
    padding='same',
    data_format='channels_last',
    activation='relu'))
  model.add(keras.layers.SpatialDropout3D(0.2))
  model.add(keras.layers.BatchNormalization())

  model.add(keras.layers.Conv3D(64, 3, strides=1,
    padding='same',
    data_format='channels_last',
    activation='relu'))
  model.add(keras.layers.SpatialDropout3D(0.2))
  model.add(keras.layers.BatchNormalization())


  model.add(keras.layers.Conv3D(128, 3, strides=2,
    padding='same',
    data_format='channels_last',
    activation='relu'))
  model.add(keras.layers.SpatialDropout3D(0.2))
  model.add(keras.layers.BatchNormalization())

  model.add(keras.layers.Conv3D(128, 3, strides=1,
    padding='same',
    data_format='channels_last',
    activation='relu'))
  model.add(keras.layers.SpatialDropout3D(0.2))
  model.add(keras.layers.BatchNormalization())


  model.add(keras.layers.Conv3D(n_classes, 1, strides=1,
    padding='same',
    data_format='channels_last',
    activation=None))
  model.add(keras.layers.Activation("relu"))
  model.add(keras.layers.SpatialDropout3D(0.2))

  model.add(keras.layers.GlobalAveragePooling3D())

  if not return_logits:
    model.add(keras.layers.Activation("softmax"))

  return model