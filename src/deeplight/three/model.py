#!/usr/bin/env python
import os
import numpy as np
import warnings
import tensorflow as tf
from innvestigate.analyzer.relevance_based.relevance_analyzer import LRPSequentialPresetBFlat
from einops import rearrange
from typing import Tuple
from ._architecture import make_keras_architecture, make_tf_architecture
from ._fit import _make_callbacks
import deeplight


class model(object):
  """A basic implementation of the 3D-DeepLight architecture
    as published in Thomas et al., 2021.

    Args:
      n_states (int, optional): How many cognitive states
          are in the output layer? Defaults to 16.
      pretrained (bool, optional): Should the model be initialized to
          the pretrained weights from Thomas et al., 2021? Defaults to True.
      batch_size (int, optional): How many samples per training batch? Defaults to 32.
      return_logits (bool, optional): Whether to return logits (or softmax). Defaults to True.
      verbose (bool, optional): Comment current program stages? Defaults to True.
      name (str, optional): Name of the model. Defaults to '3D'.
      input_shape (int, optional): Shape of input as (x, y, z)
            Defaults to MNI152NLin6Asym with shape (x: 91, y: 109, z: 91)
    """
  def __init__(
    self,
    n_states: int = 16,
    pretrained: bool = True,
    batch_size: int = 32,
    return_logits: bool = True,
    verbose: bool = True,
    name: str = '3D',
    input_shape:  Tuple[int, int, int] = (91, 109, 91)
    ) -> None:
    self.architecture = name
    self.pretrained = pretrained
    self.input_shape = input_shape # this is fixed for MNI152NLin6Asym
    self.n_states = n_states
    self.return_logits = return_logits
    self.batch_size = batch_size
    self.verbose = verbose
    if self.verbose:
      print('\nInitializing model:')
      print('\tarchitecture: {}'.format(name))
      print('\tpre-trained: {}'.format(self.pretrained))
      print('\tn-states: {}'.format(self.n_states))
      print('\tbatch-size: {}'.format(self.batch_size))

    self.model = make_tf_architecture(
      input_shape=self.input_shape,
      n_classes=self.n_states,
      batch_size=self.batch_size,
      return_logits=self.return_logits
    )

    if self.pretrained:
      self._path_pretrained_weights = os.path.join(
        os.path.dirname(deeplight.__file__),
        'three',
        'pretrained_model',
        'model-3D_DeepLight_desc-pretrained_model.hdf5'
      )
      self.load_weights(self._path_pretrained_weights)

  def load_weights(
    self,
    path: str
    ) -> None:
    """Load model weights from path."""
    stored_model = tf.keras.models.load_model(path)
    for model_layer, stored_model_layer in zip(self.model.layers, stored_model.layers):
      try:
        model_layer.set_weights(stored_model_layer.get_weights())
      except:
        print(
          'Cannot load weights for layer {} from {}, as shapes do not match'.format(
            model_layer.name, path)
        )

  def save_weights(
    self,
    path: str
    ) -> None:
    """Save model weights to path."""
    assert path.endswith('.hdf5'), 'Model needs to be stored in .hdf5 format, but {} was given!'.format(path)
    self.model.save(path)

  def _tranpose_volumes(
    self,
    volume):
    """tranpose (x, y, z) dimensions"""
    return rearrange(
      volume,
      'b x y z c  -> b z y x c'
    )

  def _add_channel_dim(
    self,
    volume):
    """add channel dimension to volume"""
    return np.expand_dims(volume, -1)

  def decode(
    self,
    volume):
    """Decode cognitive states for volume.

    Args:
        volume (array): Input volume with shape (batch-size, nx, ny, nz)

    Returns:
        array: Logits (or softmax), n x n_states
    """
    volume = self._add_channel_dim(volume)
    volume = self._tranpose_volumes(volume)
    return self.model.predict(volume)

  def fit(
    self,
    train_files: str,
    validation_files : str,
    n_onehot: int,
    onehot_idx: int,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    training_steps: int = 1000,
    validation_steps: int = 1000,
    output_path: str = 'out/',
    shuffle_buffer_size: int = 500,
    n_workers: int = 4
    ) -> tf.keras.callbacks.History:
    """Fit model.

    Args:
        train_files (list): List of paths to TFR training files.
        validation_files (list): List of paths to TFR training files.
        n_onehot (int, optional): Number of onehot states encoded in "onehot"
            entry of TFR files. Defaults to 20.
        onehot_idx (int): Array of int, indicating the onehot indices
            to use from "onehot" in TFR files; e.g., if the dataset
            contains 20 onehot-states, but you only want to train on 
            cognitive states 5-10, this would be np.arange(4, 10).
        learning_rate (float, optional): Learning rate for ADAM. Defaults to 1e-4.
        epochs (int, optional): How many traning epochs to run? Defaults to 50.
        training_steps (int, optional): How many steps per training epoch?
            Defaults to 1000.
        validation_steps (int, optional): How many validation steps per training epoch.
            Defaults to 1000.
        output_path (str, optional): Where to store model weights of each training epoch 
            and training history. Defaults to 'out/'.
        shuffle_buffer_size (int, optional): Size of training shuffle buffer. Defaults to 500.
        n_workers (int, optional): How many workers for data loading. Defaults to 4.

    Returns:
        DataFrame: Training history.
    """
    os.makedirs(output_path, exist_ok=True)

    train_dataset = deeplight.data.io.make_dataset(
      files=train_files,
      batch_size=self.batch_size,
      nx=self.input_shape[0],
      ny=self.input_shape[1],
      nz=self.input_shape[2],
      shuffle=True,
      only_parse_XY=True,
      transpose_xyz=True,
      add_channel_dim=True,
      repeat=True,
      n_onehot=n_onehot,
      onehot_idx=onehot_idx,
      shuffle_buffer_size=shuffle_buffer_size,
      n_workers=n_workers
    )
    
    val_dataset = deeplight.data.io.make_dataset(
      files=validation_files,
      batch_size=self.batch_size,
      nx=self.input_shape[0],
      ny=self.input_shape[1],
      nz=self.input_shape[2],
      shuffle=False,
      only_parse_XY=True,
      transpose_xyz=True,
      add_channel_dim=True,
      repeat=True,
      n_onehot=n_onehot,
      onehot_idx=onehot_idx,
      shuffle_buffer_size=shuffle_buffer_size,
      scope_name='validation',
      n_workers=n_workers
    )
    
    callbacks = _make_callbacks(output_path)
    
    self.model.compile(
      optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
      loss='categorical_crossentropy',
      metrics=['accuracy']
    )
    
    if self.verbose:
      print('Starting training...')
    history = self.model.fit(
      train_dataset,
      epochs=epochs,
      steps_per_epoch=training_steps,
      validation_data=val_dataset,
      validation_steps=validation_steps,
      callbacks=callbacks,
      verbose=np.int(self.verbose),
      use_multiprocessing=n_workers>1,
      workers=n_workers
    )

    return history

  def setup_lrp(self) -> None:
    """Setup LRP for 3D-DeepLight."""
    if self.return_logits is False:
      warnings.warning(
        '"'"return_logits"'" should be set to "'"True"'" when computing LRP.'
      )
    
    if self.verbose:
      print("\tSetting up LRP analyzer..")
    
    # rebuild model in keras (required for iNNvestigate)
    analyzer_model = make_keras_architecture(
      input_shape=self.input_shape,
      n_classes=self.n_states,
      batch_size=self.batch_size,
      return_logits=self.return_logits
    )
    
    for model_layer, stored_model_layer in zip(
      analyzer_model.layers, self.model.layers):
      model_layer.set_weights(
        stored_model_layer.get_weights()
      )
    
    self._analyzer = LRPSequentialPresetBFlat(
      model=analyzer_model,
      neuron_selection_mode="index",
      epsilon=1e-6
    )

  def interpret(
    self,
    volume):
    """Interpret decoding decision for volume.

    Args:
        volume (array): Input volume with shape (batch-size, nx, ny, nz)

    Returns:
        array: relevance values for each voxel of volume
    """
    pred_batch = self.decode(volume).argmax(axis=1)
    volume = self._add_channel_dim(volume)
    volume = self._tranpose_volumes(volume)
    R = self._analyzer.analyze(volume, neuron_selection=pred_batch)
    return self._tranpose_volumes(R)[...,0] # removing channel dim