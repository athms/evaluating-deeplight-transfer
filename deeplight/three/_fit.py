#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from deeplight.data import io


def _make_callbacks(output_path: str):
  """Setup checkpoint callbacks and csv logger for training."""
  # model checkpoint
  filepath = output_path+"epoch-{epoch:03d}.h5"
  checkpoint = tf.keras.callbacks.ModelCheckpoint(
  filepath, monitor='val_acc', verbose=1, save_best_only=False, period=1)
  csv_logger = tf.keras.callbacks.CSVLogger(output_path+'history.csv', separator=",", append=True)
  return [checkpoint, csv_logger]


def _fit(self,
  train_files: str,
  validation_files: str,
  n_onehot: int,
  onehot_idx: int,
  learning_rate: float,
  epochs: int,
  batch_size: int,
  training_steps: int,
  validation_steps: int,
  output_path: str,
  verbose: bool = True, 
  shuffle_buffer_size: int = 500,
  n_workers: int = 4):
  """Fit model."""
  
  train_dataset = io.make_dataset(
    files=train_files,
    batch_size=batch_size,
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
    n_workers=n_workers)
  
  val_dataset = io.make_dataset(
    files=validation_files,
    batch_size=batch_size,
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
    n_workers=n_workers)
  
  callbacks = _make_callbacks(output_path)
  
  self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  
  if verbose:
    print('Starting training...')
  history = self.model.fit(
    train_dataset,
    epochs=epochs,
    steps_per_epoch=training_steps,
    validation_data=val_dataset,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=np.int(verbose),
    use_multiprocessing=n_workers>1,
    workers=n_workers)
  
  return (history, self.model)