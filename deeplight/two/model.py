#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import warnings
from einops import rearrange
from typing import Tuple
from ._architecture import make_architecture
import deeplight


class model(object):
  """2D-DeepLight."""
  def __init__(self,
    n_states: int = 16,
    pretrained: bool = True,
    batch_size: int = 32,
    return_logits: bool = True,
    verbose: bool = True,
    name: str = '2D',
    input_shape: Tuple[int, int, int] = (91, 109, 91)
    ) -> None:
    """A basic implementation of the 2D-DeepLight architecture
    as published in Thomas et al., 2021.

    Args:
        n_states (int, optional): How many cognitive states
            are in the output layer? Defaults to 16.
        pretrained (bool, optional): Should the model be  initialized to
            the pretrained weights from Thomas et al., 2021? Defaults to True.
        batch_size (int, optional): How many samples per training batch? Defaults to 32.
        return_logits (bool, optional): Whether to return logits or softmax. Defaults to True.
        verbose (bool, optional): Comment current program stages? Defaults to True.
        name (str, optional): Name of the model. Defaults to '2D'.
        input_shape (int, optional): Shape of input as (x, y, z)
            Defaults to MNI152NLin6Asym with shape (x: 91, y: 109, z: 91)
    """
    self.architecture = name
    self.pretrained = pretrained
    self.input_shape = input_shape # fixed for MNI152NLin6Asym space
    self.n_states = n_states
    self.return_logits = return_logits
    self.batch_size = batch_size
    self.verbose = verbose
    self._R = None
    if self.verbose:
      print('\nInitializing model:')
      print('\tarchitecture: {}'.format(name))
      print('\tpre-trained: {}'.format(self.pretrained))
      print('\tn-states: {}'.format(self.n_states))
      print('\tbatch-size: {}'.format(self.batch_size))
    
    with tf.variable_scope('placeholder'):
      self._conv_keep_probs = tf.placeholder(tf.float32, [3])
      self._keep_prob = tf.placeholder(tf.float32, [])

    self.sess = tf.Session()

    with tf.variable_scope('model'):
      self.model = make_architecture(
          input_shape=self.input_shape,
          n_classes=self.n_states,
          batch_size=self.batch_size,
          conv_keep_probs=self._conv_keep_probs,
          keep_prob=self._keep_prob,
          return_logits=self.return_logits
        )
      self._volume = tf.placeholder(
        tf.float32,
        [self.batch_size*self.input_shape[0],
         self.input_shape[1],
         self.input_shape[2], 1]
      )
      self._logits = self.model.forward(self._volume)
      self._tvars = tf.trainable_variables()

    self.sess.run(tf.global_variables_initializer())

    self._path_pretrained_tvars = os.path.join(
      os.path.dirname(deeplight.__file__),
      'two', 'pretrained_model', 'model-2D_DeepLight_desc-pretrained_model.npy'
    )
    if self.pretrained:
      self.load_weights(self._path_pretrained_tvars)

  def load_weights(self, path, verbose=None):
    """load model weights from path."""
    verbose = verbose if verbose is not None else self.verbose
    stored_tvars = np.load(path, allow_pickle=True)[()]
    for tv in self._tvars:
      try:
        self.sess.run(tv.assign(stored_tvars[str(tv)]))
      except:
        if verbose:
          print('Cannot load weights for {} from {}, as shapes do not match'.format(str(tv), path))
        continue
  
  def save_weights(self, path):
    """save model weights to path."""
    tvars = self.sess.run(self._tvars)
    tvars_out = {str(tn): tv for tn, tv in zip(self._tvars, tvars)}
    np.save(path, tvars_out)

  def _stack_volumes(self, volume):
    """merge batch and z-axis"""
    return rearrange(volume, 'b z y x c -> (b z) y x c', z=self.input_shape[2])
  
  def _unstack_volumes(self, volume):
    """unmerge batch and z-axis"""
    return rearrange(volume, '(b z) y x c -> b z y x c', z=self.input_shape[2])

  def _tranpose_volumes(self, volume):
    """tranpose (x, y, z) dimensions"""
    return rearrange(volume, 'b x y z c  -> b z y x c')

  def _add_channel_dim(self, volume):
    """add channel dimension to volume"""
    return np.expand_dims(volume, -1)

  def decode(self, volume):
    """Decode cognitive states for volume.

    Args:
        volume (array): Input volume with shape (batch-size, nx, ny, nz)

    Returns:
        array: Logits (or softmax), n x n_states
    """
    volume = self._add_channel_dim(volume)
    volume = self._tranpose_volumes(volume)
    volume = self._stack_volumes(volume)
    return self.sess.run(self._logits,
        feed_dict={
          self._volume: volume,
          self._keep_prob: 1,
          self._conv_keep_probs: np.ones(3)
          }
        )

  def fit(self,
    train_files: list,
    validation_files: list,
    onehot_idx: int,
    n_onehot: int = 20,
    learning_rate: float = 1e-4,
    epochs: int = 50,
    training_steps: int = 1000,
    validation_steps: int = 1000,
    output_path: str = 'out/',
    shuffle_buffer_size: int = 500,
    n_workers: int = 4):
    """Fit the model.

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

    files = tf.placeholder(tf.string, shape=[None])
    dataset = deeplight.data.io.make_dataset(
      files=files,
      batch_size=self.batch_size,
      nx=self.input_shape[0],
      ny=self.input_shape[1],
      nz=self.input_shape[2],
      shuffle=True,
      only_parse_XY=True,
      transpose_xyz=True,
      add_channel_dim=True,
      n_onehot=n_onehot,
      onehot_idx=onehot_idx,
      shuffle_buffer_size=shuffle_buffer_size,
      n_workers=n_workers)

    iterator = dataset.make_initializable_iterator()
    volume, onehot = iterator.get_next()
    # TODO: avoid hardcoding reshape
    volume = tf.reshape(
      volume,
      [self.batch_size*self.input_shape[2],
       self.input_shape[1],
       self.input_shape[0],
       1]
    )

    with tf.variable_scope('model', reuse=tf.AUTO_REUSE): 
      logits = self.model.forward(volume)

    with tf.variable_scope('backward'):
      self._xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits,
        labels=onehot)
      self._avg_xentropy = tf.reduce_mean(self._xentropy)
      optimizer = tf.train.AdamOptimizer(learning_rate)
      self._optimizer_step = optimizer.minimize(self._avg_xentropy)

    with tf.variable_scope('metrics'):
      pred = tf.argmax(logits, axis=1)
      state = tf.argmax(onehot, axis=1)
      self._accuracy = tf.reduce_mean(
        tf.cast(tf.equal(pred, state), tf.float32)
      )

    self.sess.run(tf.global_variables_initializer()) 

    if self.pretrained:
      self.load_weights(self._path_pretrained_tvars, verbose=False)

    history = []
    for epoch in range(1, epochs+1):
      print('\n\nTraining epoch: {}'.format(epoch))
    
      np.random.shuffle(train_files)
      self.sess.run(iterator.initializer, feed_dict={files: train_files})
      xe_train, acc_train = self._train_loop(n_steps=training_steps)
      
      self.sess.run(iterator.initializer, feed_dict={files: validation_files})
      xe_val, acc_val = self._val_loop(n_steps=validation_steps)

      history.append(
        pd.DataFrame(
          {
          'epoch': epoch,
          'accuracy': acc_train,
          'loss': xe_train,
          'val_accuracy': acc_val,
          'val_loss': xe_val
          }, 
          index=np.array([epoch-1])
        )
      )
      pd.concat(history).to_csv(output_path+'history.csv')

      self.save_weights(path=output_path+"epoch-{:03d}.npy".format(epoch))

    return pd.concat(history)

  def _train_loop(self, n_steps):
    """Perform a training loop for n_steps
    """
    xe, acc = 0, 0
    with tqdm(total=n_steps) as pbar:
      for step in range(1,n_steps+1):
        try:
          _, batch_acc, batch_xe = self.sess.run(
            [self._optimizer_step, self._accuracy, self._avg_xentropy],
            feed_dict={self._keep_prob: 0.5,
                       self._conv_keep_probs: np.array([1, .8, .6])
            }
          )
          xe += batch_xe
          acc += batch_acc
          pbar.set_description(
              "loss: {:02f} - acc: {:02f}".format(
                  xe/float(step), acc/float(step)),
              refresh=True)
          pbar.update()
        except tf.errors.DataLossError:
          continue
        except (tf.errors.OutOfRangeError, tf.errors.InvalidArgumentError):
          break
    return xe / float(step), acc / float(step)

  def _val_loop(self, n_steps):
    """Perform a validation loop for n_steps (no weight updates!)
    """
    xe, acc = 0, 0
    for step in range(1,n_steps+1):
      try:
        batch_acc, batch_xe = self.sess.run(
          [self._accuracy, self._avg_xentropy],
          feed_dict={self._keep_prob: 1,
                     self._conv_keep_probs: np.ones(3)
                     }
        )
        xe += batch_xe
        acc += batch_acc
      except tf.errors.DataLossError:
        continue
      except (tf.errors.OutOfRangeError, tf.errors.InvalidArgumentError):
        break
    if self.verbose:
      print('Val. data: acc: {:02f} - loss: {:02f}'.format(
        acc/float(step), xe/float(step)))
    return xe / float(step), acc / float(step)

  def setup_lrp(self):
    """Setup LRP computation, as specified in Thomas et al., 2021.
    """
    if self.return_logits is False:
        warnings.warning('"'"return_logits"'" should be set to "'"True"'" when computing LRP.')
    if self.verbose:
        print("\tSetting up LRP analyzer..")
    with tf.variable_scope('relevance', reuse=tf.AUTO_REUSE):
        # subset logits to predicted class
        self._R = tf.where( 
          tf.equal(
            tf.one_hot(tf.argmax(self._logits, axis=1), depth=self.n_states),
            1),
          self._logits,
          tf.zeros_like(self._logits)
        )
        # backpropagate relevances
        for layer in self.model.modules[::-1]:
          self._R = self.model.lrp_layerwise(layer, self._R, 'epsilon', 1e-3)

  def interpret(self, volume):
    """Interpret decoding decision for volume.

    Args:
        volume (array): Input volume with shape (batch-size, nx, ny, nz)

    Returns:
        array: relevance values for each voxel of volume
    """
    if self._R is None:
      raise NotImplementedError('LRP is not initialized. Please call .setup_lrp() first.')
    volume = self._add_channel_dim(volume)
    volume = self._tranpose_volumes(volume)
    volume = self._stack_volumes(volume)
    R = self.sess.run(
      self._R,
      feed_dict={
        self._volume: volume,
        self._keep_prob: 1,
        self._conv_keep_probs: np.ones(3)
      }
    )
    R = self._unstack_volumes(R)
    return self._tranpose_volumes(R)[...,0] # removing channel dim