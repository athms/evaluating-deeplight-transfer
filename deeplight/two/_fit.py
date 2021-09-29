#!/usr/bin/env python
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from deeplight.data import io


def _fit(self,
        train_files: str,
        validation_files: str,
        n_onehot: int,
        onehot_idx: int,
        learning_rate: float,
        batch_size: int,
        epochs: int,
        training_steps: int,
        validation_steps: int,
        output_path: str,
        verbose: bool = True,
        shuffle_buffer_size: int = 500,
        n_workers: int = 4):
  """Fit model."""
  files = tf.placeholder(tf.string, shape=[None])
  dataset = io.make_dataset(
    files=files,
    batch_size=batch_size,
    nx=self.input_shape[2],
    ny=self.input_shape[1],
    nz=self.input_shape[0],
    shuffle=True,
    only_parse_XY=True,
    n_onehot=n_onehot,
    onehot_idx=onehot_idx,
    shuffle_buffer_size=shuffle_buffer_size,
    n_workers=n_workers)

  iterator = dataset.make_initializable_iterator()
  volume, onehot = iterator.get_next()
  # stack volumes by merging (batch size x nz)
  volume = tf.reshape(volume, [batch_size*self.input_shape[0], self.input_shape[1], self.input_shape[2], 1])

  with tf.variable_scope('model', reuse=tf.AUTO_REUSE): 
    logits = self.model.forward(volume)

  with tf.variable_scope('backward'):
    Xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
      logits=logits,
      labels=onehot)
    avg_Xentropy = tf.reduce_mean(Xentropy)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    optimizer_step = optimizer.minimize(avg_Xentropy)

  with tf.variable_scope('metrics'):
    pred = tf.argmax(logits, axis=1)
    state = tf.argmax(onehot, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, state), tf.float32))

  saver = tf.train.Saver(max_to_keep=epochs+1)

  self.sess.run(tf.global_variables_initializer()) 

  if self.pretrained:
    self.load_weights(self._path_pretrained_tvars, verbose=False)

  history = []
  for epoch in range(1, epochs+1):
    print('\n\nTraining epoch: {}'.format(epoch))
  
    xe_train, acc_train = 0, 0
    np.random.shuffle(train_files)
    self.sess.run(iterator.initializer, feed_dict={files: train_files})
    with tqdm(total=training_steps) as pbar:
      for step_train in range(1,training_steps+1):
        try:
          _, batch_acc, batch_xe = self.sess.run([optimizer_step, accuracy, avg_Xentropy],
            feed_dict={self._keep_prob: 0.5, self._conv_keep_probs: np.array([1, .8, .6])})
          xe_train += batch_xe
          acc_train += batch_acc
          pbar.set_description(
              "loss: {:02f} - acc: {:02f}".format(
                  xe_train/float(step_train), acc_train/float(step_train)),
              refresh=True)
          pbar.update()
        except tf.errors.DataLossError:
          continue
        except (tf.errors.OutOfRangeError, tf.errors.InvalidArgumentError):
          break

    xe_val, acc_val = 0, 0
    self.sess.run(iterator.initializer, feed_dict={files: validation_files})
    for step_val in range(1,validation_steps+1):
      try:
        batch_acc, batch_xe = self.sess.run([accuracy, avg_Xentropy],
          feed_dict={self._keep_prob: 1, self._conv_keep_probs: np.ones(3)})
        xe_val += batch_xe
        acc_val += batch_acc
      except tf.errors.DataLossError:
        continue
      except (tf.errors.OutOfRangeError, tf.errors.InvalidArgumentError):
        break
    if self.verbose:
      print('Val. data: acc: {:02f} - loss: {:02f}'.format(
        acc_val/float(step_val), xe_val/float(step_val)))

    history.append(pd.DataFrame({'epoch': epoch,
                                  'accuracy': acc_train/float(step_train),
                                  'loss': xe_train/float(step_train),
                                  'val_accuracy': acc_val/float(step_val),
                                  'val_loss': xe_val/float(step_val)},
                                  index=np.array([epoch-1])))
    pd.concat(history).to_csv(output_path+'history.csv')

    self.save_weights(path=output_path+"epoch-{:03d}.npy".format(epoch))

  return pd.concat(history)