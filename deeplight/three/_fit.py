#!/usr/bin/env python
import tensorflow as tf


def _make_callbacks(
  output_path: str
  ) -> list:
  """Setup checkpoint callbacks and csv logger for training."""
  # model checkpoint
  filepath = output_path+"epoch-{epoch:03d}.h5"
  checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath,
    monitor='val_acc',
    verbose=1,
    save_best_only=False,
    period=1
  )
  csv_logger = tf.keras.callbacks.CSVLogger(
    output_path+'history.csv',
    separator=",",
    append=True
  )
  return [checkpoint, csv_logger]