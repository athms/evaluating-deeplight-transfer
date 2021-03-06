#!/usr/bin/python
import numpy as np
import tensorflow as tf


def write_func_to_tfr(
    tfr_writers,
    func_data,
    states, trs,
    subject_id,
    task_id, run_id,
    n_onehot,
    onehot_task_idx,
    randomize_volumes=True):
  """Writes fMRI volumes and labels to TFRecord files.

  Args:
      tfr_writers: A sequence of TFRecord writers to store the data
      func_data: Ndarray fMRI data shape: (nx, ny, nz, time)
      states: A sequence, containing one numeric state label for each 
          volume in func_data
      trs: A sequence, containing TR of each volume in func_data
      subject_id: Integer ID of the subject that is stored in the
          TFR-files
      task_id: Integer ID of the HCP task that is stored in the 
          TFR-files
      run_id: Integer ID of the run that is stored  in the
          TFR-files
      n_onehot: Lenght of onehot vector.
      onehot_task_idx: Idx of task on onehot vector.
      randomize_volumes: Bool indicating whether the sequence of 
          volume (incl. their corresponding label) should be
          randomized before storing them in the TFR-files.

  Returns:
      None
  """
  func_data = np.array(func_data)
  states = np.array(states)
  trs = np.array(trs)
  nx, ny, nz, nv = func_data.shape
  vidx = np.arange(nv)
  if randomize_volumes:
      np.random.shuffle(vidx)
  for vi in vidx:
    writer = np.random.choice(tfr_writers)
    state = np.int(states[vi])
    state_onehot_idx = np.array(onehot_task_idx)[state]
    state_onehot = np.zeros(n_onehot)
    state_onehot[state_onehot_idx] = 1
    volume = np.array(func_data[:, :, :, vi].reshape(nx*ny*nz), dtype=np.float32)
    tr = trs[vi]
    v_sample = tf.train.Example(
      features=tf.train.Features(
        feature={
            'volume': tf.train.Feature( float_list = tf.train.FloatList(value=list(volume)) ),
            'task_id': tf.train.Feature( int64_list = tf.train.Int64List(value=[np.int64(task_id)]) ),
            'subject_id': tf.train.Feature( int64_list = tf.train.Int64List(value=[np.int64(subject_id)]) ),
            'run_id': tf.train.Feature( int64_list = tf.train.Int64List(value=[np.int64(run_id)]) ),
            'tr': tf.train.Feature( float_list = tf.train.FloatList(value=[np.float32(tr)]) ),
            'state': tf.train.Feature( int64_list = tf.train.Int64List(value=[np.int64(state)]) ),
            'onehot': tf.train.Feature( int64_list = tf.train.Int64List(value=list(state_onehot.astype(np.int64))) )
            }
        )
    )
    serialized = v_sample.SerializeToString()
    writer.write(serialized)


def parse_func_tfr(
    example_proto,
    nx, ny, nz,
    n_onehot=None,
    onehot_idx=None,
    only_parse_XY=False,
    transpose_xyz=False,
    add_channel_dim=False):
    """Parse TFR-data

    Args:
        example_proto: Single example from TFR-file
        nx, ny, nz: Integers indicating the x-/y-/z-dimensions
            of the fMRI data stored in the TFR-files
        n_onehot: Total number of states across tasks
        onehot_idx: idx that is returned from state-onehot;
            e.g., if state-onehot encoding has 20 values in total,
            but we only want to train with values 5-10,
            onehot_idx can be set to np.arange(4,10) 

    Returns:
        Parsed data stored in TFR-files. Specifically, the:

        volume: Ndarray of fMRI volume activations
        task_id: Integer ID of the HCP task 
        subject_id: Integer ID of the subject
        run_id: Integer ID of the run
        tr: TR of fmri volume (float)
        state: Integer cognitive state of the volume
        state_onehot: One-hot encoding of the states
        only_parse_XY: Bool indicating whether only volume 
            and y onehot encoding should be returned,
            as needed for integration with keras. If False,
            volume, task_id, subject_id, run_id, volume_idx,
            label, label_onehot are returned
    """
    features = {'volume': tf.FixedLenFeature([nx*ny*nz], tf.float32),
                'task_id': tf.FixedLenFeature([1], tf.int64),
                'subject_id': tf.FixedLenFeature([1], tf.int64),
                'run_id': tf.FixedLenFeature([1], tf.int64),
                'tr': tf.FixedLenFeature([1], tf.float32),
                'state': tf.FixedLenFeature([1], tf.int64),
                'onehot': tf.FixedLenFeature([n_onehot], tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    if onehot_idx is None:
        onehot_idx = np.arange(n_onehot)
    volume = tf.cast(tf.reshape(parsed_features["volume"], [nx, ny, nz]), tf.float32)
    if add_channel_dim:
        volume = tf.cast(tf.reshape(parsed_features["volume"], [nx, ny, nz, 1]), tf.float32)
        if transpose_xyz:
            volume = tf.transpose(volume, perm=[2, 1, 0, 3])
    elif transpose_xyz:
        volume = tf.transpose(volume, perm=[2, 1, 0])
    volume = tf.where(tf.is_nan(volume), tf.zeros_like(volume), volume)
    volume = tf.where(tf.is_inf(volume), tf.ones_like(volume)*1e4, volume)
    onehot = tf.cast(tf.gather(parsed_features["onehot"], onehot_idx), tf.int64)
    if only_parse_XY:
        return (volume, onehot)
    else:
        return {"volume": volume,
                "onehot": onehot,
                "task_id": parsed_features["task_id"],
                "subject_id": parsed_features["subject_id"],
                "run_id": parsed_features["run_id"],
                "tr": parsed_features["tr"],
                "state": parsed_features["state"]}


def make_dataset(
    files,
    n_onehot,
    batch_size,
    nx=91, ny=109, nz=91,
    onehot_idx=None,
    repeat=True,
    shuffle=True,
    only_parse_XY=False,
    n_workers=4,
    shuffle_buffer_size=500,
    scope_name='train',
    transpose_xyz=False,
    add_channel_dim=False):
  """Make iteratable dataset from TFR files."""
  if onehot_idx is None:
      onehot_idx = np.arange(n_onehot)
  with tf.variable_scope('data_queue_'+scope_name):
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(lambda x: parse_func_tfr(x,
        nx=nx, ny=ny, nz=nx,
        n_onehot=n_onehot,
        onehot_idx=onehot_idx,
        only_parse_XY=only_parse_XY,
        transpose_xyz=transpose_xyz,
        add_channel_dim=add_channel_dim), n_workers)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    if shuffle:
      dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    if repeat:
        dataset = dataset.repeat()
    return dataset