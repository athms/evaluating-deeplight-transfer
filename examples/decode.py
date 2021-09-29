#!/usr/bin/python
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from nilearn import image
import deeplight
import hcprep


def main():

  np.random.seed(24091)

  ap = argparse.ArgumentParser()
  ap.add_argument("--data", required=False, default='../data/',
                  help="path to TFR data files")
  ap.add_argument("--architecture", required=False, default='3D',
                  help="DeepLight architecture (2D or 3D) (default: 3D)")
  ap.add_argument("--pretrained", required=False, default=1,
                  help="use pre-trained model? (1: True or 0: False) (default: 1)")
  ap.add_argument("--task", required=False, default='MOTOR',
                  help="which HCP task? (EMOTION, GAMBLING, LANGUAGE,"\
                    "SOCIAL, MOTOR, RELATIONAL)? (default: MOTOR)")
  ap.add_argument("--subject", required=False, default=100307, 
                  help="which subject? (default: 100307)")
  ap.add_argument("--run", required=False, default='LR', 
                  help="which run? (LR or RL)? (default: LR)")
  ap.add_argument("--batch-size", required=False, default=32,
                  help="Batch size for prediction (default: 32)")
  ap.add_argument("--out", required=False, default=None,
                  help="path where predictions are saved (default: ../results/predictions/)")
  ap.add_argument("--verbose", required=False, default=1,
                  help="comment current program steps (0: no or 1: yes) (default: 1)")
  
  args = vars(ap.parse_args())
  architecture = str(args['architecture'])
  pretrained = bool(int(args['pretrained']))
  task = str(args['task'])
  subject = str(args['subject'])
  run = str(args['run'])
  batch_size = int(args['batch_size'])
  verbose = bool(int(args['verbose']))
  data_path = str(args['data'])
  if args['out'] is not None:
    out_path = str(args['out'])
    print('Path: {}'.format(out_path))
  else:
    out_path = '../results/predictions/DeepLight/{}/pretrained-{}/'.format(architecture, pretrained)
    print('"out" not defined. Defaulting to: {}'.format(out_path))

  # make sure task specification is valid
  assert task in ['EMOTION', 'LANGUAGE', 'SOCIAL', 'GAMBLING', 'MOTOR', 'RELATIONAL'],\
    'Invalid task; only pre-training tasks valid (all HCP tasks, except for WM)!'

  hcp_info = hcprep.info.basics()

  sub_out_path = out_path+'sub-{}/'.format(subject)
  os.makedirs(sub_out_path, exist_ok=True)
  if verbose:
    print('\nSaving predictions to: {}'.format(sub_out_path))

  sub_bold_img = image.load_img(hcprep.paths.path_bids_func_mni(subject, task, run, data_path))
  sub_bold_mask = image.load_img(hcprep.paths.path_bids_func_mask_mni(subject, task, run, data_path))

  tfr_file = hcprep.paths.path_bids_tfr(subject, task, run, data_path)

  dataset = deeplight.data.io.make_dataset(
    files=[tfr_file],
    n_onehot=20, # there are 20 cognitive states in the HCP data (so 20 total onehot entries)
    batch_size=batch_size,
    repeat=False,
    n_workers=2)

  iterator = dataset.make_initializable_iterator()
  iterator_features = iterator.get_next()

  if architecture == '3D':
    deeplight_variant = deeplight.three.model(
      batch_size=batch_size,
      n_states=16, # pre-trained DeepLight has 16 output states
      pretrained=pretrained,
      verbose=verbose)
  elif architecture == '2D':
    deeplight_variant = deeplight.two.model(
      batch_size=batch_size,
      n_states=16, # pre-trained DeepLight has 16 output states
      pretrained=pretrained,
      verbose=verbose)
  else:
      raise ValueError('Invalid value for DeepLight architecture. Must be 2D or 3D.')

  sess = tf.Session()
  sess.run(iterator.initializer)

  print('\nPredicting for subject {} in run {} of task {}'.format(subject, run, task))
  i, acc = 1, 0
  predictions = [] 
  while True:
    try:
      tr, onehot, volume = sess.run((iterator_features['tr'],
                                     iterator_features['onehot'],
                                     iterator_features['volume']))
      pred = deeplight_variant.decode(volume)
      for ii in range(tr.shape[0]):
          predictions.append(pd.DataFrame({'subject': subject,
                                           'task': task,
                                           'run': run,
                                           'tr': tr[ii],
                                           'true_state': onehot[ii].argmax(),
                                           'pred_state': pred[ii].argmax()},
                                          index=[i]))
          acc += (onehot[ii].argmax() == pred[ii].argmax())
          if verbose and (i%10) == 0:
            print('\tBatch: {}; acc: {} %'.format(i, (acc/i)*100))
          i += 1
    except tf.errors.OutOfRangeError:
      break
    except ValueError:
      break
  predictions = pd.concat(predictions)
  print('Total decoding accuracy: {} %'.format(np.mean(predictions['true_state']==predictions['pred_state']) * 100))
  predictions.to_csv(sub_out_path+'task-{}_sub-{}_run-{}_desc-predictions_{}-DeepLight_pretrained-{}.csv'.format(
    task, subject, run, architecture, pretrained), index=False)


if __name__ == '__main__':

  main()

    



