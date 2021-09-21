#!/usr/bin/python
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from nilearn import datasets, surface, plotting, image, masking
import deeplight
import hcprep


def main():

  # set random seed
  np.random.seed(24091)

  # parse arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("--architecture", required=False, default='3D',
                  help="DeepLight architecture (2D or 3D) (default: 3D)")
  ap.add_argument("--pretrained", required=False, default=1,
                  help="use pre-trained model? (1: True or 0_ False) (default: 1)")
  ap.add_argument("--task", required=False, default='MOTOR',
                  help="data of which task to interpret (EMOTION, GAMBLING, LANGUAGE, SOCIAL, MOTOR, RELATIONAL, WM)? (default: MOTOR)")
  ap.add_argument("--subject", required=False, default=100307, 
                  help="data of which subject to interpret? (default: 100307)")
  ap.add_argument("--run", required=False, default='LR', 
                  help="data of which run to interpret (LR or RL)? (default: LR)")
  ap.add_argument("--data", required=False, default='../data/',
                  help="path to TFR data files")
  ap.add_argument("--out", required=False,
                  help="path where DeepLight maps are saved")
  ap.add_argument("--verbose", required=False, default=1,
                  help="comment current program steps (0: no or 1: yes) (default: 1)")
  args = vars(ap.parse_args())
  # set variables
  architecture = str(args['architecture'])
  pretrained = bool(int(args['pretrained']))
  task = str(args['task'])
  subject = str(args['subject'])
  run = str(args['run'])
  verbose = bool(int(args['verbose']))
  data_path = str(args['data'])
  if args['out'] is not None:
    out_path = str(args['out'])
    print('Path: {}'.format(out_path))
  else:
    out_path = '../results/relevances/DeepLight/{}/brainmaps/'.format(architecture)
    print('"out" not defined. Defaulting to: {}'.format(out_path))

  # get hcp info
  hcp_info = hcprep.info.basics()

  # define subject out path
  sub_out_path = out_path+'sub-{}/'.format(subject)
  os.makedirs(sub_out_path, exist_ok=True) # and make sure path exists
  if verbose:
    print('\nSaving results to: {}'.format(sub_out_path))

  # load subject func img
  sub_bold_img = image.load_img(hcprep.paths.path_bids_func_mni(subject, task, run, data_path))
  sub_bold_mask = image.load_img(hcprep.paths.path_bids_func_mask_mni(subject, task, run, data_path))

  # get TFR file
  tfr_file = hcprep.paths.path_bids_tfr(subject, task, run, data_path)

  # make TFR iterable dataset
  dataset = deeplight.data.io.make_dataset(
    files=[tfr_file],
    n_onehot=20, # there are 20 cognitive states in the HCP data (so 20 total onehot entries)
    batch_size=1, # to save memory, we process 1 sample at a time!
    repeat=False,
    n_workers=2)
  
  # make iterator from dataset
  iterator = dataset.make_initializable_iterator()
  # get iterator entries
  iterator_features = iterator.get_next()

  # make DeepLight model
  if architecture == '3D':
    deeplight_variant = deeplight.three.model(
      batch_size=1,
      pretrained=pretrained,
      verbose=verbose)
  elif architecture == '2D':
    deeplight_variant = deeplight.two.model(
      batch_size=1,
      pretrained=pretrained,
      verbose=verbose)
  else:
      raise ValueError('Invalid value for DeepLight architecture. Must be 2D or 3D.')

  # setup LRP
  deeplight_variant.setup_lrp()

  # init tensorflow session
  sess = tf.Session()
  sess.run(iterator.initializer)

  # interpret
  if verbose:
    print('\nInterpreting predictions for task: {}, subject: {}, run: {}'.format(task, subject, run))
  states = []
  relevances = []
  trs = []
  acc = 0
  n = 0
  while True:
    try:
      # get volume info from TFR
      (batch_volume,
       batch_trs,
       batch_state,
       batch_onehot) = sess.run([iterator_features['volume'],
                                 iterator_features['tr'],
                                 iterator_features['state'],
                                 iterator_features['onehot']])
      # predict
      batch_pred = deeplight_variant.decode(batch_volume)
      # interpret
      batch_relevances = deeplight_variant.interpret(batch_volume) 
      # iterate batch samples
      for i in range(batch_state.shape[0]):
        # collect sample
        relevances.append(batch_relevances[i])
        states.append(batch_state[i])
        trs.append(batch_trs[i])
        # compute rolling accuracy
        acc += np.sum(batch_pred[i].argmax() == batch_onehot[i].argmax())
        n += 1
        if verbose and (n%10) == 0:
          print('\tDecoding accuracy after {} batches: {} %'.format(n, (acc/n)*100))
    except tf.errors.OutOfRangeError:
      break
  if verbose:
      print('..done.')
  
  # concat
  trs = np.concatenate(trs)
  states = np.concatenate(states)
  relevances = np.concatenate([np.expand_dims(r, 0) for r in relevances], axis=0)
  # we also transpose relevances, as DeepLight outputs relevances in shape (nz, ny, nx, 1)
  relevances = relevances.T[0]

  # sort relevances by and states by trs
  tr_idx = np.argsort(trs)
  relevances = relevances[...,tr_idx]
  states = states[tr_idx]

  # make img of relevance data
  sub_relevance_img = image.new_img_like(sub_bold_img, relevances)
  
  # save image to sub_out_path
  sub_relevance_img.to_filename(
    sub_out_path+'sub-{}_task-{}_run-{}_desc-relevances.nii.gz'.format(
      subject, task, run))

  # plot
  if verbose:
      print('\nPlotting brainmaps to: {}'.format(sub_out_path))
  for si, state in enumerate(hcp_info.states_per_task[task]):
      
      # subset imgs to cognitive state
      state_relevance_img = image.index_img(sub_relevance_img, states==si)
      # smooth relevance img for plotting
      state_relevance_img = image.smooth_img(state_relevance_img, fwhm=6)
      # average for cognitive state
      mean_state_relevance_img = image.mean_img(state_relevance_img)
      # save relevances for state
      mean_state_relevance_img.to_filename(
        sub_out_path+'sub-{}_task-{}_run-{}_desc-{}_avg_relevances.nii.gz'.format(
          subject, task, run, state))
      
      # extract 95 percentile threshold
      threshold = np.percentile(masking.apply_mask(mean_state_relevance_img, sub_bold_mask), 95)
      
      # plot surface brainmap
      plotting.plot_img_on_surf(mean_state_relevance_img,
                                views=['lateral', 'medial', 'ventral'],
                                hemispheres=['left', 'right'],
                                title='State: {}'.format(state),
                                colorbar=True,
                                cmap='inferno',
                                threshold=threshold);
      # save brainmap to subject-specific dir
      plt.savefig(sub_out_path+'sub-{}_task-{}_run-{}_desc-{}_avg_relevance_surf_brainmap.png'.format(
        subject, task, run, state), dpi=200)
      plt.clf() # close fig
      
      # also plot axial slices:
      plotting.plot_stat_map(mean_state_relevance_img, display_mode='z', cut_coords=30, cmap=plt.cm.seismic, threshold=threshold)
      plt.savefig(sub_out_path+'sub-{}_task-{}_run-{}_desc-{}_avg_relevance_axial_slices.png'.format(
        subject, task, run, state), dpi=200)
      plt.clf() # close fig

  if verbose:
    print('\n\n..done.')


if __name__ == '__main__':

  # run main
  main()

    



