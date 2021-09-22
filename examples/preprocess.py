#!/usr/bin/python
import os
import argparse
import numpy as np
import tensorflow as tf
import hcprep
import deeplight


def main():
  # set random seed
  np.random.seed(52698)

  # parse arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("--path", required=False, default='../data/',
                  help="path where HCP data is stored (default: ../data/)")
  args = vars(ap.parse_args())
  path = str(args['path'])
  
  # extract subject ids
  subjects = [int(f.split('sub-')[1]) for f in os.listdir(path) if 'sub' in f]

  # get hcp data info
  hcp_info = hcprep.info.basics()

  print('Processing {} subjects with {} runs per task.\n'.format(len(subjects), len(hcp_info.runs)))
  # iterate subjects
  for subject_id, subject in enumerate(subjects):
    print('Processing subject: {}/{}'.format(subject_id+1, len(subjects)))
    
    # iterate tasks
    for task_id, task in enumerate(hcp_info.tasks):
        
      # iterate runs
      for run_id, run in enumerate(hcp_info.runs):

        # make sure all files present
        if not np.all([os.path.isfile(hcprep.paths.path_bids_func_mni(subject, task, run, path)),
                       os.path.isfile(hcprep.paths.path_bids_func_mask_mni(subject, task, run, path)),
                       os.path.isfile(hcprep.paths.path_bids_EV(subject, task, run, path))]):
          print('Skipping subejct {} task {} run {}, because BIDS data not fully present.'.format(subject, task, run))
        else:
          # load subject data
          subject_data = hcprep.data.load_subject_data(
            task, subject, [run], path, hcp_info.t_r)
          
          # preprocess subject data
          func_imgs, states, trs = hcprep.preprocess.preprocess_subject_data(
            subject_data=subject_data, runs=[run],
            high_pass=1./128., smoothing_fwhm=3)

          # create TFR-writer for subject/task/run
          tfr_writers = [tf.io.TFRecordWriter(hcprep.paths.path_bids_tfr(subject, task, run, path))]
          
          # write preprocessed data to TFR
          deeplight.data.io.write_func_to_tfr(tfr_writers=tfr_writers,
                                              func_data=func_imgs.get_data(),
                                              states=states,
                                              trs=trs,
                                              subject_id=subject_id,
                                              task_id=task_id,
                                              run_id=run_id,
                                              n_onehot=hcp_info.n_states_total, # total number of cognitive states across tasks (for one-hot encoding)
                                              onehot_task_idx=hcp_info.onehot_idx_per_task[task], # indices of current task in onehot encoding
                                              randomize_volumes=True)
              
          # close TFR writer
          [w.close() for w in tfr_writers]


if __name__ == "__main__":

  # run main
  main()

    
