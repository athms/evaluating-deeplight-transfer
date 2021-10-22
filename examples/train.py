#!/usr/bin/python
import os
import argparse
import numpy as np
import deeplight
import hcprep


def main():

  np.random.seed(8170)

  ap = argparse.ArgumentParser()
  ap.add_argument(
    "--data",
    required=False,
    default='../data/',
    help="path to TFR data files (default: ../data/)"
  )
  ap.add_argument(
    "--architecture",
    required=False,
    default='3D',
    help="DeepLight architecture (2D or 3D) (default: 3D)"
  )
  ap.add_argument(
    "--pretrained",
    required=False,
    default=1,
    help="use pre-trained model? (1: True or 0: False) (default: 1)"
  )
  ap.add_argument(
    "--training-tasks",
    required=False,
    default=['WM'],
    nargs="+", 
    help="in which HCP tasks to train"\
         "(EMOTION, GAMBLING, SOCIAL, LANGUAGE, MOTOR,"\
         "RELATIONAL, LANGUAGE, WM)? (default: ['WM'])"
  )
  ap.add_argument(
    "--learning-rate",
    required=False,
    default=1e-3,
    help="learning rate for gradient descent with ADAM (default: 1e-3)"
  )
  ap.add_argument(
    "--batch-size",
    required=False,
    default=32,
    help="batch size for training (default: 32)"
  )
  ap.add_argument(
    "--epochs",
    required=False,
    default=50,
    help="how many training epochs? (default: 50)"
  )
  ap.add_argument(
    "--training-steps",
    required=False,
    default=100,
    help="how many steps per training epoch? (default: 100)"
  )
  ap.add_argument(
    "--n-workers",
    required=False,
    default=4,
    help="how many parallel workers to use to load data? (default: 4)"
  )
  ap.add_argument(
    "--validation-steps",
    required=False,
    default=100,
    help="how many validation steps after each training epoch? (default: 100)"
  )
  ap.add_argument(
    "--out",
    required=False,
    default=None,
    help="path where epoch models are saved during training"
  )
  ap.add_argument(
    "--verbose",
    required=False,
    default=1,
    help="comment training with prints in terminal"\
      "(0: no or 1: yes) (default: 1)"
  )
  
  args = ap.parse_args()
  architecture = str(args.architecture)
  pretrained = bool(int(args.pretrained))
  training_tasks = list(args.training_tasks)
  epochs = int(args.epochs)
  learning_rate = float(args.learning_rate)
  batch_size = int(args.batch_size)
  training_steps = int(args.training_steps)
  validation_steps = int(args.validation_steps)
  n_workers = int(args.n_workers)
  verbose = bool(int(args.verbose))
  data_path = str(args.data)
  if args.out is None:
    out_path = '../results/models/DeepLight/{}/pretrained-{}/'.format(
      architecture, pretrained)
    print(
      '"out" not defined. Defaulting to: {}'.format(out_path)
    )    
  else:
    out_path = str(args.out)

  hcp_info = hcprep.info.basics()

  # check if training tasks correctly specified
  training_tasks = np.unique(training_tasks)
  for task in training_tasks:
    if task not in hcp_info.tasks:
      raise ValueError(
        '{} not in {}'.format(task, hcp_info.tasks)
      )

  os.makedirs(out_path, exist_ok=True)

  print(
    "\nTraining settings:"
  )
  print(
    "\tHCP tasks: {}".format(training_tasks)
  )
  print(
    "\tLearning rate: {}".format(learning_rate)
  )
  print(
    '\tBatch size: {}'.format(batch_size)
  )
  print(
    "\tEpochs: {}".format(epochs)
  )
  print(
    "\tTraining steps per epoch: {}".format(training_steps)
  )
  print(
    "\tValidation steps per epoch: {}".format(validation_steps)
  )
  print(
    '\tSaving results to: {}'.format(out_path)
  )
  
  # how many cognitive states are there in the training data?
  n_states_training = np.sum(
    [
      hcp_info.n_states_per_task[task]
      for task in training_tasks
    ]
  )
  
  subjects = np.sort(
    np.unique(
      [
        int(p.split('sub-')[1])
        for p in os.listdir(data_path)
        if p.startswith('sub-')
      ]
    )
  )

  # assign subjects to training / validation data (2 / 1 split)
  subjects_training = np.random.choice(
    subjects,
    np.int(subjects.size*2/3.),
    replace=False
  )
  subjects_validation = np.array(
    [
      s for s in subjects
      if s not in subjects_training
    ]
  )
  print(
    '\nRandom subject split: '
  )
  print(
    '\t{} training subjects, {} validation subjects'.format(
    len(subjects_training), len(subjects_validation)
    )
  )
  print(
    '\tSaving subject assignment to: {}'.format(
    out_path+'subject_split.npy'
    )
  )
  np.save(
    out_path+'subject_split.npy',
    {
     'training': subjects_training,
     'validation': subjects_validation
    }
  )

  # get paths to training TFR files
  train_files = []
  for task in training_tasks:
    for subject in subjects_training:
      for run in hcp_info.runs:
        filepath = hcprep.paths.path_bids_tfr(
          subject=subject,
          task=task,
          run=run,
          path=data_path
        )
        if os.path.isfile(filepath):
          train_files.append(filepath)
  
  # get paths to validation TFR files
  validation_files = []
  for task in training_tasks:
    for subject in subjects_validation:
      for run in hcp_info.runs:
        filepath = hcprep.paths.path_bids_tfr(
          subject=subject,
          task=task,
          run=run,
          path=data_path
        )
        if os.path.isfile(filepath):
          validation_files.append(filepath)

  if architecture == '3D':
    deeplight_variant = deeplight.three.model(
      n_states=n_states_training,
      batch_size=batch_size,
      pretrained=pretrained,
      verbose=verbose
    )
  elif architecture == '2D':
    deeplight_variant = deeplight.two.model(
      n_states=n_states_training,
      batch_size=batch_size,
      pretrained=pretrained,
      verbose=verbose
    )
  else:
    raise ValueError(
      'Invalid value for DeepLight architecture. Must be 2D or 3D.'
    )

  fit_history = deeplight_variant.fit(
    train_files=train_files,
    validation_files=validation_files,
    n_onehot=20, # there are 20 values in the onehot encoding of the HCP TFR files
    # (one for each of the 20 cognitive states of the HCP data)
    # of these 20 onehot values we only care about those 
    # that belong to the current training task:
    onehot_idx=np.sort(
      np.concatenate(
        [
          hcp_info.onehot_idx_per_task[t]
          for t in training_tasks
        ]
      )
    ), 
    learning_rate=learning_rate,
    epochs=epochs,
    training_steps=training_steps,
    validation_steps=validation_steps,
    output_path=out_path,
    shuffle_buffer_size=500,
    n_workers=n_workers
  )
  
  if verbose:
    print('\n\n..done.')


if __name__ == '__main__':

    main()