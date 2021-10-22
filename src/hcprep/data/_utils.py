#!/usr/bin/python
import os
import pandas as pd
import numpy as np
from nilearn.image import load_img

from .. import paths


def _generate_ev_df(
    path,
    ev_filenames,
    task,
    subject,
    run):

    if task not in [
        'EMOTION',
        'SOCIAL',
        'GAMBLING',
        'LANGUAGE',
        'RELATIONAL',
        'MOTOR',
        'WM']:
        raise NameError('Invalid task type.')

    df_list = []
    for f in ev_filenames:

        if task is 'GAMBLING' and (
            'win.txt' in f
            or 'loss.txt' in f
            or 'neut.txt' in f):
            continue

        event_type = f.split('.')[0].split('desc-EV_')[-1]
        if task is 'SOCIAL':
            if '_resp' in f:
                event_type = event_type.split('_')[-2]
            else:
                continue
        elif task is 'WM':
            event_type = event_type.split('_')[-1]
        else:
            event_type = event_type.split('_')[0]
        
        if 'cue' in event_type:
            continue

        if task is 'WM' and event_type not in ['body', 'faces', 'places', 'tools']:
            continue
        
        try:
            ev_mat = pd.read_csv(path+f, sep='\t', header=None).values
        except:
            print('/!\ Skipping {} because it is empty.'.format(path+f))
            continue

        df_tmp = pd.DataFrame(
            {'subject': subject,
             'task': task,
             'run': run,
             'event_type': event_type,
             'onset': ev_mat[:, 0],
             'duration': ev_mat[:, 1],
             'end': ev_mat[:, 0] + ev_mat[:, 1]
            }
        )
        df_list.append(df_tmp)

    if df_list:
      return pd.concat(df_list)
    else:
      return None


def _init_datadict(
    subject,
    task,
    runs,
    path,
    t_r):

    f = {'anat': None, 'anat_mni': None, 'tr': t_r, 'runs': runs}
    for ri, run in enumerate(runs):
        n_tr = np.int(
            load_img(
                paths.path_bids_func_mni(subject, task, run, path)
            ).shape[-1]
        )
        f[run] = {
            'func': None,
            'func_mni': paths.path_bids_func_mni(subject, task, run, path),
            'func_mask_mni': paths.path_bids_func_mask_mni(subject, task, run, path),
            'n_volumes': n_tr,
            'trial': np.zeros(n_tr) * np.nan,
            'n_trial_volumes': np.zeros(n_tr) * np.nan,
            'rel_onset': np.zeros(n_tr) * np.nan,
            'trial_type': np.zeros(n_tr) * np.nan,
            'n_valid_volumes': 0,
            'n_trials': 0,
            'onset': np.arange(n_tr) * t_r,
            'end': (np.arange(n_tr) * t_r) + t_r
            }
    return f


def _add_markers_to_datadict(
    f,
    EV,
    n_volumes_discard_trial_onset=1,
    n_volumes_add_trial_end=1):

    t_r = f['tr']
    event_types = EV['event_type'].values
    unique_event_types = np.sort(np.unique(event_types))
    numerical_event_types = np.arange(unique_event_types.size)
    mapping = {}
    f['event_type_mapping'] = {}
    for num_type in numerical_event_types:
        f['event_type_mapping'][num_type] = unique_event_types[num_type]
    f['event_type_mapping'][numerical_event_types.max()+1] = 'fixation'
    tmp = np.ones_like(event_types) * np.nan
    for event_type in unique_event_types:
        tmp[np.where(event_types == event_type)[0]] = numerical_event_types[
            np.where(unique_event_types == event_type)[0]]
    EV['event_num'] = np.array(tmp)

    runs = EV['run'].unique()
    for run in runs:
        run_data = EV[EV['run'] == run].copy()
        trials = np.sort(run_data['trial'].values)
        for trial in trials:
            trial_data = run_data[run_data['trial'] == trial].copy()
            trial_type = trial_data['event_num'].values[0]
            run = trial_data['run'].values[0]
            trial_onset = trial_data['onset'].values[0]
            trial_end = trial_data['end'].values[0]
            volume_idx = np.where(
              (f[run]['onset'] >= (trial_onset + (n_volumes_discard_trial_onset*t_r))) &
              (f[run]['onset'] <= (trial_end+t_r + (n_volumes_add_trial_end*t_r))))[0]
            f[run]['trial'][volume_idx] = trial
            f[run]['n_trial_volumes'][volume_idx] = volume_idx.size
            f[run]['rel_onset'][volume_idx] = f[run]['onset'][volume_idx] - trial_onset
            f[run]['trial_type'][volume_idx] = trial_type
            f[run]['n_valid_volumes'] += len(volume_idx)
            f[run]['n_trials'] += 1

    return f


def _load_subject_data(
    subject,
    task,
    runs,
    path,
    t_r):
    
    f = _init_datadict(subject, task, runs, path, t_r)
    EV_list = []
    for run in runs:
        EV_run = pd.read_csv(paths.path_bids_EV(subject, task, run, path))
        EV_list.append(EV_run)
    return _add_markers_to_datadict(f, pd.concat(EV_list))
