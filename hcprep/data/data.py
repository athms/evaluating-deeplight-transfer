#!/usr/bin/python
import os
import pandas as pd
import numpy as np

from ._utils import _load_subject_data, _generate_ev_df


def summarize_subject_EVs(task, subject, runs, path):
    """Summarize EV data of a task and subject, across runs.

    Args:
        task: String of HCP task name
        subject: Integer HCP subject ID
        runs: A sequence of the runs ["LR", "RL"] for 
            which to summarize the EV data.
        path: Path to the local HCP data in BIDS format.

    Returns:
        EV_summary: Pandas dataframe summarizing the EV data.
    """
    df_list = []
    for run in runs:
        EV_files = [f
                    for f in os.listdir(path) if
                    ('desc-EV' in f)
                    and (task in f)
                    and ('run-{}'.format(run) in f)
                    and 'summary' not in f
                    ]
        df_tmp = _generate_ev_df(path, EV_files, task, subject, run)
        if df_tmp is not None:
            df_list.append(df_tmp)
    if df_list:
        EV_summary = pd.concat(df_list)
        EV_summary = EV_summary.sort_values(by=['run', 'onset'])
        EV_summary['trial'] = np.arange(EV_summary.shape[0])
        return EV_summary.copy().reset_index(drop=True)
    else:
        return None


def load_subject_data(task, subject, runs, path, t_r=0.72):
    """Return a dict summarizing the data of a
    subject in a run of a task.

    Args:
        task: String of HCP task name
        subject: Integer HCP subject ID
        runs: A sequence of the runs ["LR", "RL"] for 
            which to summarize the EV data.
        path: Path to the local HCP data in BIDS format.
        t_r: Repetition time of HCP data

    Returns:
        Dict of data, containing one entry per run and
            a summary of the fMRI data for this run
            (incl. the paths for the fMRI data files).
    """
    return _load_subject_data(subject, task, runs, path, t_r)