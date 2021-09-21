#!/usr/bin/python
import numpy as np
import os
import hcprep


class basics:
    """All basic descriptive information
    about the Human Connectome Project
    task-fMRI data.

    Attributes:
        tasks: task names
        states_per_task: Dictionary of state names 
            per task
        n_states_per_task: List with number of states
            per task
        runs: List or run names
        subjects: Dictionary of subject IDs for each task
        t_r: Repetition time of fMRI data (in seconds)
    """

    def __init__(self):
        self.tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE',
                      'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
        self.states_per_task = dict(EMOTION=['fear', 'neut'],
                                    GAMBLING=['loss', 'neut', 'win'],
                                    LANGUAGE=['math', 'story'],
                                    MOTOR=['lf', 'lh', 'rf', 'rh', 't'],
                                    RELATIONAL=['match', 'relation'],
                                    SOCIAL=['mental', 'other'],
                                    WM=['body', 'faces', 'places', 'tools'])
        self.n_states_per_task = {task: len(self.states_per_task[task])
                                  for task in self.tasks}
        self.n_states_total = 20
        self.onehot_idx_per_task = dict(EMOTION=np.array([0, 1]),
                                        GAMBLING=np.array([2, 3, 4]),
                                        LANGUAGE=np.array([5, 6]),
                                        MOTOR=np.array([7, 8, 9, 10, 11]),
                                        RELATIONAL=np.array([12, 13]),
                                        SOCIAL=np.array([14, 15]),
                                        WM=np.array([16, 17, 18, 19]))
        self.runs = ['LR', 'RL']
        self.subjects = {task: np.load(os.path.join(os.path.dirname(hcprep.__file__),
            'info', 'subject_ids', 'task-{}_subjects.npy'.format(task))) for task in self.tasks}            
        self.t_r = 0.72
