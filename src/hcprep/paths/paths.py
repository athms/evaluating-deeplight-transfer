#!/usr/bin/python
import os


def make_sure_path_exists(path):
    """Make sure a path exists.
    If it does not exist, create it.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def path_bids_EV(
    subject,
    task,
    run,
    path):
    """Return the path to the local EV-summary
    file of a subject in a task run.

    Args:
        subject: Integer ID of HCP subject
        task: String ID of HCP task.
        runs: String ID of HCP run (one of ["LR", "RL"])
        path: Path to local BIDS directory.
    """
    return os.path.join(
        path,
        "sub-{}".format(subject),
        "func",
        "sub-{}_task-{}_run-{}_desc-EV_summary.csv".format(subject, task, run)
    )


def path_bids_func_mni(
    subject,
    task,
    run,
    path):
    """Return the path to the local task-fMRI data
    of a subject in a task run.

    Args:
        subject: Integer ID of HCP subject
        task: String ID of HCP task.
        runs: String ID of HCP run (one of ["LR", "RL"])
        path: Path to local BIDS directory.
    """
    return os.path.join(
        path,
        "sub-{}".format(subject),
        "func",
        "sub-{}_task-{}_run-{}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz".format(subject, task, run)
    )


def path_bids_anat_mni(
    subject,
    path):
    """Return the path to the local anatomical scan
    of a subject.

    Args:
        subject: Integer ID of HCP subject
        task: String ID of HCP task.
        path: Path to local BIDS directory.
    """
    return os.path.join(
        path,
        "sub-{}".format(subject),
        "anat", 
        "sub-{}_space-MNI152NLin6Asym_res-2_desc-preproc_T1w.nii.gz".format(subject)
    )


def path_bids_func_mask_mni(
    subject,
    task,
    run,
    path):
    """Return the path to the local task-fMRI brainmask
    of a subject in a task run.

    Args:
        subject: Integer ID of HCP subject
        task: String ID of HCP task.
        runs: String ID of HCP run (one of ["LR", "RL"])
        path: Path to local BIDS directory.
    """
    return os.path.join(
        path,
        "sub-{}".format(subject),
        "func",
        "sub-{}_task-{}_run-{}_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask.nii.gz".format(subject, task, run)
    )



def path_bids_tfr(
    subject,
    task,
    run,
    path):
    """Return the path to the tfr data
    of a subject in a task run.

    Args:
        subject: Integer ID of HCP subject
        task: String ID of HCP task.
        runs: String ID of HCP run (one of ["LR", "RL"])
        path: Path to local BIDS directory.
    """
    return os.path.join(
        path,
        "sub-{}".format(subject),
        "func", "sub-{}_task-{}_run-{}_space-MNI152NLin6Asym_res-2_desc-tfr.tfrecords".format(subject, task, run)
    )