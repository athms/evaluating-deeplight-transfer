#!/usr/bin/python
import sys
import os
import numpy as np
import boto3
import botocore

from ._utils import _return_hcp_EV_file_ids, _check_key_exists
from ..data import summarize_subject_EVs
from .. import paths


def connect_to_hcp_bucket(
    ACCESS_KEY,
    SECRET_KEY):
    """Connect to HCP AWS S3 bucket with boto3

    Args:
        ACCESS_KEY, SECRET_KEY: access and secret
            keys necessary to access HCP AWS S3 storage.

    Returns:
        Boto3 bucket
    """
    boto3.setup_default_session(
        profile_name='hcp',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name='us-east-1'
    )
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('hcp-openaccess')
    return bucket, s3


def retrieve_subject_ids(
    ACCESS_KEY,
    SECRET_KEY,
    task,
    runs=['LR', 'RL'],
    n=1000):
    """Retrieve IDs of HCP subjects in a task from 
    the AWS S3 servers.

    Args:
        ACCESS_KEY, SECRET_KEY: access and secret
            keys necessary to access HCP AWS S3 storage.
        task: String ID of HCP task.
        runs: A sequence of the HCP run IDs ["LR",, "RL"]
        n: Number of subject IDs to retrieve.

    Returns:
        Subject_ids: Sequence of retrieved subject IDs.
    """
    bucket, s3 = connect_to_hcp_bucket(
        ACCESS_KEY=ACCESS_KEY,
        SECRET_KEY=SECRET_KEY
    )
    subject_ids = []
    sample_key = (
        '/MNINonLinear/' +
        'Results/' +
        'tfMRI_{}_RL/'.format(task) +
        'tfMRI_{}_RL.nii.gz'.format(task)
    )
    for o in bucket.objects.filter(Prefix='HCP'):
        if (sample_key in o.key):
            subject = o.key.split('/')[1]
            if (check_subject_data_present(bucket, subject, task, runs)
                and subject not in subject_ids):
                    subject_ids.append(subject)
        if len(subject_ids) >= n:
            break
    return subject_ids


def check_subject_data_present(
    bucket,
    subject,
    task,
    runs):
    """Check if a subject's task-fMRI data is present
    in the AWS S3 bucket.

    Args:
        bucket: Boto3 bucket created with
            hcprep.download.connect_to_hcp_bucket
        subject: Integer ID of HCP subject
        task: String ID of HCP task.
        runs: A sequence of the HCP run IDs ["LR", "RL"]

    Returns:
        Bool indicating whether task-fMRI data present.
    """
    anat_key = ('T1w.nii.gz')
    checks = []
    for run in runs:
        prefix = 'HCP/{}/MNINonLinear/Results/tfMRI_{}_{}/'.format(subject, task, run)

        tfMRI_key = (prefix+'tfMRI_{}_{}.nii.gz'.format(task, run))
        checks.append(_check_key_exists(tfMRI_key, bucket, prefix))

        tfMRI_mask_key = (prefix + 'brainmask_fs.2.nii.gz')
        checks.append(_check_key_exists(tfMRI_mask_key, bucket, prefix))

        anat_prefix = 'HCP/{}/MNINonLinear/'.format(subject)
        checks.append(_check_key_exists(anat_key, bucket, anat_prefix))

        for EV_file in _return_hcp_EV_file_ids(task):
            EV_key = (prefix+'EVs/'+EV_file)
            checks.append(_check_key_exists(EV_key, bucket, prefix))

    return np.sum(checks) == len(checks)
    

def download_file_from_bucket(
    bucket,
    bucket_id,
    output_file):
    """Download the a file (as specified by bucket_id)
    from its bucket.

    Args:
        bucket: bucket to download from
        bucket_id: bucket ID of file to download
        output_file: filepath where file is locally stored
    """
    if not os.path.isfile(output_file):
        try:
            print('downloading {}  to  {}'.format(bucket_id, output_file))
            bucket.download_file(bucket_id, output_file)
        except botocore.exceptions.ClientError as e:
            # If a client error is thrown, then check that it was a 404 error.
            # If it was a 404 error, then the bucket id does not exist.
            error_code = e.response['Error']['Code']
            if error_code == '404':
                print('! {} does not exist.'.format(bucket_id))


def download_subject_data(
    ACCESS_KEY,
    SECRET_KEY,
    subject,
    task,
    run,
    output_path):
    """Download the task-fMRI data of a HCP subject
    in a task run and write it to a local directory 
    in the Brain Imaging Data Structure (BIDS) format.

    Args:
        ACCESS_KEY, SECRET_KEY: access and secret
            keys necessary to access HCP AWS S3 storage.
        subject: Integer ID of HCP subject
        task: String ID of HCP task.
        runs: String ID of HCP run (one of ["LR", "RL"])
        output_path: Local path to which data is written.
    """
    path_sub = output_path+'sub-{}/'.format(subject)
    path_anat = path_sub+'anat/'
    path_func = path_sub+'func/'
    paths.make_sure_path_exists(path_sub)
    paths.make_sure_path_exists(path_anat)
    paths.make_sure_path_exists(path_func)

    bucket, s3 = connect_to_hcp_bucket(
        ACCESS_KEY=ACCESS_KEY, SECRET_KEY=SECRET_KEY)

    bucket_id = (
        'HCP/{}/'.format(subject) +
        'MNINonLinear/' +
        'Results/' +
        'tfMRI_{}_{}/'.format(task, run) +
        'tfMRI_{}_{}.nii.gz'.format(task, run)
    )
    output_file = paths.path_bids_func_mni(subject, task, run, output_path)
    download_file_from_bucket(bucket, bucket_id, output_file)

    bucket_id = (
        'HCP/{}/'.format(subject) +
        'MNINonLinear/' +
        'Results/' +
        'tfMRI_{}_{}/'.format(task, run) +
        'brainmask_fs.2.nii.gz'
    )
    output_file = paths.path_bids_func_mask_mni(
        subject, task, run, output_path)
    download_file_from_bucket(bucket, bucket_id, output_file)

    bucket_id = (
        'HCP/{}/'.format(subject) +
        'MNINonLinear/' +
        'T1w.nii.gz'.format(task, run)
    )
    output_file = paths.path_bids_anat_mni(subject, task, run, output_path)
    download_file_from_bucket(bucket, bucket_id, output_file)

    identifier = (
        'HCP/{}/'.format(subject) +
        'MNINonLinear/' +
        'Results/' +
        'tfMRI_{}_{}/'.format(task, run) +
        'EVs/'
    )
    for EV_file in _return_hcp_EV_file_ids(task):
        bucket_id = identifier+EV_file
        output_file = path_func+'sub-{}_task-{}_run-{}_desc-EV_{}'.format(
            subject, task, run, EV_file)
        download_file_from_bucket(bucket, bucket_id, output_file)

    output_file = paths.path_bids_EV(subject, task, run, output_path)
    if not os.path.isfile(output_file):
        print('creating EV summary: {}'.format(output_file))
        EV_summary = summarize_subject_EVs(task, subject, [run], path_func)
        if EV_summary is not None:
            EV_summary.to_csv(output_file, index=False)
        else:
            print('! no EV files found for sub-{}, task-{}, run-{}'.format(
                subject, task, run))
