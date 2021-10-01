#!/usr/bin/python
import os


def _return_hcp_EV_file_ids(task):
    if task == 'EMOTION':
        file_types = [
            'fear.txt',
            'neut.txt'
        ]

    elif task == 'GAMBLING':
        file_types = [
            'win.txt',
            'loss.txt',
            'win_event.txt',
            'loss_event.txt',
            'neut_event.txt'
        ]

    elif task == 'LANGUAGE':
        file_types = [
            'story.txt',
            'math.txt'
        ]

    elif task == 'MOTOR':
        file_types = [
            'cue.txt',
            'lf.txt',
            'rf.txt',
            'lh.txt',
            'rh.txt',
            't.txt'
        ]

    elif task == 'RELATIONAL':
        file_types = [
            'relation.txt',
            'match.txt'
        ]

    elif task == 'SOCIAL':
        file_types = [
            'mental.txt',
            'rnd.txt',
            'mental_resp.txt',
            'other_resp.txt'
        ]

    elif task == 'WM':
        file_types = [
            '0bk_body.txt',
            '0bk_faces.txt',
            '0bk_places.txt',
            '0bk_tools.txt',
            '2bk_body.txt',
            '2bk_faces.txt',
            '2bk_places.txt',
            '2bk_tools.txt',
            '0bk_cor.txt',
            '0bk_err.txt',
            '0bk_nlr.txt',
            '2bk_cor.txt',
            '2bk_err.txt',
            '2bk_nlr.txt',
            'all_bk_cor.txt',
            'all_bk_err.txt'
        ]

    else:
        file_types = None
        raise NameError('Invalid task type.')

    return file_types


def _check_key_exists(
    key,
    bucket,
    prefix):
    keys = [o.key for o in bucket.objects.filter(Prefix=prefix)]
    return key in keys