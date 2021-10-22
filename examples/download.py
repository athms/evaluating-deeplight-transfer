#!/usr/bin/python
import os
import argparse
import numpy as np
import hcprep


def main():

  np.random.seed(13089)

  ap = argparse.ArgumentParser()
  ap.add_argument(
    "--ACCESS-KEY",
    required=True,
    help="AWS S3 access key"
  )
  ap.add_argument(
    "--SECRET-KEY",
    required=True,
    help="AWS S3 secret key"
  )
  ap.add_argument(
    "--path",
    required=False,
    default='../data/',
    help="path to store data (default: ../data/)"
  )
  ap.add_argument(
    "-n",
    required=False,
    default=3,
    help="number of subjects to download per HCP task (1-500) (default: 3)"
  )
  
  args = ap.parse_args()
  ACCESS_KEY = str(args.ACCESS_KEY)
  SECRET_KEY = str(args.SECRET_KEY)
  n = int(args.n)
  path = str(args.path)
  hcprep.paths.make_sure_path_exists(path)

  hcp_info = hcprep.info.basics()    

  print(
    'Downloading data of {} subjects to {}'.format(
      n, path
    )
  )
  for task in hcp_info.tasks:
    for subject in hcp_info.subjects[task][:n]:
      for run in hcp_info.runs:
        hcprep.download.download_subject_data(
          ACCESS_KEY=ACCESS_KEY,
          SECRET_KEY=SECRET_KEY,
          subject=subject,
          task=task,
          run=run,
          output_path=path
        )


if __name__ == '__main__':
  
  main()

    
