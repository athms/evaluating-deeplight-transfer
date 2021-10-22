# Evaluating deep transfer learning for whole-brain cognitive decoding

This repository contains three python packages, which are written for Python 3.6 (see [src/](src/)): 
- `hcprep`is a simple python package that allows to easily download the [Human Connectome Project](http://www.humanconnectomeproject.org) (HCP) [task-fMRI data](https://www.humanconnectome.org/study/hcp-young-adult/project-protocol/task-fmri) in a *preprocessed* format via the [Amazon Web Services (AWS) S3 storage system](https://www.humanconnectome.org/study/hcp-young-adult/article/hcp-s1200-release-now-available-amazon-web-services).
- `deeplight` is a simple python package that allows to easily apply two DeepLight architectures (2D-DeepLight and 3D-DeepLight; see below) to whole-brain fMRI data. Both architecturs were pre-trained with the fMRI data of 400 individuals in six out of the seven HCP experimental tasks (all tasks except for the working memory task, which we left out for testing purposes; [click here for details on the HCP data](https://www.sciencedirect.com/science/article/abs/pii/S1053811913005272?via%3Dihub)). 
- `modules` is a fork of the `modules` module from [interprettensor](https://github.com/VigneshSrinivasan10/interprettensor), which `deeplight` uses to build the 2D-DeepLight architecture. Note that `modules` is licensed differently from the other python packages in this repository (see [modules/LICENSE](modules/LICENSE)).


## 0. Project Organization

```bash
├── poetry.lock        <- Details of used package versions
├── pyproject.toml     <- Lists installed dependencies
├── README.md          <- This README file
├── .gitignore         <- Specifies files that .git should ignore
|
├── scrips/
|    ├── decode.py      <- An example of how to decode fMRI data with `deeplight`
|    ├── download.py    <- An example of how to download the preprocessed HCP fMRI data with `hcprep`
|    ├── interpret.py   <- An example of how to interpret fMRI data with `deeplight`
|    └── preprocess.sh  <- An example of how to preprocess fMRI data with `hcprep`
|    └── train.py       <- An example of how to train with `hcprep`
|
└── src/
|    ├── deeplight/
|    |    └──            <- `deeplight` package
|    ├── hcprep/
|    |    └──            <- 'hcprep' package
|    ├── modules/
|    |    └──            <- 'modules' package
|    └── setup.py        <- Makes 'deeplight', `hcprep`, and `modules` pip-installable (pip install -e .)  
```


## 1. Install
`deeplight` and `hcprep` are written for Python 3.6 and require a working Python environment running on your computer ([pyenv](https://github.com/pyenv/pyenv) is recommended for Python version management).

Clone and switch to this repository:
```bash
git clone https://github.com/athms/evaluating-deeplight-transfer.git
cd evaluating-deeplight-transfer
```

This project uses [Python Poetry](https://python-poetry.org/) for dependency management. To install all required dependencies, run:
```bash
poetry install
```

To also install `deeplight`, `hcprep`, and `modules` in your `poetry` environment, run:
```bash
cd src/
poetry run pip3 install -e .
```


## 2. HCPrep
`hcprep` stores data locally in the [Brain Imaging Data Structure](https://bids.neuroimaging.io) (BIDS) format.

To make the fMRI data usable for DL analyses with TensorFlow, `hcprep` can clean the downloaded fMRI data and store these in the [TFRecords data format](https://www.tensorflow.org/tutorials/load_data/tfrecord).  

**Getting data access:**
To download the HCP task-fMRI data, you will also need AWS access to the HCP public data directory. A detailed instruction can be found [here](https://wiki.humanconnectome.org/display/PublicData/How+To+Connect+to+Connectome+Data+via+AWS). Make sure to safely store the `ACCESS_KEY` and `SECRET_KEY`; they are required to access the data via the AWS S3 storage system. 

**AWS configuration:**
Setup your local AWS client (as described [here](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html)) and add the following profile to '~/.aws/config'

```bash
[profile hcp]
region=eu-central-1
```
Choose the region based on your [location](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Concepts.RegionsAndAvailabilityZones.html).

**TFR data storage:**
`hcprep` stores the preprocessed fMRI data locally in [TFRecords format](https://www.tensorflow.org/tutorials/load_data/tfrecord), with one entry for each input fMRI volume of the data, each containing the following features:
- `volume`: the flattened voxel activations with shape 91x109x91 (flattened over the X (91), Y (109), and Z (91) dimensions)
- `task_id`, `subject_id`, `run_id`: numerical id of task, subject, and run
- `tr`: TR of the volume in the underlying experimental task
- `state`: numerical label of the cognive state associated with the volume within its task (e.g., [0,1,2,3] for the four cognitive states of the working memory task)
- `onehot`: one-hot encoding of the state across all experimental tasks that are used for training (e.g., there are 20 cognitive tasks across the seven experimental tasks of the HCP; the four cognitive states of the working memory task could thus be mapped to the last four positions of the one-hot encoding, with indices [16: 0, 17: 1, 18: 2, 19: 3])

Note that `hcprep` also provides basic descriptive information about the HCP task-fMRI data in `info.basics`:

```python
hcp_info = hcprep.info.basics()
```

`basics` contains the following information:
- `tasks`: names of all HCP experimental tasks ('EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM')
- `subjects`: dictionary containing 1000 subject IDs for each task
- `runs`: run IDs ('LR', 'RL')
- `t_r`: repetition time of the fMRI data in seconds (0.72)
- `states_per_task`: dictionary containing the label of each cognitive state of each task
- `onehot_idx_per_task`: index that is used to assign cognitive states of each task to the `onehot`encoding of the TFR-files (see `onehot` above)

For further details on the experimental tasks and their cognitive states, [click here](https://www.sciencedirect.com/science/article/abs/pii/S1053811913005272?via%3Dihub).


## 3. DeepLight
`deeplight` implements two DeepLight architectures ("2D" and "3D"), which can be accessed as `deeplight.two` (2D) and `deeplight.three` (3D).

Importantly, both DeepLight architectures operate on the level of individual whole-brain fMRI volumes (e.g., individual TRs).

**2D-DeepLight:** A whole-brain fMRI volume is first sliced into a sequence of axial 2D-images (from bottom-to-top). These images are passed to a DL model, consisting of a 2D-convolutional feature extractor as well as an LSTM unit and output layer. First, the 2D-convolutional feature extractor reduces the dimensionality of the axial brain images through a sequence of 2D-convolution layers. The resulting sequence of higher-level slice representations is then fed to a bi-directional LSTM, modeling the spatial dependencies of brain activity within and across brain slices. Lastly, 2D-DeepLight outputs a decoding decision about the cognitive state underlying the fMRI volume, through a softmax output layer with one output unit per cognitive state in the data.

**3D-DeepLight:** The whole-brain fMRI volume is passed to a 3D-convolutional feature extractor, consisting of a sequence of twelve 3D-convolution layers. The 3D-convolutional feature extractor directly projects the fMRI volume into a higher-level, but lower dimensional, representation of whole-brain activity, without the need of an LSTM. To make a decoding decision, 3D-DeepLight utilizes an output layer that is composed of a 1D- convolution and global average pooling layer as well as a softmax activation function. The 1D-convolution layer maps the higher-level representation of whole-brain activity of the 3D-convolutional feature extractor to one representation for each cognitive state in the data, while the global average pooling layer and softmax function then reduce these to a decoding decision.

To interpret the decoding decisions of the two DeepLight architectures, relating their decoding decisions to the fMRI data, `deeplight` makes use of the [LRP technique](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140). The LRP technique decomposes individual decoding decisions of a DL model into the contributions of the individual input features (here individual voxel activities) to these decisions. 

Both deeplight architectures implement basic `fit`, `decode`, and `interpret` functions, next to other functionalities. For details on how to {train, decode, interpret} with `deeplight`, see the scripts in [scripts/](scripts/).

For further methdological details regarding the two DeepLight architectures, see the upcoming preprint.

**Note** that we currently recommend to run any applications of `interpret` with 2D-DeepLight on CPUs instead of a GPU, due to a high memory demand (assuming that your available CPU memory is larger than your available GPU memory). This switch can be made by setting the environment variable `export CUDA_VISIBLE_DEVICES=""`. We are currently working on reducing the overall memory demand of `interpret` with 2D-DeepLight and will push a code update soon. 


## 4. Example scripts
You can find a set of example python scripts in [scripts/](scripts/), which illustrate how to download and preprocess task-fMRI data from the Human Connectome Project with `hcprep` and how to {train on, decode, interpret} fMRI data with the two DeepLight architectures implemented in `deeplight`.

You can run individual scripts in your `poetry`environment with: 
```bash
cd scripts/
poetry run python <SCRIPT NAME>
```

