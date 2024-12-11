# Segmentation of Near-Infrared Snowpack Images Using Deep Learning

Work in progress.. and so is this repository!

## Description

This repository fine-tunes SAM2 to segment near-infrared (NIR) images of snowpacks into different layers.

## Acknowledgements

This repository is based on the repository template by Francis Pelletier, which is licensed under the MIT License. Additionally, this project utilizes SAM2, which is licensed under the Apache-2.0 License. We would like to express our gratitude to the authors for making their code publicly available.
 
## Python Version

This project uses Python version 3.10 and up.

## Build Tool

This project uses `poetry` as a build tool. Using a build tool has the advantage of 
streamlining script use as well as fix path issues related to imports.

## Quick setup

For more in depth information, read the other sections below, starting at the 
[Detailed documentation section](#detailed-documentation).

### Install poetry

Skip this step if `poetry` is already installed. 

See [Installing Poetry as a Standalone section](docs/poetry_installation.md#installing-poetry-as-a-standalone-tool) 
 if working on a compute cluster.

1. Install pipx `pip install pipx` 
2. Install poetry with pipx: `pipx install poetry`

### Create project's virtual environment

1. Read the documentation on the specific cluster if required:
   * [How to create a virtual environment for the Mila cluster](docs/environment_creation_mila.md)
   * [How to create an environment for the DRAC cluster](docs/environment_creation_drac.md) 
2. Create environment : `virtualenv <PATH_TO_ENV>`
   * Or, using venv : `python3 -m venv <PATH_TO_ENV>`
3. Activate environment : `source <PATH_TO_ENV>/bin/activate`

Alternatively, if you want or need to use `conda`:

1. Read the documentation about [conda environment creation](docs/conda_environment_creation.md)
2. Create the environment : `conda env create python=<PYTHON_VERSION_NUMBER> -n <NAME_OF_ENVIRONMENT>`
3. Activate environment : `conda activate <NAME_OF_ENVIRONMENT>`

### Install

1. Install the snowpack package : `poetry install`
2. Initialize pre-commit : `pre-commit install`

## Pre-training Checkpoints

Download the SAM2 pre-training checkpoints by running:

`!wget -O sam2_hiera_small.pt "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_small.pt"`

Please store the checkpoints in `snowpack/model/model_checkpoints`.

## Settings

For the experiments presented here, we use SAM as the backbone model. We also use data augmentation, which you can find in the pipeline at `snowpack/augmentations.py`. To see a sample, you can run `tests/augmentation_test.py`.

To load the snowpack data for training, testing, or inference, we use a custom `Dataset` class defined in `snowpack/dynamic_tiled_dataset.py`, along with the previously mentioned transforms. This dataset expects `data_dir` to point to a directory containing two subdirectories: `masks` and `images`. The images and masks should have matching names, with masks ending in `_mask`, and both should be in `.tiff` format. The dataset class splits each image and its corresponding mask into multiple, possibly overlapping, tiles and presents them to the `DataLoader`. Below, we show three sample batches of training data with augmentations, using a batch size of 16. Each image is followed by its corresponding mask on the right.


