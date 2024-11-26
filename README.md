# README

## Description

Repository template that focuses on simplicity and ease of use. 

This is ideal for quick projects or code publication.

The purpose of this template is to help with code quality, structure and 
reproducibility.

This template is not intended to be used as is for libraries or applications that will
be maintained over time, as several things are missing from it, like change logs, 
advanced tools and coding standards (though it can be expanded for such uses).

This template creates a python package, contained in [src/](src/), that will
contain your different modules.

For more information about python packages and modules, 
[Python Modules and Packages – An Introduction](https://realpython.com/python-modules-packages/).

## Initialization

Please follow these steps:

1. Setup the repository:
   * Automatic way - On the [template's github page](https://github.com/RolnickLab/lab-basic-template), 
     create a new repository by using the `Use this template` button, near the top right corner. 
     Do not include all branches.
     * If you already have existing code, transfer it either in [src/](src/) or [scripts/](scripts/), 
       depending on it's nature
       * Modules (python code that is meant to be _imported_ in other python files) should go into the 
         [src folder](src/README.md)
       * Python scripts that are meant to be executed via the command line
         should go into the [scripts folder](scripts/README.md)

   * It can also be done manually (though longer and more error prone):
      1. Clone or download the `lab-basic-template` repository (this repository)
      2. Either start a new GitHub repository, or select an existing one (the target repository)
      3. Copy the files and folders of the `lab-basic-template` repository into your target repository.
         * Do not copy the `.git` folder from the `lab-basic-template`.
         * Move your existing code
           * Modules (python code that is meant to be _imported_ in other python files) should go into the 
             [src folder](src/README.md)
           * Python scripts that are meant to be executed via the command line
             should go into the [scripts folder](scripts/README.md)


2. Rename the python package (optional step) - This will allow you to use `from <package_name> import ...` 
   instead of `from src import ...`  :
   1. Rename [src folder](src) to your package name
      * Make sure the name is in `snake_case`, like other python modules and packages.
      * Bad examples : `my-package`, `MyPackage`, `My Package`
      * Good example : `my_package`
   2. Set the package name on line #2 of the [pyproject.toml](pyproject.toml) file by replacing `src` with the 
      same package name used above.
3. Write your name on line #5 and write a short description on line #4 in [pyproject.toml](pyproject.toml)
4. Follow the rest of the instructions in this README
5. Remove this section (_Initialization_) from the README of your target repository and modify it's title 
   and description

**Important note**
If you are planning to use this for a new project and expect to use the DRAC cluster 
as well as other clusters/locations, it is recommended to first set up your environment 
on DRAC, as the versions of Python libraries are often a bit behind compared to the Mila 
cluster.

This will make your project more portable and will prevent many dependency management 
problems while working across different clusters.

Installing this module for the first time (see [Installation](#installation)) 
will create the `poetry.lock` file, which will set the different library versions used 
by the project, and therefore help with reproducibility and reduce the classic but 
annoying "but it works on my machine" situation.

However, this `poetry.lock` file can be problematic when using locally compiled python 
wheels, [like is recommended in their documentation](#drac-specific-install-directives).

If working on multiple different clusters, it might be better to add the `poetry.lock` 
file to your `.gitignore`, and manage your dependencies with either explicit versions or 
with [Tilde requirements](https://python-poetry.org/docs/dependency-specification/#tilde-requirements).
 
## Python Version

This project uses Python version 3.10 and up.

## Build Tool

This project uses `poetry` as a build tool. Using a build tool has the advantage of 
streamlining script use as well as fix path issues related to imports.

## Quick setup

This is a short step by step, no nonsense way of setting things up and start working 
right away. 

For more in depth information, read the other sections below, starting at the 
[Detailed documentation section](#detailed-documentation).

**Reminder:** When working on the clusters, you will always need to load the 
appropriate module before you can activate your environment
* For python virtual environments: `module load python/<PYTHON_VERSION>`
* For conda environments : `module load miniconda/3`

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

1. Install your package : `poetry install`
2. Initialize pre-commit : `pre-commit install`

### Development

1. [Add required dependencies](./CONTRIBUTING.md#adding-dependencies)
2. Create some new modules in the [src](src/) folder!

## Detailed documentation

### Environment Management

This section and those following go into more details for the different setup steps.

Your project will need a virtual environment for your dependencies.

* [How to create a virtual environment for the Mila cluster](docs/environment_creation_mila.md)
* [How to create an environment for the DRAC cluster](docs/environment_creation_drac.md)
* [How to create a Conda environment](docs/conda_environment_creation.md)
* [Migrating to DRAC from another environment](docs/migrating_to_drac.md)

There are different ways of managing your python version in these environments. On the 
clusters, you have access to different python modules, and through `conda` you have access 
to practically all the python versions that exist. 

However, on your own system, if you do not wish to use `conda`, you will have to either 
manually install different versions of python manually for them to be usable by `poetry` 
or use a tool like [pyenv](https://github.com/pyenv/pyenv).

Do note that `conda` is not available on the DRAC cluster, and there are some extra steps
to use `conda` on the Mila cluster compared to a workstation.

Once you know in which environment you will be working, we can proceed to install `poetry`.

`poetry` can be installed a number of ways, depending on the environment choice, and if
you are working on your local machine vs a remote cluster. See the following 
documentation to help you determine what is best for you.

* [How to install poetry](docs/poetry_installation.md)


### Installation

Once the virtual environment is built and `poetry` is installed, follow these steps to 
install the package's dependencies:

1. Make sure your virtual environment is active
2. Install the package and its dependencies using the following command:
    * `poetry install`
    * Alternatively, you can also install using `pip install -e .`, which will install 
      your package, [configured scripts](https://python-poetry.org/docs/pyproject#scripts) 
      and dependencies, but without creating a `poetry.lock` file.

#### DRAC Specific Install Directives

This is in accordance with the official [DRAC documentation](https://docs.alliancecan.ca/wiki/Python#Creating_and_using_a_virtual_environment), 
adapted for use with `poetry`.

It is common practice, though not enforced, to re-create your Python environment inside 
your job. Since most nodes on the DRAC cluster do not have access to the internet, the 
dependencies are therefore installed from pre-built wheels. If this does not apply to 
you, or if you work on Cedar, which nodes do have access to the internet, you can 
disregard the following.

To install the dependencies from these wheels, use the following steps:

1. Create your environment like specified in [environment_creation_drac.md](docs/environment_creation_drac.md)
2. Instead of using `poetry install`, use `pip install -e . --no-index`
   * This will install the package, [configured scripts](https://python-poetry.org/docs/pyproject#scripts) 
     as well as the dependencies. However, `pip` can only install main dependencies and 
     will not be able to install `poetry` defined groups

### Development

If you want to contribute to this repository, the development dependencies will also need to added.

1. Install `pre-commit` and other dev dependencies using the following command:
   * `poetry install --with dev`
     * `pre-commit` is used for code quality and code analysis
   * Configure `pre-commit` by running the following command: `pre-commit install`
2. Optional Pylint check
   * While not enforced by the pre-commit tool, running Pylint on your code can help
     with code quality, readability and even catch errors or bad coding practices.
   * To run this tool : `pylint src/ scripts/ tests/`
   * For more information, see the [Pylint library](https://pylint.readthedocs.io/en/stable/)
3. Optional Cyclomatic Complexity check (McCabe)
   * While not enforced by the pre-commit tool, running a complexity check on your code can help
     with code quality, readability and even catch errors or bad coding practices.
   * To run this tool : `flake8 --max-complexity 7 src/ tests/ scripts/`
   * For more information, see [McCabe Checker](https://github.com/PyCQA/mccabe)
4. Python library dependencies
   * To keep things simple, it is recommended to store all new dependencies as main 
     dependencies, unless you are already familiar with dependency management. 
5. Read and follow the [Contributing guidelines](CONTRIBUTING.md)

## Settings

For the experiments presented here, we use SAM as the backbone model. We also use data augmentation, which you can find in the pipeline at `snowpack/augmentations.py`. To see a sample, you can run `tests/augmentation_test.py`. The result of such a test is given on a sample data seen in the figure below:

![Augmentation Test Results](assets/train_test_transforms_results.png)


