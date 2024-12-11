#!/bin/bash
#SBATCH -p long
#SBATCH --time=08:00:00
#SBATCH --error=error.txt

module load python/3.10
source .env/bin/activate
poetry install

python -m snowpack.patches.visualize 