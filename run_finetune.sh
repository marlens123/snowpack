#!/bin/bash
#SBATCH -p long
#SBATCH --time=08:00:00
#SBATCH --error=error.txt
#SBATCH --gres=gpu:rtx8000

module load python/3.10
source .env/bin/activate
poetry install

#python -m snowpack.main_finetune --path_to_config "configs/multiclass_larger_lr.json"
python -m snowpack.main_finetune --path_to_config "configs/multiclass_ps_400_medium_lr.json"
