#!/bin/bash
#SBATCH -p long
#SBATCH --time=08:00:00
#SBATCH --error=error.txt
#SBATCH --mem=64GB
#SBATCH --gres=gpu:rtx8000

module load python/3.10
source .env/bin/activate
poetry install

export WANDB_API_KEY=d2ca547c9f807e8db70308537f4d7b64b6077b81

export CUDA_LAUNCH_BLOCKING=1

python -m snowpack.main_finetune --path_to_config "configs/multiclass_larger_lr.json"
python -m snowpack.main_finetune --path_to_config "configs/multiclass_full_image.json"