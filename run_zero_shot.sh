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

python -m snowpack.main_inference_zero_shot --path_to_config "configs/revert_boundary_resize_simple_zero_shot.json" --zero_shot