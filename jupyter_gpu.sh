#!/bin/bash
#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --job-name=jupyter_gpu
#SBATCH --output=output/output_%j_%x.out 
#SBATCH --error=error/error_%j_%x.err 


module load anaconda
source activate nndl

jupyter notebook --ip=$(hostname -i) --port=8886