#!/bin/bash
#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --job-name=jupyter_gpu
#SBATCH --output=output/output_%x_%j.out 
#SBATCH --error=error/error_%x_%j.err 


module load anaconda
source activate nndl

jupyter notebook --ip=$(hostname -i) --port=8886