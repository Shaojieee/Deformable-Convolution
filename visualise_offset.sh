#!/bin/bash
#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=8GB
#SBATCH --job-name=offset
#SBATCH --output=output/output_%j_%x.out 
#SBATCH --error=error/error_%j_%x.err 


module load anaconda
source activate nndl_gpu

cd /home/FYP/szhong005/nndl/Deformable-Convolution



python -W ignore visualise_offset.py \
                --video \
                --fps 12 \
                --duration 10 \
                --output_dir "xai" \
                --image_file "xai/sample.png" \
                --model_weights "xai/model_weights.pth" \
                --resnet_version "152" \
                --with_deformable_conv 0 0 0 0\
                --unfreeze_conv 0 0 0 3\
                --unfreeze_offset \
                --unfreeze_fc \
                --num_classes 10 \
    
