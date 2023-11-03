#!/bin/bash
#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --job-name=train_resnet
#SBATCH --output=output/output_%j_%x.out 
#SBATCH --error=error/error_%j_%x.err 


module load anaconda
source activate nndl

cd /home/FYP/szhong005/nndl/Deformable-Convolution

python -W ignore main.py \
                --fp16 \
                --output_dir "resnet_101_fashionmnist" \
                --dataset "fashionmnist" \
                --resnet_version "101" \
                --mode "min" \
                --restore_best_weight \
                --learning_rate 0.005 \
                --train_batch_size 32 \
                --eval_batch_size 32 \
                --num_epochs 2
