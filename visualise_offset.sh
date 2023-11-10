

source activate nndl_gpu

python -W ignore visualise_offsets.py \
                --video \
                --fps 12 \
                --duration 10 \
                --output_dir "xai" \
                --image_file "xai/fashionmnist.png" \
                --model_weights "best_weights.pth" \
                --resnet_version "50" \
                --with_deformable_conv 0 0 0 1\
                --unfreeze_conv 0 0 0 1\
                --unfreeze_offset \
                --unfreeze_fc \
                --num_classes 10 \
    
