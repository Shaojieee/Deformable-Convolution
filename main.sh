source activate nndl_gpu

python -W ignore main.py \
                --dataset "fashionmnist" \
                --resnet_version "50" \
                --with_deformable_conv 0 0 0 1\
                --unfreeze_conv 0 0 0 1\
                --unfreeze_offset \
                --unfreeze_fc \
                --early_stopping \
                --patience 10 \
                --mode "min" \
                --min_delta 0 \
                --restore_best_weight \
                --learning_rate 0.001 \
                --train_batch_size 128 \
                --eval_batch_size 128 \
                --num_epochs 200 

    
