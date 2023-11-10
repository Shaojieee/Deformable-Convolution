# Experimenting with Deformable Convolution

This repository contains code to train ResNet models with Deformable Convolution using the FashionMNIST and CIFAR10 datasets.

---
## Installation
Ensure that you have conda set up on your device. 
Install the environment by running the following command.

```
conda env create -f env.yml
source activate nndl_gpu
```
By default, the environment name will be `nndl_gpu`.

## Running Experiments
Experiments can be run using the `main.sh` script.
You can read up on the explanation of the different flags in the `parse_args.py` file under the `main_parse_args` function.

```
python -W ignore main.py \
                --dataset $DATASET \
                --output_dir 'results/exp_1' \
                --resnet_version $VERSION \
                --with_deformable_conv 0 0 0 3\
                --unfreeze_conv 0 0 0 3\
                --unfreeze_offset \
                --unfreeze_fc \
                --early_stopping \
                --patience 10 \
                --mode "min" \
                --min_delta 0 \
                --restore_best_weight \
                --learning_rate 0.0001 \
                --train_batch_size 64 \
                --eval_batch_size 64 \
                --num_epochs 200 
```

For `--with_deformable_conv` flag, `1 2 3 4` will alter the ResNet model in the following way.
1. Replace the last `1` 3x3 convolution layer in the 2nd ResNet Conv block with Deformable Convolution. 
2. Replace the last `2` 3x3 convolution layer in the 3rd ResNet Conv block with Deformable Convolution. 
3. Replace the last `3` 3x3 convolution layer in the 4th ResNet Conv block with Deformable Convolution. 
4. Replace the last `4` 3x3 convolution layer in the 5th ResNet Conv block with Deformable Convolution. 