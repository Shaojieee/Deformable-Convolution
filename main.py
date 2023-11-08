from parse_args import main_parse_args
from data import generate_torch_dataset, fashionmnist_image_transform, cifar10_image_transform
from model import resnet
from utils import EvaluationCallback, ModelCheckpoint, evaluation_fn
from train import train
from test import test
from tune import objective

import datetime
import os
import json
from tqdm import tqdm
import torch
import torch.nn
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator 
import optuna

def main():

    args = main_parse_args()
    print(args)

    args.has_deformable_conv = any(args.with_deformable_conv)


    if args.output_dir==None:
        # Setting output dir
        args.output_dir = f'./resnet_{args.resnet_version}{"_tuned" if args.tune else ""}_{"deformable" if args.has_deformable_conv else "normal"}_{args.dataset}_{datetime.datetime.now().strftime("%Y%m%d_%H%M")}'
    # Making output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Saving experiment config
    args_dict = vars(args)
    with open(f"{args.output_dir}/experiment_details.json", 'w') as f:
        json.dump(args_dict, f)
    
    if args.fp16:
        args.mixed_precision="fp16"
    else:
        args.mixed_precision="no"

    # For training on the GPU
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        cpu=args.cpu
    )

    device = accelerator.device
    print(f'Device:{device}')

    # Generate the torch Dataset object
    train_dataset, val_dataset, test_dataset, num_classes = generate_torch_dataset(
        args.dataset,  
        val_size=0.2,
        transform=fashionmnist_image_transform() if args.dataset=='fashionmnist' else cifar10_image_transform(),
        debug=args.debug
    )
    args.num_classes = num_classes
    print(f"No. of Classes: {num_classes}")
    
    # Convert Dataset in DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=True)

    # Perform tuning using optuna Tree-structured Parzen Estimator algorithm
    if args.tune:
        sampler = optuna.samplers.TPESampler()    
        study = optuna.create_study(
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=2, n_warmup_steps=5, interval_steps=3
            ),
            direction='minimize'
        )
        # Run the optimisation algorithm
        study.optimize(func=(lambda x: objective(x,accelerator, args, train_dataloader, val_dataloader)), timeout=4*60*60, n_trials=10)
        print(f'Best Loss Value: {study.best_value}')
        print(f"Best Parameters: {study.best_params}")

        optuna_df = study.trials_dataframe()
        optuna_df.to_csv(f'{args.output_dir}/optuna_study.csv')

        fig = optuna.visualization.plot_contour(study,params=['lr'])
        fig.write_image(f"{args.output_dir}/contour_plot.png")

        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image(f"{args.output_dir}/parallel_coordinate.png")
        # Setting the optimised learning rate
        args.learning_rate = study.best_params['lr']

        print('Training with best parameters')


    # Create model according to experiment configs
    model = resnet(
        pretrained=True, 
        # Num of output classes
        num_classes=args.num_classes,
        # ResNet Version to use
        version=args.resnet_version, 
        # Location ofDeformable Convolution
        dcn=args.with_deformable_conv,
        # Convolution layer to unfreeze
        unfreeze_conv=args.unfreeze_conv,
        # Bool to unfreeze offset
        unfreeze_offset=args.unfreeze_offset,
        # Bool to unfreeze output layer
        unfreeze_fc=args.unfreeze_fc,
    )

    # Whether to use user specified weights
    if args.model_weights:
        try:
            model.load_state_dict(torch.load(args.model_weights, map_location=torch.device('cpu')), strict=True)
        except Exception as e:
            print(e)
            model.load_state_dict(torch.load(args.model_weights, map_location=torch.device('cpu')), strict=False)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )

    # Callback that runs after every epoch to log results
    train_callback = EvaluationCallback(evaluation_fn, type='Train')
    val_callback = EvaluationCallback(evaluation_fn, type='Val')
    test_callback = EvaluationCallback(evaluation_fn, type='Test')

    # Callback that runs after every epoch to check for early stopping and to store best weights
    model_checkpoint = ModelCheckpoint(
        early_stop=args.early_stopping, 
        patience=args.patience, 
        min_delta=args.min_delta, 
        mode=args.mode, 
        restore_best_weights=args.restore_best_weights
    )

    # Send the model and data to the GPU
    model, train_dataloader, val_dataloader, test_dataloader, optimizer = accelerator.prepare(model, train_dataloader, val_dataloader, test_dataloader, optimizer)

    # Training function
    train(
        args,
        model, 
        train_dataloader, 
        val_dataloader, 
        loss_fn, 
        optimizer, 
        args.num_epochs,
        accelerator,
        train_callbacks=[train_callback],
        val_callbacks=[val_callback, model_checkpoint]
    )

    # Saving logs
    train_callback.save_results(args.output_dir, 'train.csv')
    val_callback.save_results(args.output_dir, 'val.csv')

    # Restore models best weights if True else use weights at last epoch for testing
    if args.restore_best_weights:
        model_checkpoint.load_best_weights(model)
    
    # Testing function
    test(
        model, 
        loss_fn,
        test_dataloader,
        callbacks=[test_callback]
    )

    # Saving test results
    test_callback.save_results(args.output_dir, 'test.csv')
    # Save the best weights
    model_checkpoint.save_best_weights(accelerator, args.output_dir, 'best_weights.pth')



if __name__=='__main__':
    main()



