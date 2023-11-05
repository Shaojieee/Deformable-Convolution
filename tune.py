import optuna
import torch 
import torch.nn as nn
from model import resnet
from utils import ModelCheckpoint
from train import train


def objective(trial, accelerator, args, train_dataloader, val_dataloader):
    # Hyperparameters we want optimize
    params = {
        "lr": trial.suggest_loguniform('lr', 1e-4, 1e-2),
        # "optimizer_name": trial.suggest_categorical('optimizer_name',["SGD", "Adam"])
    }
    
    # Get pretrained model
    model = resnet(
        pretrained=True, 
        num_classes=args.num_classes,
        version=args.resnet_version, 
        dcn=args.with_deformable_conv,
        unfreeze_dcn=args.unfreeze_dcn,
        unfreeze_offset=args.unfreeze_offset,
        unfreeze_fc=args.unfreeze_fc,
    )

    # Define criterion
    loss_fn = nn.CrossEntropyLoss()
    
    # Configure optimizer
    # optimizer = getattr(
    #     torch.optim, params["optimizer_name"]
    # )(model.parameters(), lr=params["lr"])

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    model, train_dataloader, val_dataloader, optimizer = accelerator.prepare(model, train_dataloader, val_dataloader, optimizer)

    early_stopper = ModelCheckpoint(
        early_stop=True, 
        patience=10, 
        min_delta=0, 
        mode='min', 
        restore_best_weights=False
    )

    best_val_loss = train(
        model, 
        train_dataloader, 
        val_dataloader, 
        loss_fn, 
        optimizer, 
        args.num_epochs,
        accelerator,
        val_callbacks=[early_stopper]
    )
    
    return best_val_loss