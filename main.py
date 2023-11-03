from parse_args import parse_args
from data import generate_torch_dataset, fashionmnist_image_transform
from model import resnet
from utils import EvaluationCallback, ModelCheckpoint, evaluation_fn
from train import train
from test import test

import datetime
import os
from tqdm import tqdm
import torch
import torch.nn
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator 

def main():

    args = parse_args()
    print(args)

    if args.output_dir==None:
        args.output_dir = f'./resnet_{args.resnet_version}_{"deformable" if args.with_deformable_conv else "normal"}_{args.dataset}_{datetime.datetime.now().strftime("%Y%m%d_%H%M")}'

    os.makedirs(args.output_dir, exist_ok=True)
    if args.fp16:
        args.mixed_precision="fp16"
    else:
        args.mixed_precision="no"

    

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        cpu=args.cpu
    )

    device = accelerator.device
    print(f'Device:{device}')


    train_dataset, val_dataset, test_dataset, num_classes = generate_torch_dataset(
        args.dataset,  
        val_size=0.2,
        transform=fashionmnist_image_transform() if args.dataset=='fashionmnist' else transforms.ToTensor()
    )

    print(num_classes)
    

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=True)

    model = resnet(version=args.resnet_version, pretrained=True, dcn=args.with_deformable_conv, num_classes=num_classes)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )

    train_callback = EvaluationCallback(evaluation_fn, type='Train')
    val_callback = EvaluationCallback(evaluation_fn, type='Val')
    test_callback = EvaluationCallback(evaluation_fn, type='Test')
    model_checkpoint = ModelCheckpoint(
        early_stop=args.early_stopping, 
        patience=args.patience, 
        min_delta=args.min_delta, 
        mode=args.mode, 
        restore_best_weights=args.restore_best_weights
    )

    model, train_dataloader, val_dataloader, test_dataloader, optimizer = accelerator.prepare(model, train_dataloader, val_dataloader, test_dataloader, optimizer)

    train(
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

    train_callback.save_results(args.output_dir, 'train.csv')
    val_callback.save_results(args.output_dir, 'val.csv')

    if args.restore_best_weights:
        model_checkpoint.load_best_weights(model)
    
    test(
        model, 
        loss_fn,
        test_dataloader,
        callbacks=[test_callback]
    )


    test_callback.save_results(args.output_dir, 'test.csv')
    model_checkpoint.save_best_weights(accelerator, args.output_dir, 'best_weights.pth')



if __name__=='__main__':
    main()



