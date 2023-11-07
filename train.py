from accelerate import Accelerator
import torch
from tqdm import tqdm
import numpy as np
import time
from util import EvaluationCallback, ModelCheckpoint

def train(args, model, train_dataloader, val_dataloader, loss_fn, optimizer, num_epochs, accelerator, train_callbacks=[], val_callbacks=[]):


    best_val_loss = np.inf
    for epoch in tqdm(range(num_epochs)):
        start_train_time = time.time()
        train_loss = 0.0
        model.train()
        Y_true, Y_pred = [], []
        print(f'Epoch {epoch}')
        for i, data in enumerate(train_dataloader, 0):
            optimizer.zero_grad()
            inputs, labels = data
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            accelerator.backward(loss)
            optimizer.step()
            
            Y_true.append(labels); Y_pred.append(outputs)
            train_loss += loss.detach().cpu().item()

        end_train_time = time.time()
        
        # Convert to tensor
        Y_true = torch.cat(Y_true)
        Y_pred = torch.cat(Y_pred)
        Y_true = Y_true.cpu()
        Y_pred = Y_pred.cpu()

        # To get the avg loss
        avg_train_loss = train_loss / len(train_dataloader)

        for callback in train_callbacks:
            callback.on_epoch_end(
                model=model,
                loss=avg_train_loss,
                Y_true=Y_true,
                Y_pred=Y_pred,
                epoch=epoch,
                accelerator=accelerator,
                time_taken=end_train_time-start_train_time
            )
            if isinstance(callback, EvaluationCallback):
                callback.save_results(args.output_dir, 'train.csv')
            if isinstance(callback, ModelCheckpoint):
                callback.save_best_weights(accelerator, args.output_dir)


        # Evaluation phase
        start_val_time = time.time()
        model.eval()
        val_loss = 0.0
        Y_true, Y_pred = [], []
        with torch.no_grad():
            for data in val_dataloader:
                inputs, labels = data
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                Y_true.append(labels); Y_pred.append(outputs)
                val_loss += loss.detach().cpu().item()
        
        end_val_time = time.time()

        # Convert to tensor
        Y_true = torch.cat(Y_true)
        Y_pred = torch.cat(Y_pred)
        Y_true = Y_true.cpu()
        Y_pred = Y_pred.cpu()

        # To get the avg loss
        avg_val_loss = val_loss / len(val_dataloader)
        best_val_loss = min(avg_val_loss, best_val_loss)

        for callback in val_callbacks:
            stop_training = callback.on_epoch_end(
                model=model,
                loss=avg_val_loss,
                Y_true=Y_true,
                Y_pred=Y_pred,
                epoch=epoch,
                accelerator=accelerator,
                time_taken=end_val_time-start_val_time
            )

            if isinstance(callback, EvaluationCallback):
                callback.save_results(args.output_dir, 'val.csv')
            if isinstance(callback, ModelCheckpoint):
                callback.save_best_weights(accelerator, args.output_dir)

            if stop_training:
                return best_val_loss

    return best_val_loss

    
