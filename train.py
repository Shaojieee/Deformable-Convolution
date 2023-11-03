from accelerate import Accelerator
import torch
from tqdm import tqdm

def train(model, train_dataloader, val_dataloader, loss_fn, optimizer, num_epochs, accelerator, train_callbacks=[], val_callbacks=[]):

    for epoch in tqdm(range(num_epochs)):
        
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

        
        # Convert to tensor
        Y_true = torch.cat(Y_true)
        Y_pred = torch.cat(Y_pred)
        Y_true = Y_true.cpu()
        Y_pred = Y_pred.cpu()

        # To get the avg loss
        avg_train_loss = train_loss / len(train_dataloader)

        for callback in train_callbacks:
            stop_training = callback.on_epoch_end(
                model=model,
                loss=avg_train_loss,
                Y_true=Y_true,
                Y_pred=Y_pred,
                epoch=epoch,
                accelerator=accelerator
            )
            if stop_training:
                return


        # Evaluation phase
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
        
        # Convert to tensor
        Y_true = torch.cat(Y_true)
        Y_pred = torch.cat(Y_pred)
        Y_true = Y_true.cpu()
        Y_pred = Y_pred.cpu()

        # To get the avg loss
        avg_val_loss = val_loss / len(val_dataloader)
    
        for callback in val_callbacks:
            stop_training = callback.on_epoch_end(
                model=model,
                loss=avg_val_loss,
                Y_true=Y_true,
                Y_pred=Y_pred,
                epoch=epoch,
                accelerator=accelerator
            )
            if stop_training:
                return

