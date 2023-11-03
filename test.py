from accelerate import Accelerator
import torch

def test(model, loss_fn, test_dataloader, callbacks=[]):
    """ Test the model.
    """

    model.eval()
    test_loss = 0.0
    Y_true, Y_pred = [], []
    with torch.no_grad():
        for data in test_dataloader:
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
    avg_train_loss = train_loss / len(test_dataloader)

    for callback in val_callbacks:
        callback.on_epoch_end(
            model=model,
            loss=avg_train_loss,
            Y_true=Y_actual,
            Y_pred=Y_pred,
            epoch=epoch
        )
