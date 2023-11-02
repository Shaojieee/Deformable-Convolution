from accelerate import Accelerator
import torch

def train(model, train_dataloader, val_dataloader, loss_fn, optimizer, epochs):
    """ Train the model for a number of epochs.
    """
    accelerator = Accelerator()
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    train_losses = []
    train_acc = []
    val_losses = []
    val_acc = []
    for epoch in range(epochs):

        train_loss = 0.0
        val_loss = 0.0

        # Training phase
        correct = 0
        total = 0
        model.train()
        for i, data in enumerate(train_dataloader, 0):
            optimizer.zero_grad()
            inputs, labels = data
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            accelerator.backward(loss)
            optimizer.step()
            
            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # track running loss for train
        train_losses.append(train_loss)
        #track accuracy per epoch
        train_acc.append(round(correct/total,4))

        # Evaluation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_dataloader:
                inputs, labels = data
                outputs = model(inputs)
                loss = loss_fn(outputs,labels)

                val_loss+=loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_losses.append(val_loss)
        val_acc.append(round(correct/total,4))

    return  train_losses,val_losses,train_acc,val_acc
