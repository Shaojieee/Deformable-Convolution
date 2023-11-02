from accelerate import Accelerator
import torch

def test(model, test_dataloader):
    """ Test the model.
    """
    accelerator = Accelerator()
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_dataloader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = round(correct/total,4)

    return accuracy