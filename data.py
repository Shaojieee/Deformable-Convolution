import torchvision.datasets as datasets
from torchvision import transforms
import torch
import os



def generate_torch_dataset(
    dataset_name, # Which dataset to use 'cifar10', 'fashionmnist'
    val_size=0.2,
    transform=transforms.ToTensor(), # Convert image to tensor
    target_transform=None,
    debug=False,
    seed=42
):
    os.makedirs('./datasets', exist_ok=True)
    root = f'./datasets/{dataset_name}'

    if dataset_name=='cifar10':
        dataset_class = datasets.CIFAR10
    elif dataset_name=='fashionmnist':
        dataset_class = datasets.FashionMNIST
    

    train_dataset = dataset_class(
        root=root,
        train=True,
        transform=transform, 
        target_transform=target_transform,
        download=True
    )

    if isinstance(train_dataset.targets, list):
        num_classes = len(set(train_dataset.targets))
    else:
        num_classes = len(train_dataset.targets.unique())

    if debug:
        train_dataset = torch.utils.data.Subset(train_dataset, [x for x in range(100)])

    train_size = int((1-val_size) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size], generator=generator)

    test_dataset = dataset_class(
        root=root,
        train=False,
        transform=transform, 
        target_transform=target_transform,
        download=True
    )

    
    return train_dataset, val_dataset, test_dataset, num_classes


def fashionmnist_image_transform():
    transform = transforms.Compose([ 
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) 
    return transform

def cifar10_image_transform():
    transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) 
    return transform