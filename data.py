import torchvision.datasets as datasets
from torchvision import transforms
import torch
import os


# Function to return Dataset
def generate_torch_dataset(
    dataset_name, # Which dataset to use 'cifar10', 'fashionmnist'
    val_size=0.2,
    transform=transforms.ToTensor(), # Convert image to tensor
    target_transform=None,
    debug=False,
    seed=42
):
    # Download dataset in new directory
    os.makedirs('./datasets', exist_ok=True)
    root = f'./datasets/{dataset_name}'

    if dataset_name=='cifar10':
        dataset_class = datasets.CIFAR10
    elif dataset_name=='fashionmnist':
        dataset_class = datasets.FashionMNIST
    
    # Get train dataset from pytorch
    train_dataset = dataset_class(
        root=root,
        train=True,
        transform=transform, 
        target_transform=target_transform,
        download=True
    )

    # Get the no. of classes in the dataset
    if isinstance(train_dataset.targets, list):
        num_classes = len(set(train_dataset.targets))
    else:
        num_classes = len(train_dataset.targets.unique())

    # Get subset of data for debugging purposes
    if debug:
        train_dataset = torch.utils.data.Subset(train_dataset, [x for x in range(100)])

    # Split into train and val dataset
    train_size = int((1-val_size) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size], generator=generator)

    # Get the test dataset from pytorch
    test_dataset = dataset_class(
        root=root,
        train=False,
        transform=transform, 
        target_transform=target_transform,
        download=True
    )

    
    return train_dataset, val_dataset, test_dataset, num_classes


# Transformation to perform on fashionmnist
def fashionmnist_image_transform(normalise=True):
    transform = [ 
        # Convert to 3 channels for ResNet
        transforms.Grayscale(3),
        # Resize to 224 for ResNet
        transforms.Resize(224),
        transforms.ToTensor(),
    ]
    if normalise:
        # Normalise according to ResNet pre-trained model
        transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    transform = transforms.Compose(transform) 
    return transform

# Transformation to perform on cifar 10
def cifar10_image_transform(normalise=True):
    transform = [
        # Resize to 224 for ResNet
        transforms.Resize(224),
        transforms.ToTensor(),
    ]
    if normalise:
        # Normalise according to ResNet pre-trained model
        transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    transform = transforms.Compose(transform) 
    return transform