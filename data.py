import torchvision.datasets as datasets
from torchvision import transforms
import torch
import os




def generate_torch_dataset(
    dataset_name, # Which dataset to use 'cifar10', 'fashionmnist'
    train=True, # Return train set if true
    val=True, # Break the train set into train and val if true
    test=True, # Return the test set if true
    val_size=0.2,
    transform=[transforms.ToTensor()], # Convert image to tensor
    target_transform=None,
    seed=42
):
    os.makedirs('./datasets', exist_ok=true)
    root = f'./datasets/{dataset}'

    if dataset_name=='cifar10':
        dataset_class = datasets.CIFAR10
    elif dataset_name=='fashionmnist':
        dataset_class = datasets.FashionMNIST
    
    datasets = []
    if train:
        train_dataset = dataset_class(
            root=root,
            train=True,
            transform=transform, 
            target_transform=target_transform
        )

        if val:
            train_size = int((1-val_size) * len(full_dataset))
            val_size = len(full_dataset) - train_size
            generator = torch.Generator().manual_seed(seed)
            train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=generator)
            datasets.append(train_dataset)
            datasets.append(val_dataset)
        else:
            datasets.append(train_dataset)
    
    if test:
        test_dataset = dataset_class(
            root=root,
            train=False,
            transform=transform, 
            target_transform=target_transform
        )
            
        datasets.append(test_dataset)  

    
    return *datasets


