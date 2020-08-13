import torch
from torchvision import datasets, transforms
import os
import config

# Data augmentation and normalization for training
# Just normalization for validation
def load_data(data_dir, img_size, batch_size):
    
    if not os.path.exists(data_dir):
        raise("Data dir {} not exist".format(data_dir))
        exit()
    
    mean = config.MODEL_MEAN   #[0.485, 0.456, 0.406]
    std  = config.MODEL_STD    #[0.229, 0.224, 0.225]
    
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(60),
            transforms.ToTensor(),
            transforms.Normalize(
            mean, std)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
            mean, std)
        ])
    }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
        'valid': datasets.ImageFolder(os.path.join(data_dir, 'valid'), data_transforms['valid']),
        'test' : datasets.ImageFolder(os.path.join(data_dir, 'test'),  data_transforms['valid'])
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size[x], True) for x in ['train', 'valid', 'test']
    }
    
    print("Data augmentation and normalization is successfull...")
    
    return dataloaders

