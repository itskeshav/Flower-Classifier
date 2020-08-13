from tqdm import tqdm
import torch
import os


def train_fn(data_loaders, model, optimizer, criterion, device, phase='train'):
     

    # set model back to train mode
    model.train()
    
    # initialize variables to monitor training loss and accuracy
    running_loss = 0.0
    running_acc = 0
    running_correct = 0
    
    for image, target in tqdm(data_loaders[phase]):

        # move data and target to available device
        image, target = image.to(device), target.to(device)

        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model 
        output = model(image)

        # calculate the average batch loss
        loss = criterion(output, target)

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # perform a single optimization step (parameter update)
        optimizer.step()

        # Track the training loss
        running_loss += loss.item() * len(target)

        # Track accuracy
        _, pred = torch.max(output, 1)
        running_correct += torch.sum(pred == target).item()
    
    # update average losses and accuracy
    len_of_dataset = len(data_loaders[phase].dataset)
    running_loss = running_loss / len_of_dataset
    running_acc = running_correct / len_of_dataset 
        
    return running_loss, running_acc


def eval_fn(data_loaders, model, criterion, device, phase='valid'):
    
    # set model to evaluation mode
    model.eval()
    
    # initialize variables to monitor validation loss and accuracy
    running_loss = 0.0
    running_acc = 0
    running_correct = 0
    
    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
    
        for image, target in tqdm(data_loaders[phase]):

            # move data and target to available device
            image, target = image.to(device), target.to(device)

            # forward pass: compute predicted outputs by passing inputs to the model 
            output = model(image)

            # calculate the average batch loss
            loss = criterion(output, target)

            # record the validation loss
            running_loss += loss.item() * len(target)

            # calulate accuracy
            _, pred = torch.max(output, 1)
            running_correct += torch.sum(pred == target).item()
        
        
        # update average losses and accuracy
        len_of_dataset = len(data_loaders[phase].dataset)
        running_loss = running_loss / len_of_dataset
        running_acc = running_correct / len_of_dataset
        
    return running_loss, running_acc


        