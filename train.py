#!/usr/bin/env python3
""" train.py
train.py train a new network on a specified data set
"""

__author__ = "Keshav Kumar <keshavraj203@gmail.com>"
__version__ = "1.0.0"
__license__ = "MIT"


import torch
from workspace_utils import active_session

from matplotlib import pyplot as plt
import numpy as np
import time
import copy
import os

from model import initialize_model
from engine import train_fn, eval_fn
from dataset import load_data
import config
from utils import save_ckp, load_ckp, get_train_args


def optimizer_fn(model_name, model, lr_rate=0.01):
    
    if model_name.startswith("resnet"):
        parameter_to_upd = model.fc.parameters()
    else:
        parameter_to_upd = model.classifier.parameters()     
    optimizer = torch.optim.Adagrad(parameter_to_upd, lr=lr_rate, weight_decay=0.001)
    return optimizer

def loss_fn():
    criterion = torch.nn.CrossEntropyLoss()
    return criterion

def main(ckp_path=None):
    """ckp_path (str): checkpoint_path
    Train the model from scratch if ckp_path is None else
    Re-Train the model from previous checkpoint
    """
    cli_args = get_train_args(__author__, __version__)

    # Variables
    data_dir = cli_args.data_dir
    save_dir = cli_args.save_dir
    file_name = cli_args.file_name
    use_gpu = cli_args.use_gpu
    
    # LOAD DATA
    data_loaders = load_data(data_dir, config.IMG_SIZE, config.BATCH_SIZE)
    
    # BUILD MODEL
    if ckp_path == None:
        model = initialize_model(model_name= config.MODEL_NAME, num_classes=config.NO_OF_CLASSES,
                             feature_extract=True, use_pretrained=True)
    else:
        model = load_ckp(ckp_path)

                       
    # Device is available or not 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # If the user wants the gpu mode, check if cuda is available
    if (use_gpu == True) and (torch.cuda.is_available() == False):
        print("GPU mode is not available, using CPU...")
        use_gpu = False
    
        

    # MOVE MODEL TO AVAILBALE DEVICE     
    model.to(device)
    
    # DEFINE OPTIMIZER
    optimizer = optimizer_fn(model_name=config.MODEL_NAME, model=model, lr_rate=config.LR_RATE)
    
    # DEFINE SCHEDULER
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="min", 
                                                           patience=5,
                                                           factor=0.3,verbose=True)
    
    # DEFINE LOSS FUNCTION                                                       
    criterion = loss_fn() 
    
    # LOAD BEST MODEL'S WEIGHTS
    best_model_wts = copy.deepcopy(model.state_dict()) 
    
    # BEST VALIDATION SCORE
    if ckp_path == None:
        best_score = -1 # IF MODEL IS TRAIN FROM SCRATCH
    else:
        best_score = model.best_score # IF MODEL IS RE-TRAIN
          
    # NO OF ITERATION
    no_epochs = config.EPOCHS
    # KEEP TRACK OF LOSS AND ACCURACY IN EACH EPOCH
    stats = {'train_losses': [], 'valid_losses': [], 'train_accuracies': [], 'valid_accuracies': []}
    
    
    print("Models's Training Start......")
    
    
    for epoch in range(1, no_epochs + 1):
        train_loss , train_score   = train_fn(data_loaders, model, optimizer, criterion, device, phase='train')
        val_loss   , val_score     = eval_fn(data_loaders, model, criterion, device = config.DEVICE, phase='valid')
        scheduler.step(val_loss)
        
        
        # SAVE MODEL'S WEIGHTS IF MODEL' VALIDATION ACCURACY IS INCREASED
        if val_score > best_score:
            print('Validation score increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                best_score, val_score))
            best_score = val_score   
            best_model_wts = copy.deepcopy(model.state_dict())   #Saving the best model' weights
            
        
        # MAKE A RECORD OF AVERAGE LOSSES AND ACCURACY IN EACH EPOCH FOR PLOTING
        stats['train_losses'].append(train_loss)
        stats['valid_losses'].append(val_loss)
        stats['train_accuracies'].append(train_score)
        stats['valid_accuracies'].append(val_score)
            

        # PRINT TRAINING AND VALIDATION LOOS/ACCURACIES AFTER EACH EPOCH
        epoch_len = len(str(no_epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{no_epochs:>{epoch_len}}] ' +
                     '\t' +
                     f'train_loss: {train_loss:.5f} ' +
                     '\t' +
                     f'train_score: {train_score:.5f} ' +
                     '\t' +
                     f'valid_loss: {val_loss:.5f} ' +
                     '\t' +
                     f'valid_score: {val_score:.5f}'
                     )
        print(print_msg)
        
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    
    # create checkpoint variable and add important data
    model.class_to_idx = data_loaders['train'].dataset.class_to_idx
    model.best_score = best_score
    model.model_name = config.MODEL_NAME
    checkpoint = {
        'epoch':no_epochs,
        'lr_rate':config.LR_RATE,
        'model_name':config.MODEL_NAME,
        'batch_size':config.BATCH_SIZE,
        'valid_score':best_score,
        'optimizer': optimizer.state_dict(),
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
        }
    
    # SAVE CHECKPOINT
    save_ckp(checkpoint, save_dir, file_name)
    
    
    print("Models's Training is Successfull......")
    
    return model    
        
if __name__ == "__main__":
    with active_session():
    # do long-running work here
        main()        
    







