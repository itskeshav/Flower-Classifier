__author__ = "Keshav Kumar <keshavraj203@gmail.com>"
__version__ = "1.0.0"

import os
import torch
from torchvision import models
from torch import nn
from model import initialize_model
import config
import argparse


def save_ckp(state, save_dir, file_name=None):
    """Saves model and training parameters at checkpoint + state['model_name'] + '.pth'.
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        save_dir: (string) folder where parameters are to be saved
    """
    
    # MAKE A NEW SAVE_DIR TO SAVE CHECKPOINT IF IT IS NOT AVAILABLE
    if not os.path.exists(save_dir):
        print("Checkpoint Directory does not exist! Making directory {}".format(save_dir))
        os.mkdir(save_dir)

    # CREATE A FILE PATH TO SAVE CHECKPOINT IF FILE NAME IS NONE
    if file_name is None:
        checkpoint_path = os.path.join(save_dir, state['model_name']+'.pth')
        
    else:
        # FILE NAME IS VALID OR NOT
        if file_name[-4:] != '.pth':
            print("INVALID FILE NAME EXTENSION FOR CHECKPOINT")
            PRINT("USE .pth EXTENSION TO SAVE CHECKPOINT")
            exit()
        # CREATE A FILE PATH TO SAVE CHECKPOINT IF FILE NAME IS NOT NONE
        checkpoint_path = os.path.join(save_dir, file_name)
    

    # SAVE A CHECKPOINT    
    torch.save(state, checkpoint_path)
    

    
#  A function that loads a checkpoint and rebuilds the model
def load_ckp(checkpoint_path, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location= 'cpu'
    
    if not os.path.exists(checkpoint_path):
        print("{} File doesn't exist ".format(checkpoint_path))
        exit()
    
    # LOAD CHECKPPOINT
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # BUILD MODEL WHICH IS USED TO LOAD MODEL'S WEIGHTS
    model = initialize_model(checkpoint['model_name'], num_classes=config.NO_OF_CLASSES, feature_extract=True, use_pretrained=True)
    
    # LOAD A MODEL'S WEIGHTS  
    model.load_state_dict(checkpoint['state_dict'])
    # LOAD MODEL ATTRIBUTES
    model.class_to_idx = checkpoint['class_to_idx']
    model.best_score = checkpoint['valid_score']
    model.model_name = checkpoint['model_name']

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return model



def get_train_args(__author__, __version__):
    """
    Get arguments for command line train.py
    """

    parser = argparse.ArgumentParser(
        description="Train and save an image classification model.",
        usage="python3 train.py flowers/ --use_gpu",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'data_dir',
        action="store",
        help="Directory from where the training data is loaded"
    )
    
    parser.add_argument(
        '--save_dir',
        action="store",
        default="checkpoints",
        dest="save_dir",
        type=str,
        help="Directory to save checkpoints"
    )

    parser.add_argument(
        '--file_name',
        action="store",
        default= None,
        dest='file_name',
        help='Checkpoint filename',
    )
    
    parser.add_argument(
        '--use_gpu',
        action="store_true",
        default=False,
        dest="use_gpu",
        help="Enables the gpu mode"
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s ' + __version__ + ' [' + __author__ + ']'
    )
    

    return parser.parse_args()
    
    
def get_test_args(__author__, __version__):
    """
    Get arguments for command line test.py
    """

    parser = argparse.ArgumentParser(
        description="Predict an image with the classification model.",
        usage="python3 predict.py input checkpoint --category_names cat_to_name.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'input',
        action="store",
        help="Path to image that is going to be predicted"
    )

    parser.add_argument(
        'checkpoint',
        action="store",
        help="File containing the model checkpoint"
    )
    
    parser.add_argument(
        '--top_k',
        action="store",
        default=5,
        dest="top_k",
        type=int,
        help="Top K most likely classes"
    )
    
    parser.add_argument(
        '--categories_names',
        action="store",
        default="cat_to_name.json",
        dest='categories_names',
        type=str,
        help='Path to file containing the categories',
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s ' + __version__ + ' [' + __author__ + ']'
    )

    return parser.parse_args()



