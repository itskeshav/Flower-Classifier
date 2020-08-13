#!/usr/bin/env python3
""" predict.py
Udacity AIND - Kesahv Kumar
predict.py receives an image an predict using the model classification
"""
__author__ = "Keshav Kumar <keshavraj203@gmail.com>"
__version__ = "1.0.0"
__license__ = "MIT"



import torch
from PIL import Image
import numpy as np
import pandas as pd
import json
import os

import config
from utils import load_ckp, get_test_args
from model import initialize_model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
        image = image_path (str)
        return Tensor
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    from math import floor
    
    image = Image.open(image).convert("RGB")
    
    # Resize with aspect ratio
    width, height = image.size
    size = 256
    ratio = float(width)/float(height)
    if width > height:
        new_height = ratio * size
        image = image.resize((size, int(floor(new_height))), Image.ANTIALIAS)
    else:
        new_width = ratio * size
        image = image.resize((int(floor(new_width)), size), Image.ANTIALIAS)
    
    # Center crop
    width, height = image.size
    size = 224
    
    image = image.crop((
            (width - size) / 2,  # left
            (height - size) / 2, # top
            (width + size) / 2,  # right
            (height + size) / 2  # bottom
        ))
    
    image = np.array(image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    image = image.transpose((2, 0, 1))
    
    return torch.from_numpy(image)


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if not os.path.exists(image_path):
        raise("{} File doesn't exist..".format(image_path))
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    model.cpu()
    
    image = process_image(image_path)
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        output_idx = model(image.float())
        ps = (torch.exp(output_idx) / torch.exp(output_idx).sum(dim=1))
        top_p, top_class = ps.topk(topk)
        
    idx_to_class = {idx: cls for cls, idx in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in top_class.numpy()[0]]
    prob = top_p.numpy()[0]
        
    return prob, classes

def main():
    """
        Image Classification Prediction
    """
    cli_args = get_test_args(__author__, __version__)
    
    # Variables
    image_path = cli_args.input
    checkpoint_path = cli_args.checkpoint
    top_k = cli_args.top_k
    categories_names = cli_args.categories_names

    # LOAD THE PRE-TRAINED MODEL
    model = load_ckp(checkpoint_path, optimizer=None)
    # PREDICT THE TOP_K PROBABILITY AND ITS CORRESPONDING CLASS FROM WHICH IT IS BELONG
    probs, classes = predict(image_path, model, top_k)
    
    # Check the categories file
    if not os.path.isfile(categories_names):
        print(f'Categories file {categories_names} was not found.')
        exit(1)
    
    # Label mapping
    with open(categories_names, 'r') as f:
        cat_to_name = json.load(f)
        
    class_names = [cat_to_name[idx] for idx in classes]
    
    # Display prediction
    data = pd.DataFrame({' Classes': classes, '  Flower': class_names, 'Probability': probs })
    data = data.sort_values('Probability', ascending = False)
    print('The item identified in the image file is:')
    print(data)
    
if __name__ == '__main__':
    main()



    
