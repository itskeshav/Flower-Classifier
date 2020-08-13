
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
# https://stackoverflow.com/questions/52532914/pytorch-passing-architecture-type-with-argprse  

from torchvision import models
from torch import nn
import os


def set_parameter_requires_grad(model, feature_extracting=True):
    """ Set Model Parameters’ .requires_grad attribute
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract=True, use_pretrained=True):

    model_ft = None
    
    model_name_list = ['alexnet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 
    'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19', 'densenet121', 'densenet169', 'densenet201','densenet161']
    
    
    # model name is valid or not
    if model_name not in model_name_list:
        print("Invalid model name, choose the model name from a following list")
        print(model_name_list)
        os._exit(0)
        
        
    # load the pretrained model
    model_ft = models.__dict__[model_name](pretrained = use_pretrained)
    
    # Set Model Parameters’ .requires_grad attribute
    set_parameter_requires_grad(model_ft, feature_extract)
    
    # Re-Define the last layer
    if model_name.startswith("resnet"):
        """ Resnet
        """
        input_size = model_ft.fc.in_features
        model_ft.fc = nn.Linear(input_size, num_classes)
        
        
    elif model_name.startswith("vgg"):
        """ Vgg
        """
        input_size = model_ft.classifier[0].in_features
        model_ft.classifier = nn.Sequential(nn.Linear(input_size, num_classes))
        
    elif model_name.startswith("alexnet"):
        """"Alexnet
        """
        input_size = model_ft.classifier[1].in_features
        model_ft.classifier = nn.Sequential(nn.Dropout(0.5),
                                            nn.Linear(input_size, num_classes))    
        
    else:
        """Densenet
        """
        input_size =  model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(input_size, num_classes)
    
    #print("{} Model Initialisation Successfull....".format(model_name))
        
    return model_ft


    

    
    


