import os
import random
import argparse
import subprocess

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import numpy as np
import torch 


def get_transforms():

    train_transforms = A.Compose(
        [
            A.Resize(256, 256, interpolation=cv2.INTER_CUBIC),
            A.Normalize(),
            ToTensorV2(),
        ]
        )

    val_transforms = A.Compose(
        [   
            A.Resize(256, 256, interpolation=cv2.INTER_CUBIC),
            A.Normalize(),
            ToTensorV2(),
        ]
        )


    test_transforms = A.Compose(
        [   
            A.Resize(256, 256, interpolation=cv2.INTER_CUBIC),
            A.Normalize(),
            ToTensorV2(),
        ]
        )
    
    return {'train_transforms': train_transforms, 
            'val_transforms': val_transforms,
            'test_transforms': test_transforms}


def runcmd(cmd, is_wait=False, *args, **kwargs):
    # function for running command
    process = subprocess.Popen(
        cmd,
        text = True,
        shell = True
    )
    
    if is_wait:
        process.wait()

def str2bool(v):
    """
    src: https://stackoverflow.com/a/43357954
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def seed_everything(seed=73):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_training_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("We are training on", device)
    return device

def acc_metrics(preds, targets):
    return (preds.argmax(1) == targets).sum() / preds.shape[0]
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']