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
            A.Resize(112, 112, interpolation=cv2.INTER_CUBIC),
            A.Normalize(),
            ToTensorV2(),
        ]
        )
    val_transforms = A.Compose(
        [   
            A.Resize(112, 112, interpolation=cv2.INTER_CUBIC),
            A.Normalize(),
            ToTensorV2(),
        ]
        )
    test_transforms = A.Compose(
        [   
            A.Resize(112, 112, interpolation=cv2.INTER_CUBIC),
            A.Normalize(),
            ToTensorV2(),
        ]
        )
        
    return {'train_transforms': train_transforms, 
            'val_transforms': val_transforms,
            'test_transforms': test_transforms}

class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)

def runcmd(cmd, is_wait=False, *args, **kwargs):
    # function for running command
    process = subprocess.Popen(
        cmd,
        text = True,
        shell = True
    )
    
    if is_wait:
        process.wait()

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