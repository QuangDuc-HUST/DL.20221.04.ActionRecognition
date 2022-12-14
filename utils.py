import os
import random

import subprocess

import numpy as np
import torch 


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