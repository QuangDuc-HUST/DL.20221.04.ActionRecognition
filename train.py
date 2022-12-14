#
# main file 
# 

import torch

from utils import seed_everything


def get_training_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("We are training on ", device)
    return device

if __name__ == '__main__':

    # set everything
    seed_everything(seed=73)

    # get training device
    device = get_training_device()





