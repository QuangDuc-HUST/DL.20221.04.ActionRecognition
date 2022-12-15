import torch
from torch import nn
import os 

from .utils import custom_c3d_model, c3d_model
from ..utils import run_cmd


class C3D(c3d_model.C3D):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("C3D")
        parser.add_argument("--drop_out", type=float, default=0.5)
        parser.add_argument("--pretrain", action='store_false')
        parser.add_argument("--weight_path", type=str, default='c3d.pickle')
        return parent_parser

    def __init__(self, drop_out, n_class, pretrain, weight_path, *args, **kwargs):
        super(C3D, self).__init__(drop_out=drop_out)

        self.fc8 = nn.Linear(4096, n_class)
        
        if pretrain:
            if not os.path.exists(weight_path):
                print("LUL")
            
