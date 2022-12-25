import os 
from urllib import request


import torch
from torch import nn

from .utils import c3d_model



class C3D(c3d_model.C3D):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("C3D")
        parser.add_argument("--drop_out", type=float, default=0.5)
        parser.add_argument("--pretrain", action='store_false')
        parser.add_argument("--weight_folder", type=str, default='./model/weights/')
        return parent_parser

    def __init__(self, drop_out, n_class, pretrain, weight_folder, *args, **kwargs):
        super(C3D, self).__init__(drop_out=drop_out)
        
        if pretrain:
            file_name = "c3d_weights.pickle"
            weight_path = os.path.join(weight_folder, file_name)
            if not os.path.exists(weight_path):
                print("Download C3D pretrained on Sports1M..")
                LINK_DOWNLOAD = "https://aimagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle"
                response = request.urlretrieve(LINK_DOWNLOAD,  weight_path)

            self.load_state_dict(torch.load(weight_path))

            print("Load C3D pretrained weights successfully.")

        self.fc8 = nn.Linear(4096, n_class)
