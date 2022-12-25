#
# API for non local i3res model
#
import os
from urllib import request

import torch
from torch import nn

from .utils import non_local_i3res_model

class NonLocalI3Res(non_local_i3res_model.I3Res50):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("non_local")
        parser.add_argument("--use_nl", action='store_true')
        parser.add_argument("--weight_folder", type=str, default='./model/weights/')

        return parent_parser

    def __init__(self, num_classes, use_nl, weight_folder, 
                block=non_local_i3res_model.Bottleneck, 
                layers=[3, 4, 6, 3],
                *args,
                **kwargs):

        super().__init__(block=block, layers=layers, use_nl=use_nl)
        
        LINK_DOWNLOAD = {"I3Res50":         "https://transfer.sh/YL1fAh/i3res_baseline.pth",
                         "I3Res50_nonlocal":"https://transfer.sh/PhrU8U/i3res_nonlocal.pth"}

        if use_nl: 
            link = LINK_DOWNLOAD["I3Res50_nonlocal"]
            file_name = "i3res_nonlocal.pth"
        else:
            link = LINK_DOWNLOAD["I3Res50"]
            file_name = "i3res_baseline.pth"

        weight_path = os.path.join(weight_folder, file_name)

        if not os.path.exists(weight_path):
            print(f"Download the weight to {weight_path}")
            response = request.urlretrieve(link,  weight_path)    

        self.load_state_dict(torch.load(weight_path))
        print("Load pretrained weights successfully..")

        input_fc = self.fc.in_features
        self.fc = nn.Linear(input_fc, num_classes)