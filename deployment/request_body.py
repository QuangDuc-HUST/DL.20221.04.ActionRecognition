from pydantic import BaseModel
from fastapi import Query, FastAPI, UploadFile, Form, File
from typing import Set, Union
from enum import Enum


class Lrcn_request_body(BaseModel):
    model_name: str = Form(default='lrcn')
    latent_dim: int = Form(default=512)
    hidden_size: int = Form(default=256)
    lstm_layers: int = Form(default=2)
    bidirectional: bool = Form(default=True)


class C3d_request_body(BaseModel):
    model_name: str = Form(default='c3d')
    drop_out: float = Form(default=0.5)
    pretrain: bool = Form(default=False)
    weight_path: str = Form(default='c3d.pickle')

    

    

    
