import wandb
from model.lrcn import LRCN
from model.c3d import C3D
import utils
import cv2
import torch
import os
import random
from torch.nn import functional as F
import traceback
from app.core.constants.constants import *
import glob
import moviepy.editor as moviepy
import inference


def get_default_agr(args):

    args.device = utils.get_training_device()
    args.weight_path = glob.glob(f'./artifacts/*{args.model_name.replace("_", "-")}*/*.pth')[0]
    args.dataset = DATASET_101

    return args


def download_model():
    api = wandb.Api()

    try:
        artifact = api.artifact(f'dandl/dl_action_recognition/{ARTIFACT_NAME_LRCN}:latest', type='model')

        print('Download LRCN started..')
        artifact.get_path(WEIGHT_FILE).download()
        print('Download LRCN completed!')

        artifact = api.artifact(f'dandl/dl_action_recognition/{ARTIFACT_NAME_C3D}:latest', type='model')

        print('Download C3D started..')
        artifact.get_path(WEIGHT_FILE).download()
        print('Download C3D completed!')

        artifact = api.artifact(f'dandl/dl_action_recognition/{ARTIFACT_NAME_I3D}:latest', type='model')

        print('Download I3D started..')
        artifact.get_path(WEIGHT_FILE).download()
        print('Download I3D completed!')

        artifact = api.artifact(f'dandl/dl_action_recognition/{ARTIFACT_NAME_NON_LOCAL}:latest', type='model')

        print('Download NONLOCAL started..')
        artifact.get_path(WEIGHT_FILE).download()
        print('Download NONLOCAL completed!')

        artifact = api.artifact(f'dandl/dl_action_recognition/{ARTIFACT_NAME_LATE_FUSION}:latest', type='model')

        print('Download LATE FUSION started..')
        artifact.get_path(WEIGHT_FILE).download()
        print('Download LATE FUSION completed!')
    except Exception as e:
        print(traceback.format_exc())


def predict(input, args):

    args = get_default_agr(args)
    print(args)
    args.video_path = input

    output = torch.from_numpy(inference.predict(**vars(args)))

    return torch.topk(output, 10).indices, torch.topk(output.flatten(), 10).values


def convert_avi_to_mp4(filename):

    clip = moviepy.VideoFileClip(filename=filename)
    clip.write_videofile('./app/staging/video/temp_video.mp4')
    clip.close()

    print('done')
