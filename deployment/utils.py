import wandb
from model.lrcn import LRCN
from model.c3d import C3D
import utils
import cv2
import torch
import albumentations as A
import os
import random
from albumentations.pytorch import ToTensorV2
from torch.nn import functional as F
import traceback
from deployment.constants import *


def get_default_agr(args):
    args['restore_file'] = MODEL_FILE
    args['num_workers'] = NUM_WORKERS
    args['device'] = utils.get_training_device()

    if args['model_name'] == "lrcn":
        args['resize_to'] = RESIZE_LRCN
    elif args['model_name'] == "c3d":
        args['resize_to'] = RESIZE_C3D
    
    return args


def get_model(args):

    if args['model_name'] == 'lrcn':
        ARTIFACT_NAME = ARTIFACT_NAME_LRCN
    else:
        ARTIFACT_NAME = ARTIFACT_NAME_C3D

    ## Un-comment when not have model weights 
    # run = wandb.init(project="dl_action_recognition", entity="dandl", anonymous='allow')
    # try:
    #     artifact = run.use_artifact(f'dandl/dl_action_recognition/{ARTIFACT_NAME}:v0', type='model')

    #     print('Download started..')
    #     model = artifact.get_path(MODEL_FILE).download()
    #     print('Download completed!')
    # except Exception as e:
    #     print(traceback.format_exc()) 
    # run.finish()

    # set everything
    utils.seed_everything(seed=73)

    # Get model 
    if args['model_name'] == "lrcn":
        net = LRCN(**args,
                    n_class=NUM_CLASSES)
    elif args['model_name'] == "c3d":
        net = C3D(**args,
                    n_class=NUM_CLASSES)
    
    net.to(args['device'])

    # Load weights
    load_checkpoint(f'./artifacts/{ARTIFACT_NAME}-v0/{MODEL_FILE}', net)
    
    return net

def read_image(img, transform):
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if transform is not None:
        img = transform(image=img)['image']
    
    return img
    
def extract_frames_from_videos_lrcn(video_path, args, sequence_length=5):

    frames = []
    print(f"Start extract {sequence_length} frames..")
    try:
        
        #Read the Video
        video_reader = cv2.VideoCapture(video_path)
        #get the frame count
        frame_count=int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        #Calculate the interval after which frames will be added to the list
        skip_interval = max(int(frame_count/sequence_length), 1)
        #iterate through video frames
        for counter in range(sequence_length):
            #Set the current frame postion of the video
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, counter * skip_interval)
            #Read the current frame 
            ret, frame = video_reader.read()
            if not ret:
                break;
            #Resize the image
            # frame = cv2.resize(frame, (height, width), interpolation = cv2.INTER_CUBIC)
            
            # cv2.imwrite(saved_path + "_" + str(counter+1) + '.png', frame)
            frames.append(read_image(frame, get_transforms(args)['test_transforms']))
    except Exception as e:
        print(f"An error occured while extracting {sequence_length} frames")
        print(traceback.format_exc())
    print(f"Extracted {sequence_length} frames")

    video_reader.release()

    return torch.stack(frames, dim=0).to(args['device'], non_blocking=True)

def extract_frames_from_videos_c3d(video_path, args, sequence_length=16):

    frames = []
    print(f"Start extract {sequence_length} frames..")
    try:
        video_reader = cv2.VideoCapture(video_path)
        #get the frame count
        frame_count=int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        
        #Calculate the starting point
        if frame_count < sequence_length:
            raise NotImplementedError
            
        start_point = max(random.randint(0, frame_count-sequence_length-2), 0)
        
        #iterate through video frames
        for counter in range(sequence_length):
            #Set the current frame postion of the video
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, start_point + counter)
            #Read the current frame 
            ret, frame = video_reader.read()
            if not ret:
                break;
            frames.append(read_image(frame, get_transforms(args)['test_transforms']))

    except Exception as e:
        print(f"An error occured while extracting {sequence_length} frames")
        print(traceback.format_exc())
    print(f"Extracted {sequence_length} frames")

    video_reader.release()

    return torch.stack(frames, dim=0).to(args['device'], non_blocking=True)


def predict(input, args):

    args = get_default_agr(args)
    print(args)

    if args['model_name'] == 'lrcn':
        inputs = extract_frames_from_videos_lrcn(input, args)
    else:
        inputs = extract_frames_from_videos_c3d(input, args)
    print(inputs.shape)
    net = get_model(args)

    with torch.no_grad():
        net.eval()
        output = net(torch.unsqueeze(inputs, dim=0))

    return torch.topk(output, 10).indices, torch.topk(F.softmax(output.flatten(), dim=0), 10).values
    
def get_transforms(args):

    train_transforms = A.Compose(
        [
            A.Resize(args['resize_to'], args['resize_to'], interpolation=cv2.INTER_CUBIC),
            A.Normalize(),
            ToTensorV2(),
        ]
        )
    val_transforms = A.Compose(
        [   
            A.Resize(args['resize_to'], args['resize_to'], interpolation=cv2.INTER_CUBIC),
            A.Normalize(),
            ToTensorV2(),
        ]
        )
    test_transforms = A.Compose(
        [   
            A.Resize(args['resize_to'], args['resize_to'], interpolation=cv2.INTER_CUBIC),
            A.Normalize(),
            ToTensorV2(),
        ]
        )
        
    return {'train_transforms': train_transforms, 
            'val_transforms': val_transforms,
            'test_transforms': test_transforms}


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """

    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict']) #maybe epoch as well

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint





    