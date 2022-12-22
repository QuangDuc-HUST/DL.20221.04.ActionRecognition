import pickle
import wandb
from model.lrcn import LRCN
from model.c3d import C3D
import utils
import os
import argparse
import cv2
import torch
import glob


def get_arg_parser(model_name):

    parser = argparse.ArgumentParser()

    # PROGRAM level args
    parser.add_argument('--restore_file',type=str, default='best.pth')


    # DataModule specific args
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)

    if model_name == "lrcn":
        parser = LRCN.add_model_specific_args(parser)
    elif model_name == "c3d":
        parser = C3D.add_model_specific_args(parser)

    return parser.parse_args()


def get_model(model_name, type):

    NUM_CLASSES = 101
    MODEL_FILE = 'best.pth'
    
    run = wandb.init(project="dl_action_recognition", entity="dandl")

    if type=='lrcn':
        ARTIFACT_NAME = '3eyhjzfd_model'
    else:
        ARTIFACT_NAME = '38ckat92_model'
    
    artifact = run.use_artifact(f'dandl/dl_action_recognition/{ARTIFACT_NAME}:v0', type='model')
    model = artifact.get_path(MODEL_FILE).download()

    args = get_arg_parser(model_name)

    args.device = utils.get_training_device()
    dict_args = vars(args)

    # set everything
    utils.seed_everything(seed=73)

    # Get model 
    if args.model_name == "lrcn":
        net = LRCN(**dict_args,
                    n_class=NUM_CLASSES)
    elif args.model_name == "c3d":
        net = C3D(**dict_args,
                    n_class=NUM_CLASSES)
    
    net.to(args.device)

    # Load weights
    utils.load_checkpoint(f'./artifact/dl_action_recognition/{ARTIFACT_NAME}/{MODEL_FILE}', net)
    
    run.finish()

    return net

def read_image(image_path, transform):
        
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if transform is not None:
        img = transform(image=img)['image']
    
    return img

def get_model_input(filename):

    lst_imgs = glob.glob(f'./deployment/staging/{filename}*')

    imgs = torch.stack([read_image(path, utils.get_transforms()['test_transforms']) for path in lst_imgs], dim=0)

    return imgs
    
def feature_extraction(video_path, saved_path):
    try:
        width = 256
        height = 256
        sequence_length = 5 
        
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
            frame = cv2.resize(frame, (height, width), interpolation = cv2.INTER_CUBIC)
            
            cv2.imwrite(saved_path + "_" + str(counter+1) + '.png', frame)
    except Exception as e:
        print("An error occured while extracting")
        print(e)
    print("Feature extraction completed")
    video_reader.release()


def predict(input, filename, model_name):

    feature_extraction(input, f'./deployment/staging/{filename}')

    img_inputs = get_model_input(filename)

    model = get_model(model_name)







    