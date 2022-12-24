import argparse
import random

import torch
import cv2

from model.lrcn import LRCN
from model.c3d import C3D
from model.i3d import I3D
from model.non_local_i3res import NonLocalI3Res
from model.late_fusion import LateFusion

import utils

def get_arg_parser():
    """
    Get options from CLI
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--weight_path',type=str, default='./ckp/baseline/best.pth')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['hmdb51', 'ucf101'])

    # Module specific args
    parser.add_argument('--model_name', type=str, required=True, 
                        choices=["lrcn", "c3d", "i3d", "non_local", "late_fusion"])

    ## Get the model name now 
    temp_args, _ = parser.parse_known_args()

    if temp_args.model_name == "lrcn":
        parser = LRCN.add_model_specific_args(parser)

        parser.add_argument('--sample_type', type=str, default='5_frames_uniform' )
        parser.add_argument('--resize_to', type=int, default=256) 

    elif temp_args.model_name == "c3d":
        parser = C3D.add_model_specific_args(parser)
        # Data transform
        parser.add_argument('--sample_type', type=str, default= '16_frames_conse_rand' )
        parser.add_argument('--resize_to', type=int, default=112)   
    
    elif temp_args.model_name == "i3d":
        parser = I3D.add_model_specific_args(parser)
         # Data transform
        parser.add_argument('--sample_type', type=str, default='16_frames_conse_rand' )
        parser.add_argument('--resize_to', type=int, default=224)   

    elif temp_args.model_name == "non_local":
        parser = NonLocalI3Res.add_model_specific_args(parser)

        parser.add_argument('--sample_type', type=str, default='16_frames_conse_rand' )
        parser.add_argument('--resize_to', type=int, default=224) 
    
    elif temp_args.model_name == "late_fusion":
        parser = LateFusion.add_model_specific_args(parser)
        parser.add_argument('--sample_type', type=str, default='5_frames_uniform' )
        parser.add_argument('--resize_to', type=int, default=256)   # 5 uni

    return parser.parse_args()


def extract_features_from_video(video_path, type, transform, *args, **kwargs):
    """
    return a stack of torch based on type
    """
    frames = []
    #Read the Video
    video_reader = cv2.VideoCapture(video_path)

    frame_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    

    if type == "5_frames_uniform":
        if frame_count < 5:
            raise NotImplementedError

        #Calculate the interval after which frames will be added to the list
        skip_interval = max(int(frame_count / 5), 1)

        for counter in range(5):
        #Set the current frame postion of the video
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, counter * skip_interval)
            #Read the current frame 
            ret, frame = video_reader.read()
            if not ret:
                break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(image=img)['image'])

    elif type == "16_frames_conse_rand":
        if frame_count < 16:
            raise NotImplementedError

        start_point = random.randint(0, frame_count-16-1)

        for counter in range(16):
            #Set the current frame postion of the video
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, start_point + counter)
            #Read the current frame 
            ret, frame = video_reader.read()
            if not ret:
                break
            
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(image=img)['image'])
    else:
        raise("Not supporting this sample type.")

    video_reader.release()
    return torch.stack(frames, dim=0)

def load_model(dataset, model_name, **kwargs):
    # Get NUM_CLASSES
    if dataset == 'hmdb51':
        NUM_CLASSES = 51
    else:
        NUM_CLASSES = 101


    if model_name == "lrcn":
        net = LRCN(**kwargs,
                    n_class=NUM_CLASSES)
    elif model_name == "c3d":
        net = C3D(**kwargs,
                    n_class=NUM_CLASSES)
    elif model_name == "i3d":
        net = I3D(**kwargs,
                    num_classes=NUM_CLASSES)

    elif model_name == "non_local":
        net = NonLocalI3Res(**kwargs,
                            num_classes=NUM_CLASSES)
        
    elif model_name == "late_fusion":
        net = LateFusion(**kwargs, 
                         n_class=NUM_CLASSES)

    return net

def predict_model(model, features, args):

    model.eval()
    with torch.no_grad():
        clip = features.to(args.device, non_blocking=True)
        logit = model(clip.unsqueeze(0))
        output = torch.softmax(logit, dim=0)
    
    return output.detach().cpu().numpy()

def predict(video_path, weight_path, dataset, model_name, sample_type, resize_to):

    utils.seed_everything(seed=73)

    device = utils.get_training_device()

    class Args():
        def __init__(self, sample_type, resize_to):
            self.sample_type = sample_type
            self.resize_to = resize_to
            self.device = device
    
    args = Args(sample_type, resize_to)


    net = load_model(dataset, model_name)

    net.to(device)

    utils.load_checkpoint(weight_path, net)

    transform = utils.get_transforms(args)['test_transforms']
    
    features = extract_features_from_video(video_path, args.sample_type, transform)

    output = predict_model(net, features, args)

    return output

if __name__ == '__main__':
    
    args = get_arg_parser()
    predict(args.video_path, args.weight_path, args.dataset, args.model_name, args.sample_type, args.resize_to)