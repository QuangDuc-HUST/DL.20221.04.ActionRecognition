from model.c3d_2_dataset import C3D
from model.non_local_i3res_2_dataset import NonLocalI3Res
from model.late_fusion_2_dataset import LateFusion
import argparse
import random
import torch
import cv2
import utils
import warnings
warnings.simplefilter("ignore", UserWarning)


def get_arg_parser():
    """
    Get options from CLI
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--weight_path', type=str, default='./ckp/baseline/best.pth')
    parser.add_argument('--dataset_1', type=str, default='ucf101',
                        choices=['hmdb51', 'ucf101'])

    parser.add_argument('--dataset_2', type=str, default='hmdb51',
                        choices=['hmdb51', 'ucf101'])

    parser.add_argument('--first_dataset', action='store_true')

    # Module specific args
    parser.add_argument('--model_name', type=str, required=True,
                        choices=["c3d", "non_local", "late_fusion"])

    # Get the model name now
    temp_args, _ = parser.parse_known_args()

    if temp_args.model_name == "c3d":
        parser = C3D.add_model_specific_args(parser)
        # Data transform
        parser.add_argument('--sample_type', type=str, default='16_frames_conse_rand')
        parser.add_argument('--resize_to', type=int, default=112)

    elif temp_args.model_name == "non_local":
        parser = NonLocalI3Res.add_model_specific_args(parser)

        parser.add_argument('--sample_type', type=str, default='16_frames_conse_rand')
        parser.add_argument('--resize_to', type=int, default=224)

    elif temp_args.model_name == "late_fusion":
        parser = LateFusion.add_model_specific_args(parser)
        parser.add_argument('--sample_type', type=str, default='5_frames_uniform')
        parser.add_argument('--resize_to', type=int, default=256)   # 5 uni

    return parser.parse_args()


def extract_features_from_video(video_path, type, transform, *args, **kwargs):
    """
    return a stack of torch based on type
    """
    frames = []
    # Read the Video
    video_reader = cv2.VideoCapture(video_path)

    frame_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    if type == "5_frames_uniform":
        if frame_count < 5:
            raise NotImplementedError

        # Calculate the interval after which frames will be added to the list
        skip_interval = max(int(frame_count / 5), 1)

        for counter in range(5):
            # Set the current frame postion of the video
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, counter * skip_interval)
            # Read the current frame
            ret, frame = video_reader.read()
            if not ret:
                break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(image=img)['image'])

    elif type == "16_frames_conse_rand":
        if frame_count < 16:
            raise NotImplementedError

        start_point = random.randint(0, frame_count - 16 - 1)

        for counter in range(16):
            # Set the current frame postion of the video
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, start_point + counter)
            # Read the current frame
            ret, frame = video_reader.read()
            if not ret:
                break

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(image=img)['image'])
    else:
        raise("Not supporting this sample type.")

    video_reader.release()
    return torch.stack(frames, dim=0)


def load_model(dataset_1, dataset_2, model_name, **kwargs):
    # Get NUM_CLASSES
    # Get NUM_CLASSES
    NUM_CLASSES = 0
    if dataset_1 == 'hmdb51':
        NUM_CLASSES_1 = 51

    else:
        NUM_CLASSES_1 = 101

    if dataset_2 == 'hmdb51':
        NUM_CLASSES_2 = 51
    else:
        NUM_CLASSES_2 = 101

    if model_name == "c3d":
        net = C3D(**kwargs,
                  n_class_1=NUM_CLASSES_1, n_class_2=NUM_CLASSES_2)

    elif model_name == "non_local":
        net = NonLocalI3Res(**kwargs,
                            num_classes_1=NUM_CLASSES_1, num_classes_2=NUM_CLASSES_2)

    elif model_name == "late_fusion":
        net = LateFusion(**kwargs,
                         n_class_1=NUM_CLASSES_1,
                         n_class_2=NUM_CLASSES_2)

    return net


def predict_model(model, features, first_dataset, args):

    model.eval()
    with torch.no_grad():
        clip = features.to(args.device, non_blocking=True)
        logit = model(clip.unsqueeze(0), first_dataset).squeeze()
        output = torch.softmax(logit, dim=0)

    return output.detach().cpu().numpy()


def predict(video_path, weight_path, dataset_1, dataset_2, model_name, sample_type, resize_to, first_dataset, **kwargs):

    utils.seed_everything(seed=73)

    device = utils.get_training_device()

    class Args():
        def __init__(self, sample_type, resize_to):
            self.sample_type = sample_type
            self.resize_to = resize_to
            self.device = device

    args = Args(sample_type, resize_to)

    net = load_model(dataset_1, dataset_2, model_name, **kwargs)

    net.to(device)

    utils.load_checkpoint(weight_path, net)

    transform = utils.get_transforms(args)['test_transforms']

    features = extract_features_from_video(video_path, args.sample_type, transform)

    output = predict_model(net, features, first_dataset, args)

    return output


if __name__ == '__main__':

    args = get_arg_parser()
    dict_args = vars(args)
    output = predict(**dict_args)
    print(output)
    print(output.argmax())
