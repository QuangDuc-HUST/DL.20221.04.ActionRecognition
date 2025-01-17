#
#
# Supporting functions for the project
#
#

from albumentations.pytorch import ToTensorV2
import os
import json
import random
import shutil
import subprocess
import pandas as pd
import albumentations as A
import cv2
import numpy as np
import torch
import wandb


def get_transforms(args):

    train_transforms = A.ReplayCompose(
        [
            # A.RandomResizedCrop(args.resize_to, args.resize_to, interpolation=cv2.INTER_CUBIC),
            A.Resize(args.resize_to, args.resize_to, interpolation=cv2.INTER_CUBIC),
            # A.HorizontalFlip(p=0.2),
            # A.VerticalFlip(p=0.2),
            # A.ColorJitter(p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ])

    val_transforms = A.ReplayCompose(
        [
            A.Resize(args.resize_to, args.resize_to, interpolation=cv2.INTER_CUBIC),
            # A.CenterCrop(args.resize_to, args.resize_to),
            A.Normalize(),
            ToTensorV2(),
        ])

    test_transforms = A.ReplayCompose(
        [
            # A.CenterCrop(args.resize_to, args.resize_to),
            A.Resize(args.resize_to, args.resize_to, interpolation=cv2.INTER_CUBIC),
            A.Normalize(),
            ToTensorV2(),
        ])

    return {'train_transforms': train_transforms,
            'val_transforms': val_transforms,
            'test_transforms': test_transforms}


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


class WandbLogger():
    def __init__(self, args):
        self.args = args

        import wandb
        self._wandb = wandb

        # Initialise a W&B run
        if self._wandb.run is None:

            self._wandb.login(anonymous='must')
            self._wandb.init(
                project=args.project,
                notes=args.notes,
                config=args
            )

    def set_steps(self):

        self._wandb.define_metric('trainer/global_step')

        self._wandb.define_metric('train/*', step_metric='trainer/global_step')
        self._wandb.define_metric('val/*', step_metric='trainer/epoch')

    def log_checkpoints(self):

        print("Uploading checkpoints to wandb ...")

        ckp_dir = self.args.ckp_dir

        model_artifact = self._wandb.Artifact(
            self._wandb.run.id + '_model', type='model'
        )

        model_artifact.add_dir(ckp_dir)

        self._wandb.log_artifact(model_artifact, aliases=["latest"])

    def log_info(self):

        log_dir = self.args.ckp_dir

        if not os.path.exists(log_dir):
            print("Checkpoint Directory does not exist! Making directory {}".format(log_dir))
            os.makedirs(log_dir)

        wandb_log_json_path = os.path.join(log_dir, 'wandb_info.json')

        dict_info = {'id': self._wandb.run.id,
                     'path': self._wandb.run.path,
                     'url': self._wandb.run.url,
                     'artifact_path': self._wandb.run.path + '_model:latest'   # Update the lastest :D
                     }
        # Write
        with open(wandb_log_json_path, 'w') as f:
            json.dump(dict_info, f, indent=4)

    @staticmethod
    def save_metrics(metrics, args):
        """
        Update the run summary and upload test metrics into the artifact
        """
        # get wandb_info
        with open(os.path.join(args.ckp_dir, 'wandb_info.json')) as f:
            wandb_info = json.load(f)

        # get API
        api = wandb.Api()
        # get run
        wandb_run = api.run(wandb_info['path'])

        for metric, value in metrics.items():
            wandb_run.summary[f'test/{metric}'] = value

        wandb_run.summary.update()

    @staticmethod
    def save_file_artifact(project_name, file_path, artifact_type, args):

        with open(os.path.join(args.ckp_dir, 'wandb_info.json')) as f:
            wandb_info = json.load(f)

        import wandb

        with wandb.init(project=project_name) as run:
            artifact_name = str(wandb_info['id'])
            artifact = wandb.Artifact(artifact_name, artifact_type)  # default artifact_name = run.id
            artifact.add_file(file_path)
            run.log_artifact(artifact)
            run.finish()
            path = run.path

        api = wandb.Api()
        run = api.run(path)
        run.delete()

        print(f"Save {file_path} to {artifact_name} in project {project_name}")
        print(f"Delete the run {path} after creating the artifact.")
        print('-' * 20)


def get_map_id_to_label(dataset_name):
    if dataset_name == "hmdb51":
        annotation_path = './data/HMDB51/annotation/video_class_to_label.csv'
    elif dataset_name == "ucf101":
        annotation_path = './data/UCF101/annotation/video_class_to_label.csv'

    df = pd.read_csv(annotation_path)
    df = df.sort_values(by="label_id")

    id_to_label = {}
    label_to_id = {}

    for _, row in df.iterrows():
        id_to_label[row['label_id']] = row['video_class_id']
        label_to_id[row['video_class_id']] = row['label_id']

    return id_to_label, label_to_id


def runcmd(cmd, is_wait=False, *args, **kwargs):
    # function for running command
    process = subprocess.Popen(
        cmd,
        text=True,
        shell=True
    )

    if is_wait:
        process.wait()


def seed_everything(seed=73):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def save_dict_to_json(d, json_path):

    with open(json_path, 'w') as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth'. If is_best==True, also saves
    checkpoint + 'best.pth'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
            (epoch, state_dict, optimizer)
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    file_path = os.path.join(checkpoint, 'last.pth')

    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.makedirs(checkpoint)

    print(f"Saving checkpoint...")
    torch.save(state, file_path)

    if is_best:
        shutil.copyfile(file_path, os.path.join(checkpoint, 'best.pth'))


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

    print(f"Load checkpoint from {checkpoint}")

    checkpoint = torch.load(checkpoint)

    model.load_state_dict(checkpoint['state_dict'])  # maybe epoch as well

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def get_training_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("We are on", device, "..")
    return device


def acc_metrics(preds, targets):
    return (preds.argmax(1) == targets).sum() / preds.shape[0]


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
