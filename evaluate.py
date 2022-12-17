#
# For evaluate valset and test set
#
import os
import argparse

from tqdm import tqdm

import torch
from torch import nn

from model.lrcn import LRCN
from model.c3d import C3D
from model.data_loader import ActionRecognitionDataWrapper 

import utils
from utils import WandbLogger

def get_arg_parser():

    parser = argparse.ArgumentParser()

    # PROGRAM level args
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--ckp_dir', type=str, default='./ckp/baseline')
    parser.add_argument('--restore_file',type=str, default='best.pth')


    # DataModule specific args
    parser.add_argument('--dataset', type=str, required=True, choices=['hmdb51', 'ucf101'])
    parser.add_argument('--data_split', type=str, default='split1', choices=['split1', 'split2', 'split3'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)


    # Module specific args
    ## which model to use
    parser.add_argument('--model_name', type=str, required=True, choices=["lrcn", "c3d"])

    ## Get the model name now 
    temp_args, _ = parser.parse_known_args()

    if temp_args.model_name == "lrcn":
        parser = LRCN.add_model_specific_args(parser)

        # Data transform
        parser.add_argument('--resize_to', type=int, default=256)   # 5 uni

    elif temp_args.model_name == "c3d":
        parser = C3D.add_model_specific_args(parser)

        # Data transform
        parser.add_argument('--resize_to', type=int, default=128)


    # Wandb specific args
    parser.add_argument('--enable_wandb', action='store_true')

    return parser.parse_args()

def evaluate(model, data_loader, criterion, metrics, args):
    
    model.eval()

    acc_summ = 0
    loss_summ = 0 

    with torch.no_grad():
        with tqdm(total=len(data_loader)) as t:
            for data in data_loader:

                image, label = data
                    
                image = image.to(args.device, non_blocking=True)
                label = label.to(args.device, non_blocking=True)

                #forward
                output = model(image)
                loss = criterion(output, label)

                acc = metrics(output, label)

                acc_summ += acc.item()
                loss_summ += loss.item()

                t.update()

    acc_mean = acc_summ / len(data_loader)
    loss_mean = loss_summ / len(data_loader)
    
    print(f'Eval loss: {loss_mean:05.3f} ... Eval acc: {acc_mean:05.3f}')

    return loss_mean, acc_mean


if __name__ == '__main__':
    """
    Evaluation for test set
    """
    # Get parser argument
    args = get_arg_parser()
    # get training device
    args.device = utils.get_training_device()
    dict_args = vars(args)

    # set everything
    utils.seed_everything(seed=73)


    # Get data wrapper
    data_wrapper = ActionRecognitionDataWrapper(**dict_args, 
                                                transforms=utils.get_transforms(args))
    
    # Get test loader
    test_loader = data_wrapper.get_test_dataloader()

    # Get NUM_CLASSES
    if args.dataset == 'hmdb51':
        NUM_CLASSES = 51
    else:
        NUM_CLASSES = 101

    # Get model 
    if args.model_name == "lrcn":
        net = LRCN(**dict_args,
                    n_class=NUM_CLASSES)
    elif args.model_name == "c3d":
        net = C3D(**dict_args,
                    n_class=NUM_CLASSES)
    net.to(args.device)

    # Loss functions
    criterion = nn.CrossEntropyLoss() 

    # Load weights
    utils.load_checkpoint(os.path.join(args.ckp_dir, args.restore_file), net)

    # Evaluate
    _, test_acc = evaluate(net, test_loader, criterion, utils.acc_metrics, args)
    test_metrics = {'test_acc':test_acc}

    json_path = os.path.join(args.ckp_dir, 'metrics_test.json')
    utils.save_dict_to_json(test_metrics, json_path)
    print(f"Save metrics to {json_path}")


    if args.enable_wandb:
        print("Log to results to wandb summary ...")
        WandbLogger.save_metrics(test_metrics, args)