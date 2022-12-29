#
# main file 
# 

import os
import argparse
from tqdm import tqdm
import random

import torch
from torch import nn

from model.lrcn import LRCN
from model.c3d import C3D
from model.i3d import I3D
from model.non_local_i3res import NonLocalI3Res
from model.late_fusion_2_dataset import LateFusion

from model.data_loader_2_dataset import ActionRecognitionDataWrapper 

from evaluate_2_dataset import val_evaluate

import utils
from utils import seed_everything, get_training_device, acc_metrics, get_lr, get_transforms


def get_arg_parser():
    """
    Get options from CLI
    """
    parser = argparse.ArgumentParser()

    # PROGRAM level args
    parser.add_argument('--data_dir_1', type=str, required=True)
    parser.add_argument('--data_dir_2', type=str, required=True)

    parser.add_argument('--ckp_dir', type=str, default='./ckp/baseline')

    # DataModule specific args
    parser.add_argument('--dataset_1', type=str, required=True, 
                        choices=['hmdb51', 'ucf101'])

    parser.add_argument('--dataset_2', type=str, required=True, 
                    choices=['hmdb51', 'ucf101'])

    parser.add_argument('--data_split', type=str, default='split1', 
                        choices=['split1', 'split2', 'split3'])

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--clip_per_video', type=int, default=1)

    # hyperparameters specific args
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sl_gammar', type=float, default=0.999)

    # Module specific args
    ## which model to use
    parser.add_argument('--model_name', type=str, required=True, 
                        choices=["lrcn", "c3d", "i3d", "non_local", "late_fusion"])

    ## Get the model name now 
    temp_args, _ = parser.parse_known_args()


    if temp_args.model_name == "late_fusion":
        parser = LateFusion.add_model_specific_args(parser)
        parser.add_argument('--resize_to', type=int, default=256)   # 5 uni

    elif temp_args.model_name == "lrcn":
        parser = LRCN.add_model_specific_args(parser)
        # Data transform
        parser.add_argument('--resize_to', type=int, default=256)   # 5 uni

    elif temp_args.model_name == "c3d":
        parser = C3D.add_model_specific_args(parser)
        # Data transform
        parser.add_argument('--resize_to', type=int, default=112)

    elif temp_args.model_name == "i3d":
        parser = I3D.add_model_specific_args(parser)

         # Data transform
        parser.add_argument('--resize_to', type=int, default=224)   

    elif temp_args.model_name == "non_local":
        parser = NonLocalI3Res.add_model_specific_args(parser)
        parser.add_argument('--resize_to', type=int, default=224) 
    


    # Wandb specific args
    parser.add_argument('--enable_wandb', action='store_true')
    parser.add_argument('--project', type=str, default="dl_action_recognition")
    parser.add_argument('--notes', type=str, default='')
    parser.add_argument('--wandb_ckpt', action='store_true')
    parser.add_argument('--save_loss_steps', type=int, default=10)
    parser.add_argument('--save_ckp_epochs', type=int, default=0)   #default is not 


    return parser.parse_args()


def train(model, train_loader_1, train_loader_2, criterion, optimizer, scheduler, wandb_logger, start_steps, args):
    """
    One model training loop
    """
    model.train() 
    loss_avg = utils.RunningAverage()
    len_train_loader_1 = len(train_loader_1)
    len_train_loader_2 = len(train_loader_2)
    propr = len_train_loader_1 / (len_train_loader_1 + len_train_loader_2)

    iter_train_loader_1 = iter(train_loader_1)
    iter_train_loader_2 = iter(train_loader_2)

    with tqdm(total=len_train_loader_1 + len_train_loader_2) as t:

        for step in range(len_train_loader_1 + len_train_loader_2):
            it = start_steps + step
            if (random.random() < propr):
                try:
                    print("Dataset 1")
                    is_first_dataset = True
                    image, label = next(iter_train_loader_1)
                except StopIteration:
                    print("Stop iteration Dataset 1, Dataset 2")
                    is_first_dataset = False
                    image, label = next(iter_train_loader_2)
            
            else:
                try: 
                    print("Dataset 2")
                    is_first_dataset = False
                    image, label = next(iter_train_loader_2)
                except StopIteration:
                    print("Stop iteration Dataset 2, Dataset 1")
                    is_first_dataset = True
                    image, label = next(iter_train_loader_1)

            image = image.to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True)
            
            #forward
            output = model(image, is_first_dataset)
            loss = criterion(output, label)
            
            #backward
            loss.backward()
            optimizer.step() #update weight          
            optimizer.zero_grad() #reset gradient
        
            loss_avg.update(loss.item())

            if wandb_logger and it  % args.save_loss_steps == 0:

                wandb_logger._wandb.log({'train/loss': loss_avg()}, commit=False)
                wandb_logger._wandb.log({'train/lr': get_lr(optimizer)}, commit=False)
                wandb_logger._wandb.log({'trainer/global_step': it})

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    current_lr = get_lr(optimizer)

    #step scheduler   
    scheduler.step() 


    print("Learning Rate: {}..".format(current_lr),
          "Train Loss: {:.3f}..".format(loss_avg()))


def train_and_valid(epochs, model, train_loader_1, train_loader_2, val_loader_1, val_loader_2, criterion,  optimizer, scheduler, ckp_dir, wandb_logger, args):
    """
    Train and valid process including many epochs
    """
    if wandb_logger:
        wandb_logger.log_info() # Get wandb info
        wandb_logger.set_steps()

    best_val_acc_1 = 0.0 

    for epoch in range(epochs):
        
        print("Epoch {}/{}".format(epoch + 1, epochs))
        
        train(model, train_loader_1, train_loader_2, criterion, optimizer, scheduler, wandb_logger, epoch * (len(train_loader_1) + len(train_loader_2)), args)

        val_loss_1, val_acc_1 = val_evaluate(model, val_loader_1, criterion, acc_metrics, True, args)

        val_loss_2, val_acc_2=  val_evaluate(model, val_loader_2, criterion, acc_metrics, False, args)

        if wandb_logger:
            wandb_logger._wandb.log({
                                    'val/loss_1': val_loss_1,
                                    'val/acc_1': val_acc_1,
                                    'val/loss_2': val_loss_2,
                                    'val/acc_2': val_acc_2,
                                    'trainer/epoch': epoch + 1,
                                     },
                                     commit=False) # Let's the train loop commit

        # Logging
        is_best = val_acc_1 >= best_val_acc_1
        if is_best:
            print("- Found new best accuracy performance on val 1")

        # Checkpoint saving
        utils.save_checkpoint({ 'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'optim_dict': optimizer.state_dict(),},
                                is_best=is_best,
                                checkpoint=ckp_dir)

        # Save the best
        if is_best:
            best_val_acc_1 = val_acc_1

            b_json_path = os.path.join(ckp_dir, 'metrics_val_best_weights.json')
            utils.save_dict_to_json({'val_acc':val_acc_1}, b_json_path)

        # Save the last
        l_json_path = os.path.join(ckp_dir, 'metrics_val_last_weights.json')
        utils.save_dict_to_json({'val_acc':val_acc_1}, l_json_path) 

        # Upload checkpoint periodically
        if wandb_logger and args.save_ckp_epochs and not epoch % args.save_ckp_epochs:
            wandb_logger.log_checkpoints()


    #Finish the train and val loop and save artifact wandb
    if wandb_logger and args.wandb_ckpt and args.ckp_dir:
        wandb_logger._wandb.log({})  # Commit the last
        
        wandb_logger.log_checkpoints()

if __name__ == '__main__':

    # Get parser argument
    args = get_arg_parser()
    # get training device
    args.device = get_training_device()
    dict_args = vars(args)

    # set everything
    seed_everything(seed=73)


    # Get data wrapper
    data_wrapper = ActionRecognitionDataWrapper(**dict_args, 
                                                transforms=get_transforms(args))
    
    # Get NUM_CLASSES
    NUM_CLASSES = 0
    if args.dataset_1 == 'hmdb51':
        NUM_CLASSES_1 = 51
    else:
        NUM_CLASSES_1 = 101
    
    if args.dataset_2 == 'hmdb51':
        NUM_CLASSES_2 = 51
    else:
        NUM_CLASSES_2 = 101




    # Get model 
    if args.model_name == "late_fusion":
        net = LateFusion(**dict_args,n_class_1=NUM_CLASSES_1, n_class_2=NUM_CLASSES_2)
    elif args.model_name == "lrcn":
        net = LRCN(**dict_args,
                    n_class=NUM_CLASSES)
    elif args.model_name == "c3d":
        net = C3D(**dict_args,
                    n_class=NUM_CLASSES)
    elif args.model_name == "i3d":
        net = I3D(**dict_args,
                    num_classes=NUM_CLASSES)
    elif args.model_name == "non_local":
        net = NonLocalI3Res(**dict_args,
                            num_classes=NUM_CLASSES)

    net.to(args.device)
    # Loss functions
    criterion = nn.CrossEntropyLoss()

    # Optimizer vs scheduler
    optimizer = torch.optim.Adam(params = net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = args.sl_gammar)

    # Wandb logger init
    if args.enable_wandb:
        wandb_logger = utils.WandbLogger(args)
    else:
        wandb_logger = None

    # Training
    train_and_valid(epochs=args.max_epochs, 
                    model=net, 
                    train_loader_1=data_wrapper.get_train_1_dataloader(),
                    train_loader_2=data_wrapper.get_train_2_dataloader(), 
                    val_loader_1=data_wrapper.get_val_1_dataloader(),
                    val_loader_2=data_wrapper.get_val_2_dataloader(), 
                    criterion=criterion, 
                    optimizer=optimizer, 
                    scheduler=scheduler, 
                    wandb_logger=wandb_logger,
                    ckp_dir=args.ckp_dir,
                    args=args
                    )

    # Finish wandb
    if wandb_logger:
        wandb_logger._wandb.finish()
