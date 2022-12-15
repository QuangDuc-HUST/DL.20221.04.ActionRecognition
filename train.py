#
# main file 
# 

import argparse
from tqdm import tqdm

import wandb

import numpy as np

import torch
from torch import nn

from model.lrcn import LRCN
from model.c3d import C3D
from model.data_loader import ActionRecognitionDataWrapper 

import utils
from utils import seed_everything, get_training_device, acc_metrics, get_lr, get_transforms

def get_arg_parser():

    parser = argparse.ArgumentParser()

    # PROGRAM level args
    parser.add_argument('--data_dir', type=str, required=True)
    # parser.add_argument('--ckp_dir', default='/mnt/ducnq/ckp-mood')

    # DataModule specific args
    parser.add_argument('--dataset', type=str, required=True, choices=['hmdb51', 'ucf101'])
    parser.add_argument('--data_split', type=str, default='split1', choices=['split1', 'split2', 'split3'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)

    # hyperparameters specific args
    parser.add_argument('--save_summary_steps', type=int, default=10) ## Just like pytorch lightning
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sl_gammar', type=float, default=0.999)

    # Module specific args
    ## which model to use
    parser.add_argument('--model_name', type=str, required=True, choices=["lrcn", "c3d"])

    ## Get the model name now 
    temp_args, _ = parser.parse_known_args()

    if temp_args.model_name == "lrcn":
        parser = LRCN.add_model_specific_args(parser)
    elif temp_args.model_name == "c3d":
        parser = C3D.add_model_specific_args(parser)

    return parser.parse_args()



def train_and_valid(epochs, model, train_loader, val_loader, criterion,  optimizer, scheduler, wandb_init, args):
        
    # wandb
    wandb.login(anonymous="must")
  
    wandb.init(**wandb_init)

    torch.cuda.empty_cache()
    
    model.to(device)
        
    for e in range(epochs):

        model.train() 
        loss_avg = utils.RunningAverage()
        with tqdm(total=len(train_loader)) as t:
            for i , data in enumerate(train_loader):
                #training phase
                image, label = data
                
                image = image.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                
                #forward
                output = model(image)
                loss = criterion(output, label)
                
                #backward
                loss.backward()
                optimizer.step() #update weight          
                optimizer.zero_grad() #reset gradient
            
                loss_avg.update(loss.item())

                if i  % args.save_summary_steps:
                    pass

                t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
                t.update()

        model.eval()    
        val_loss = 0
        val_acc = 0
        with tqdm(total=len(val_loader)) as t:
            with torch.no_grad():
                for _, data in enumerate(val_loader):
                    image, label= data
                    
                    image = image.to(device, non_blocking=True) 
                    label = label.to(device, non_blocking=True)
                    
                    output = model(image)
                    #loss
                    loss = criterion(output, label)
                    #evaluation metrics
                    val_acc += acc_metrics(output, label)
                    
                    val_loss += loss.item()

                    t.update()

        #step scheduler   
        current_lr = get_lr(optimizer)

        scheduler.step()       

        
        print("Epoch:{}/{}..".format(e+1, epochs),
                "Learning Rate: {}".format(current_lr),
                "Train Loss: {:.3f}..".format(loss_avg()),
                "Val Loss: {:.3f}..".format(val_loss/len(val_loader)),
                "Val Acc: {:.3f}..".format(val_acc/len(val_loader))
                )
        

        wandb.log(
        {
            "Epoch": e,
            "Learning Rate": current_lr,
            "Train Loss": loss_avg(),
            "Val Loss": val_loss/len(val_loader),
            "Val Acc" :val_acc/len(val_loader),
        })
            

    wandb.finish()

if __name__ == '__main__':

    # Get parser argument
    args = get_arg_parser()
    dict_args = vars(args)

    # set everything
    seed_everything(seed=73)

     # Wandb settings
    WANDB_NOTE = "Test colab"
    wandb_init =  {
                "project":"DL-Recognition", 
                "notes": WANDB_NOTE,
                "config": {
                      "architecture": "LCRN",
                      "optimizers and schedulers": "adam, explr",
                      **dict_args
                      }
    }

    # get training device
    device = get_training_device()

    # Get data wrapper
    data_wrapper = ActionRecognitionDataWrapper(**dict_args, 
                                                transforms=get_transforms())
    
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
           
    # Loss functions
    criterion = nn.CrossEntropyLoss()

    # Optimizer vs scheduler
    optimizer = torch.optim.Adam(params = net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = args.sl_gammar)

    # Training
    train_and_valid(epochs=args.max_epochs, 
              model=net, 
              train_loader=data_wrapper.get_train_dataloader(), 
              val_loader=data_wrapper.get_val_dataloader(), 
              criterion=criterion, 
              optimizer=optimizer, 
              scheduler=scheduler, 
              wandb_init=wandb_init,
              args=args
              )


