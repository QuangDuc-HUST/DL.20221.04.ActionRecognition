#
# main file 
# 

import os
import time
import argparse
from tqdm import tqdm

import wandb

import numpy as np

import torch
from torch import nn

from model.lrcn import LRCN
from model.data_loader import ActionRecognitionDataWrapper 
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

    # Module specific args
    parser = LRCN.add_model_specific_args(parser)

    # hyperparameters specific args
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sl_gammar', type=float, default=0.999)


    return parser.parse_args()


def fit(epochs, model, train_loader, val_loader, criterion,  optimizer, scheduler, wandb_init):
        
    # wandb
    wandb.login(anonymous="must")

    wandb.init(**wandb_init)

    torch.cuda.empty_cache()
    
    train_losses = []
    val_losses = []
    val_acces = []
    lrs = []
    
    min_loss = np.inf

    decrease = 1 ; not_improve=0

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        
        running_loss = 0
        #training loop
        model.train()
        for _, data in enumerate(tqdm(train_loader)):
            #training phase
            image, label = data
            
            image = image.to(device); label = label.to(device);
            
            #forward
            output = model(image)
            loss = criterion(output, label)
            
            #backward
            loss.backward()
            optimizer.step() #update weight          
            optimizer.zero_grad() #reset gradient
            
            running_loss += loss.item()
            
        else:
            model.eval()
            val_loss = 0
            val_acc = 0
            #validation loop
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    image, label= data
                    
                    image = image.to(device); label = label.to(device);
                    
                    output = model(image)
                    #loss
                    loss = criterion(output, label)
                    #evaluation metrics
                    val_acc += acc_metrics(output, label)
                    
                    val_loss += loss.item()
            
            #step scheduler
            lrs.append(get_lr(optimizer))
            
            scheduler.step() 
            
            #calculati mean for each batch
            
            train_losses.append(running_loss/len(train_loader))
            val_losses.append(val_loss/len(val_loader))


            if min_loss > (val_loss/len(val_loader)):
                print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (val_loss/len(val_loader))))
                min_loss = (val_loss/len(val_loader))
                decrease += 1
                if decrease % 3 == 0:
                    print('saving model...')
                    torch.save(model, 'LCRNs-{:.3f}.pt'.format(val_acc/len(val_loader)))
                    

            if (val_loss/len(val_loader)) > min_loss:
                not_improve += 1
                min_loss = (val_loss/len(val_loader))
                print(f'Loss Not Decrease for {not_improve} time')
                if not_improve == 20:
                    print('Loss not decrease for 20 times, Stop Training')
                    break
            
            val_acces.append(val_acc/len(val_loader))
            
            print("Epoch:{}/{}..".format(e+1, epochs),
                  "Learning Rate: {}".format(lrs[-1]),
                  "Train Loss: {:.3f}..".format(running_loss/len(train_loader)),
                  "Val Loss: {:.3f}..".format(val_loss/len(val_loader)),
                  "Val Acc: {:.3f}..".format(val_acc/len(val_loader)),
                  "Time: {:.2f}m".format((time.time()-since)/60))
            
            wandb.log(
            {
             "Epoch": e + 1,
             "Learning Rate": lrs[-1],
             "Train Loss": running_loss/len(train_loader),
             "Val Loss": val_loss/len(val_loader),
             "Val Acc" :val_acc/len(val_loader),
            })
            
    history = {'train_loss' : train_losses, 
               'val_loss': val_losses, 
               'val_acc':val_acces,
               'lrs': lrs}
    ##
   
    
    print('Total time: {:.2f} m' .format((time.time()- fit_time)/60))
    ##
    wandb.finish()
    return history


if __name__ == '__main__':

    # Get parser argument
    args = get_arg_parser()
    dict_args = vars(args)

    # set everything
    seed_everything(seed=73)

    # get training device
    device = get_training_device()

    # Get data wrapper
    data_wrapper = ActionRecognitionDataWrapper(**dict_args, 
                                                transforms=get_transforms())
    # Get model 
    if args.dataset == 'hmdb51':
        NUM_CLASSES = 51
    else:
        NUM_CLASSES = 101

    net = LRCN(**dict_args,
               n_class=NUM_CLASSES)
        
    # Loss functions
    criterion = nn.CrossEntropyLoss()

    # Optimizer vs scheduler
    optimizer = torch.optim.Adam(params = net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = args.sl_gammar)

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


    # Training
    fit(epochs=args.max_epochs, 
              model=net, 
              train_loader=data_wrapper.get_train_dataloader(), 
              val_loader=data_wrapper.get_val_dataloader(), 
              criterion=criterion, 
              optimizer=optimizer, 
              scheduler=scheduler, 
              wandb_init=wandb_init)


