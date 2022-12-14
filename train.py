#
# main file 
# 

import os
import time
from tqdm import tqdm

import wandb

import numpy as np

import torch
from torch import nn

from model.lrcn import LRCN
from model.data_loader import ActionRecognitionDataWrapper, get_transforms

from utils import seed_everything


def get_training_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("We are training on", device)
    return device


def acc_metrics(preds, targets):
    return (preds.argmax(1) == targets).sum() / preds.shape[0]
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

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
        for i, data in enumerate(tqdm(train_loader)):
            #training phase
            image, label = data
            
            image = image.to(device); label = label.to(device);
            
            #forward
            model.Lstm.reset_hidden_state()
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
                    
                    model.Lstm.reset_hidden_state()
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

    LR = 1e-4
    NUM_EPOCH = 5
    NUM_WORKERS = 2
    BATCH_SIZE = 64
    SL_GAMMA = 0.999
    NUM_CLASSES = 51
    DATA_FOLDER_PATH = 'data/HMDB51/5_frames_uniform/'

    # set everything
    seed_everything(seed=73)

    # get training device
    device = get_training_device()


    
    data_wrapper = ActionRecognitionDataWrapper(DATA_FOLDER_PATH,
                                                'hmdb51',
                                                'split1',
                                                get_transforms(),
                                                BATCH_SIZE,
                                                NUM_WORKERS)

    net = LRCN(latent_dim=512,
                hidden_size=256,
                lstm_layers=2,
                bidirectional=True,
                n_class=NUM_CLASSES)
        
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params = net.parameters(), lr=LR)
                                
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = SL_GAMMA)

    # Wandb note
    WANDB_NOTE = "Test colab"

    wandb_init =  {
                "project":"DL-Recognition", 
                "notes": WANDB_NOTE,
                "config": {
                      "dataset": "HMDB51",
                      "architecture": "LCRN",
                      "learning_rate": LR ,
                      "epochs": NUM_EPOCH,
                      "batch size": BATCH_SIZE,
                      "optimizers and schedulers": "adam, explr gamma 1",
                      }
    }

    fit(epochs=NUM_EPOCH, 
              model=net, 
              train_loader=data_wrapper.get_train_dataloader(), 
              val_loader=data_wrapper.get_val_dataloader(), 
              criterion=criterion, 
              optimizer=optimizer, 
              scheduler=scheduler, 
              wandb_init={})


