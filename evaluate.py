#
# For evaluate valset and test set
#

from tqdm import tqdm

import torch

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