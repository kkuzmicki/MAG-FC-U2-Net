import argparse
import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from pathlib import Path
import tqdm
import json
import utils
#import sklearn.preprocessing
import numpy as np
import random
import os
import copy
import math
#import matplotlib.pyplot as plt
from u2net import u2net
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed
# python -m torch.distributed.launch --nproc_per_node=4 train.py

import warnings
warnings.filterwarnings('ignore')

tqdm.monitor_interval = 0
torch.backends.cudnn.benchmark = True # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936

# 1. Change mode to 'train'
# 2. Get 8,12seconds fragmets from song in waveform format
# 3. Turn this waveforms to spectrograms
# 4. mask???
# 5. Let model give output from spectrogram (X) on the input
# 6. For this output, loss function's value is calculated
# 7. This loss function's value is given to scaler
# 8. Scaler gives it to optimizer
# 9. Optimizer changes weights etc.
# 10. Loss value is saved
# 11. Repeat for each song multiple times
# 12. At the end of each epoch, this function returns average of all loss values from this epoch
def train(args, model, device, train_loader, optimizer, scaler):
    losses = []
    model.train() # changes model's mode to 'train' - some layers behave differently while training from while testing

    # x - mixture (all tracks summed up) | y - vocal
    for x, y in tqdm.tqdm(train_loader):

        x, y = x.to(device), y.to(device) # train.py: x >>> torch.Size([12, 2, 130560]) y >>> torch.Size([12, 2, 130560])

        X = utils.Spectrogram(utils.STFT(x, device, args.fft, args.hop)) # train.py: torch.Size([12, 2, 513, 256])
        Y = utils.Spectrogram(utils.STFT(y, device, args.fft, args.hop)) # train.py: torch.Size([12, 2, 513, 256])
        X = X[:,:,:args.bins,:] # train.py: torch.Size([12, 2, 513, 256]) (arg.bins by default is 513) IF IN STFT RETURN_COMPLEX=FALSE, THEN torch.Size([12, 2, 513, 256])
        Y = Y[:,:,:args.bins,:] # train.py: torch.Size([12, 2, 513, 256])
        
        mask = torch.ones_like(Y).to(device) # Returns a tensor filled with the scalar value 1, with the same size as input # train.py: torch.Size([12, 2, 513, 256])
        mask[Y*10/5<X] = 0.0 # torch.Size([12, 2, 513, 256])
        optimizer.zero_grad() # Sets the gradients of all optimized torch. Tensor s to zero.

        # print(X.shape)
        with torch.cuda.amp.autocast():

            Y_est, Y_mask = model(X) # train.py: Y_est=torch.Size([12, 2, 513, 256]) Y_mask=torch.Size([12, 2, 513, 256])
            loss = F.mse_loss(Y_est, Y) + F.binary_cross_entropy_with_logits(Y_mask, mask) # train.py: tensor(1.5345, device='cuda:0', grad_fn=<AddBackward0>)
            # F.mse_loss(Y_est*F.sigmoid(Y_mask), Y*mask) # psnr

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update() # each batch (default=12) changes weights - NOT EACH EPOCH (as I thought)

        losses.append(loss.item())

        # break
    return np.array(losses).mean()

def get_parser():
    parser = argparse.ArgumentParser(description='Music Separation Training')
    parser.add_argument('--local_rank', type=int, default=0) # ??? https://medium.com/red-buffer/getting-started-with-pytorch-distributed-54ae933bb9f0 ???
    parser.add_argument('--nprocs', type=int) # not used
    parser.add_argument('--root', type=str, default='DATASET_16kHz_2channels') # source of training data

    parser.add_argument('--model', type=str, default="models/musdb16_model_first_v4") # name for model (may already exist)
    parser.add_argument('--pretrained', dest='pretrained', action='store_true') # creates new model
    #parser.add_argument('--pretrained', type=bool, default='true') # uses saved model
    parser.add_argument('--target', type=str, default='vocals') # what source to extract

    parser.add_argument('--sample-rate', type=int, default=16000) # sample rate of source songs
    parser.add_argument('--channels', type=int, default=2) # number of channels of source songs
    parser.add_argument('--samples-per-track', type=int, default=100) # how many fragments per each song

    # Training Parameters
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch-size', type=int, default=12) # originally 12
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=30) # originally 80
    parser.add_argument('--weight-decay', type=float, default=0.00001)
    parser.add_argument('--seed', type=int, default=42, metavar='S')

    # Model Parameters
    parser.add_argument('--dur', type=int, default=256)

    parser.add_argument('--fft', type=int, default=1024)
    parser.add_argument('--hop', type=int, default=512)
    parser.add_argument('--bins', type=int, default=None)

    parser.add_argument('--nb-workers', type=int, default=12) # default=12; 6 or 12 doesn't make difference, 12< is getting worse and worse; 
                                                             # https://stackoverflow.com/questions/53998282/how-does-the-number-of-workers-parameter-in-pytorch-dataloader-actually-work
    
    parser.add_argument('--device', type=str, default="cuda")

    return parser

def main():

    # torch.use_deterministic_algorithms(True) # uncomment if want to start deterministic mode

    parser = get_parser()
    args = parser.parse_args()
    
    if args.bins is None:
        args.bins = args.fft//2 + 1

    device = torch.device("cuda")

    model_path = Path(args.model)
    model_path.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed) # sets seed for Python's random
    torch.manual_seed(args.seed) # sets seed for torch's random (both CPU and CUDA)

    train_dataset = data.load_datasets(args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.nb_workers, pin_memory=False
    ) # this shuffle works with torch's random - seed

    model = u2net(2,2,args.bins).to(device)
    # model = myModel(args).to(device)

    # https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-deep-learning-optimizers/
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.3)
    scaler = torch.cuda.amp.GradScaler()
    es = utils.EarlyStopping(patience=args.patience)

    if args.pretrained:
        with open(Path(args.model, args.target + '.json'), 'r') as stream:
            results = json.load(stream)

        target_model_path = Path(args.model, args.target + ".chkpnt")
        checkpoint = torch.load(target_model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
 
        epochs = range(results['epochs_trained']+1, args.epochs + 1)
        train_losses = results['train_loss_history']
        best_epoch = results['best_epoch']
        es.best = results['best_loss']
        es.num_bad_epochs = results['num_bad_epochs']
    else:
        epochs = range(1, args.epochs + 1)
        train_losses = []
        best_epoch = 0

    pbar = tqdm.tqdm(epochs)
    for epoch in pbar:

        train_loss = train(args, model, device, train_loader, optimizer, scaler)
        scheduler.step()

        train_losses.append(train_loss)
        pbar.set_postfix(loss=train_loss,lr=optimizer.state_dict()['param_groups'][0]['lr'])

        stop = es.step(train_loss)
        best_epoch = epoch if train_loss == es.best else best_epoch

        if args.local_rank == 0:
            utils.save_checkpoint({
                'epoch': epoch ,
                'state_dict': model.state_dict(), # simply a Python dictionary object that maps each layer to its parameter tensor
                'best_loss': es.best,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
            },
                is_best = train_loss == es.best,
                path=Path(args.model),
                target=args.target
            )

            # save params
            params = {
                'epochs_trained': epoch,
                'args': vars(args),
                'best_loss': es.best,
                'best_epoch': best_epoch,
                'train_loss_history': train_losses,
                'num_bad_epochs': es.num_bad_epochs
            }

            with open(Path(args.model, args.target + '.json'), 'w') as outfile:
                outfile.write(json.dumps(params, indent=4, sort_keys=True))

        if stop:
            print("Apply Early Stopping")

            break

if __name__ == "__main__":
    main()
