import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from stm.classification.speech_cls.loader.dataloader import get_dataloader
from stm.classification.speech_cls.model import AudioModel, TextModel
from stm.utils.train_utils import Logger, AverageMeter, ProgressMeter, EarlyStopping, save_hparams, print_model_params

parser = argparse.ArgumentParser(description='PyTorch MSD Training')
parser.add_argument('--dataset_type', type=str, default="IEMOCAP")
parser.add_argument('--modality', type=str, default="audio")
parser.add_argument('-j', '--workers', default=24, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup_epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=2, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--min_lr', default=1e-9, type=float)
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--print_freq', default=50, type=int)
# train detail


def main():
    args = parser.parse_args()
    main_worker(args)

def main_worker(args):
    train_loader = get_dataloader(args, 'TRAIN')
    val_loader = get_dataloader(args, 'VALID')
    if args.modality == "audio":
        model = AudioModel(
            dataset_type=args.dataset_type,
            freeze_type="feature"
        )
    elif args.modality == "text":
        model = TextModel(
            dataset_type=args.dataset_type,
            freeze_type="feature"
        )
    print_model_params(args, model)
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.2
        )
    earlystopping_callback = EarlyStopping(tolerance=20)
    cudnn.benchmark = True

    save_dir = f"exp/{args.dataset_type}/{args.modality}_{args.batch_size}_{args.lr}"
    logger = Logger(save_dir)
    save_hparams(args, save_dir)

    best_val_loss = np.inf
    for epoch in range(0, args.epochs):
        # train for one epoch
        train(train_loader, model, optimizer, epoch, logger, args)
        val_loss = validate(val_loader, model, epoch, args)
        logger.log_val_loss(val_loss, epoch)
        # save model
        if val_loss < best_val_loss:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, f'{save_dir}/best.pth')
            best_val_loss = val_loss

        earlystopping_callback(val_loss, best_val_loss)
        if earlystopping_callback.early_stop:
            print("We are at epoch:", epoch)
            break

def train(train_loader, model, optimizer, epoch, logger, args):
    train_losses = AverageMeter('Train Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),[train_losses],prefix="Epoch: [{}]".format(epoch))
    iters_per_epoch = len(train_loader)
    model.train()
    for data_iter_step, batch in enumerate(train_loader):
        lr = adjust_learning_rate(optimizer, data_iter_step / iters_per_epoch + epoch, args)
        if args.modality == "audio":
            x = batch['audios']
            x_mask = batch['audio_mask']
            y = batch['binarys']
        elif args.modality == "text":
            x = batch['token']
            x_mask = batch['token_mask']
            y = batch['binarys']

        if args.gpu is not None:
            x = x.cuda(args.gpu, non_blocking=True)
            x_mask = x_mask.cuda(args.gpu, non_blocking=True)
            y = y.cuda(args.gpu, non_blocking=True)
        # compute output
        _, loss = model(x, x_mask, y)
        train_losses.step(loss.item(), x.size(0))
        logger.log_train_loss(loss, epoch * iters_per_epoch + data_iter_step)
        logger.log_learning_rate(lr, epoch * iters_per_epoch + data_iter_step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if data_iter_step % args.print_freq == 0:
            progress.display(data_iter_step)

def validate(val_loader, model, epoch, args):
    losses_val = AverageMeter('Valid Loss', ':.4e')
    progress_val = ProgressMeter(len(val_loader),[losses_val],prefix="Epoch: [{}]".format(epoch))
    model.eval()
    epoch_end_loss = []
    for data_iter_step, batch in enumerate(val_loader):
        if args.modality == "audio":
            x = batch['audios']
            x_mask = batch['audio_mask']
            y = batch['binarys']
        elif args.modality == "text":
            x = batch['token']
            x_mask = batch['token_mask']
            y = batch['binarys']
        if args.gpu is not None:
            x = x.cuda(args.gpu, non_blocking=True)
            x_mask = x_mask.cuda(args.gpu, non_blocking=True)
            y = y.cuda(args.gpu, non_blocking=True)
        _, loss = model(x, x_mask, y)
        epoch_end_loss.append(loss.detach().cpu())
        losses_val.step(loss.item(), x.size(0))
        if data_iter_step % args.print_freq == 0:
            progress_val.display(data_iter_step)
    val_loss = torch.stack(epoch_end_loss).mean(0, False)
    return val_loss

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

if __name__ == '__main__':
    main()