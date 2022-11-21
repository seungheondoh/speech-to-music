import argparse
import builtins
import math
from operator import is_
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
from stm.metric_learning.loader.dataloader import get_dataloader
from stm.metric_learning.model import EmbModel
from stm.utils.train_utils import Logger, AverageMeter, ProgressMeter, EarlyStopping, save_hparams, print_model_params

parser = argparse.ArgumentParser(description='PyTorch MSD Training')
parser.add_argument('--speech_type', type=str, default="IEMOCAP")
parser.add_argument('--audio_type', type=str, default="Audioset")
parser.add_argument('--modality', type=str, default="audio")
parser.add_argument('--framework', type=str, default="triplet")
parser.add_argument('--tid', type=str, default="0")
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup_epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--min_lr', default=1e-9, type=float)
parser.add_argument('--emo_lambda', default=1, type=float)
parser.add_argument('--is_ranksim', default=0, type=int)
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU id to use.')
parser.add_argument('--print_freq', default=50, type=int)

def main():
    args = parser.parse_args()
    main_worker(args)

def main_worker(args):
    train_loader = get_dataloader(args, 'TRAIN')
    val_loader = get_dataloader(args, 'VALID')
    model = EmbModel(
        modality = args.modality, 
        framework = args.framework,
        is_ranksim = args.is_ranksim,
        emo_lambda = args.emo_lambda
    )
    print_model_params(args, model)
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        args.lr,
    )
    earlystopping_callback = EarlyStopping(tolerance=5)
    cudnn.benchmark = True

    save_dir = f"exp/{args.audio_type}_{args.speech_type}/{args.modality}_{args.framework}_{args.is_ranksim}_{args.emo_lambda}_{args.tid}"
    logger = Logger(save_dir)
    save_hparams(args, save_dir)

    best_val_loss = np.inf
    for epoch in range(0, args.epochs):
        # train for one epoch
        train(train_loader, model, optimizer, epoch, logger, args)
        val_loss, val_stm_loss = validate(val_loader, model, epoch, args)
        logger.log_val_loss(val_loss, epoch)
        logger.log_val_stm_loss(val_stm_loss, epoch)
        # logger.log_val_emo_loss(val_emo_loss, epoch)
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
        s_emb = batch["s_emb"].cuda(args.gpu, non_blocking=True)
        s_tag_emb = batch["s_tag_emb"].cuda(args.gpu, non_blocking=True)
        s_va = batch["s_va"].cuda(args.gpu, non_blocking=True)
        s_binary = batch["s_binary"].cuda(args.gpu, non_blocking=True)
        m_emb = batch["m_emb"] .cuda(args.gpu, non_blocking=True)
        m_tag_emb = batch["m_tag_emb"].cuda(args.gpu, non_blocking=True)
        m_va = batch["m_va"].cuda(args.gpu, non_blocking=True)
        m_binary = batch["m_binary"].cuda(args.gpu, non_blocking=True)
        # compute output
        loss, _, _ = model(s_emb, s_tag_emb, s_va, s_binary, m_emb, m_tag_emb, m_va, m_binary)
        train_losses.step(loss.item(), s_emb.size(0))
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
    epoch_end_loss, epoch_end_stm_loss, epoch_end_emo_loss = [], [], []
    for data_iter_step, batch in enumerate(val_loader):
        s_emb = batch["s_emb"].cuda(args.gpu, non_blocking=True)
        s_tag_emb = batch["s_tag_emb"].cuda(args.gpu, non_blocking=True)
        s_va = batch["s_va"].cuda(args.gpu, non_blocking=True)
        s_binary = batch["s_binary"].cuda(args.gpu, non_blocking=True)
        m_emb = batch["m_emb"] .cuda(args.gpu, non_blocking=True)
        m_tag_emb = batch["m_tag_emb"].cuda(args.gpu, non_blocking=True)
        m_va = batch["m_va"].cuda(args.gpu, non_blocking=True)
        m_binary = batch["m_binary"].cuda(args.gpu, non_blocking=True)
        # compute output
        loss, stm_loss, _ = model(s_emb, s_tag_emb, s_va, s_binary, m_emb, m_tag_emb, m_va, m_binary)
        epoch_end_loss.append(loss.detach().cpu())
        epoch_end_stm_loss.append(stm_loss.detach().cpu())
        losses_val.step(loss.item(), s_emb.size(0))
        if data_iter_step % args.print_freq == 0:
            progress_val.display(data_iter_step)
    val_loss = torch.stack(epoch_end_loss).mean(0, False)
    val_stm_loss = torch.stack(epoch_end_stm_loss).mean(0, False)
    return val_loss, val_stm_loss

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