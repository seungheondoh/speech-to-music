import argparse
import builtins
import math
import os
import json
from tqdm import tqdm
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
from stm.regression.music_reg.loader.dataloader import get_dataloader
from stm.regression.music_reg.model import MusicModel
from stm.utils.eval_utils import load_pretrained, get_r2

from stm.utils.train_utils import Logger, AverageMeter, ProgressMeter, EarlyStopping, save_hparams, print_model_params

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from stm.constants import (CLSREG_DATASET, AUDIOSET_TAGS, SPEECH_SAMPLE_RATE)

def save_cm(predict, label, label_name, save_path):
    predict_ = [label_name[i] for i in predict]
    label_ = [label_name[i] for i in label]
    cm = confusion_matrix(label_, predict_, labels=label_name)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_name)
    disp.plot(xticks_rotation="vertical")
    plt.savefig(os.path.join(save_path, 'cm.png'), dpi=150)


parser = argparse.ArgumentParser(description='PyTorch MSD Training')
parser.add_argument('--dataset_type', type=str, default="Audioset")
parser.add_argument('--modality', type=str, default="audio")
parser.add_argument('-j', '--workers', default=24, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup_epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--min_lr', default=1e-9, type=float)
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU id to use.')
parser.add_argument('--print_freq', default=50, type=int)
# train detail


def main():
    args = parser.parse_args()
    main_worker(args)

def main_worker(args):
    test_loader = get_dataloader(args, 'TEST')
    model = MusicModel(pretrained_path=os.path.join(CLSREG_DATASET, "pretrained/music/compact_student.ckpt"))
    save_dir = f"exp/{args.dataset_type}/{args.modality}_{args.batch_size}_{args.lr}"
    model = load_pretrained(save_dir, model)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    model.eval()

    fnames, predictions, ground_truth = [], [], []
    for batch in tqdm(test_loader):
        x = batch['audio']
        y = batch['va']
        if args.gpu is not None:
            x = x.cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            logit = model.infer(x)
        predictions.append(logit.detach().cpu())
        ground_truth.append(y)
    predictions = torch.cat(predictions, dim=0).numpy()
    ground_truth = torch.cat(ground_truth, dim=0).numpy()
    r2, rv ,va = get_r2(ground_truth, predictions)
    results = {
        "r2":r2, 
        "valence":rv,
        "arousal":va
    }
    with open(os.path.join(save_dir, "results.json"), mode="w") as io:
        json.dump(results, io, indent=4)

if __name__ == '__main__':
    main()