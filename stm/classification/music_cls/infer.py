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
from stm.classification.music_cls.loader.dataloader import get_dataloader
from stm.classification.music_cls.model import MusicModel
from stm.utils.eval_utils import load_pretrained, get_scores

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
    test_loader = get_dataloader(args, 'ALL')
    model = MusicModel(pretrained_path=os.path.join(CLSREG_DATASET, "pretrained/music/compact_student.ckpt"))
    save_dir = f"exp/{args.dataset_type}/{args.modality}_{args.batch_size}_{args.lr}"
    model = load_pretrained(save_dir, model)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    pretrain_dir = os.path.join(CLSREG_DATASET, "feature", args.dataset_type, "pretrained", args.modality)
    os.makedirs(pretrain_dir, exist_ok=True)
    model.eval()
    for batch in tqdm(test_loader):
        x = batch['audio']
        fname = batch['fname'][0]
        if args.gpu is not None:
            x = x.cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            embeddings = model.extractor(x)
        embeddings = embeddings.squeeze(0).detach().cpu()
        torch.save(embeddings, os.path.join(pretrain_dir, f"{fname}.pt"))
    
if __name__ == '__main__':
    main()