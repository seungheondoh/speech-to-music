import argparse
import os
import json
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
from tqdm import tqdm
from transformers import Wav2Vec2Processor, DistilBertTokenizer
from stm.classification.speech_cls.loader.dataloader import get_dataset
from stm.classification.speech_cls.model import AudioModel, TextModel
from stm.utils.eval_utils import load_pretrained, get_scores

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from stm.constants import (CLSREG_DATASET,IEMOCAP_TAGS, HIKIA_TAGS, RAVDESS_TAGS, SPEECH_SAMPLE_RATE)

def save_cm(predict, label, label_name, save_path):
    predict_ = [label_name[i] for i in predict]
    label_ = [label_name[i] for i in label]
    cm = confusion_matrix(label_, predict_, labels=label_name)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_name)
    disp.plot(xticks_rotation="vertical")
    plt.savefig(os.path.join(save_path, 'cm.png'), dpi=150)

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
parser.add_argument('--print_freq', default=100, type=int)
# train detail


def main():
    args = parser.parse_args()
    main_worker(args)

def main_worker(args):
    test_dataset = get_dataset(args.dataset_type, 'ALL')
    if args.modality == "audio":
        model = AudioModel(
            dataset_type=args.dataset_type,
            freeze_type="feature"
        )
        processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
    elif args.modality == "text":
        model = TextModel(
            dataset_type=args.dataset_type,
            freeze_type="feature"
        )
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    save_dir = f"exp/{args.dataset_type}/{args.modality}_{args.batch_size}_{args.lr}"
    model = load_pretrained(save_dir, model)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    cudnn.benchmark = True
    pretrain_dir = os.path.join(CLSREG_DATASET, "feature", args.dataset_type, "pretrained", args.modality)
    os.makedirs(pretrain_dir, exist_ok=True)
    model.eval()
    for item in tqdm(test_dataset):
        fname = item["fname"]
        if args.modality == "audio":
            audio = item["audio"]
            audio_encoding = processor(audio, return_tensors='pt',sampling_rate=SPEECH_SAMPLE_RATE)
            x = audio_encoding['input_values']
        elif args.modality == "text":
            text = item["text"]
            text_encoding = tokenizer(text, return_tensors='pt')
            x = text_encoding['input_ids']
        if args.gpu is not None:
            x = x.cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            embeddings = model.pooling_extractor(x, mask=None)
        embeddings = embeddings.squeeze(0).detach().cpu()
        torch.save(embeddings, os.path.join(pretrain_dir, f"{fname}.pt"))
if __name__ == '__main__':
    main()