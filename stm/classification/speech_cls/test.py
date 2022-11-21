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
from stm.constants import (IEMOCAP_TAGS, HIKIA_TAGS, RAVDESS_TAGS, SPEECH_SAMPLE_RATE)

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
    test_dataset = get_dataset(args.dataset_type, 'TEST')
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
    model.eval()
    fnames, predictions, ground_truth = [], [], []
    for item in tqdm(test_dataset):
        fname = item["fname"]
        binary = item["binary"]
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
            prediction = model.infer(x)
        predictions.append(prediction.squeeze(0).detach().cpu().numpy())
        ground_truth.append(binary)
        fnames.append(fname)
    predictions = np.stack(predictions)
    ground_truth = np.stack(ground_truth)
    WA, UA, roc_auc, pr_auc, pre_rec_f1_macro, pre_rec_f1_micro = get_scores(predictions, ground_truth)
    result = {
        "WA": WA, 
        "UA": UA, 
        "roc_auc": roc_auc, 
        "pr_auc":pr_auc,
        "pre_rec_f1_macro":pre_rec_f1_macro,
        "pre_rec_f1_micro":pre_rec_f1_micro
    }
    with open(os.path.join(save_dir, "results.json"), mode="w") as io:
        json.dump(result, io, indent=4)
    if args.dataset_type == "IEMOCAP":
        save_cm(predictions.argmax(axis=1), ground_truth.argmax(axis=1), IEMOCAP_TAGS, save_dir)
    elif args.dataset_type == "HIKIA":
        save_cm(predictions.argmax(axis=1), ground_truth.argmax(axis=1), HIKIA_TAGS, save_dir)
    elif args.dataset_type == "RAVDESS":
        save_cm(predictions.argmax(axis=1), ground_truth.argmax(axis=1), RAVDESS_TAGS, save_dir)

if __name__ == '__main__':
    main()