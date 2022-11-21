import argparse
import builtins
import math
import os
import json
from tqdm import tqdm
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
from stm.metric_learning.loader.dataloader import get_dataset
from stm.metric_learning.model import EmbModel
from stm.utils.eval_utils import load_pretrained, rank_eval

parser = argparse.ArgumentParser(description='PyTorch MSD Training')
parser.add_argument('--speech_type', type=str, default="IEMOCAP")
parser.add_argument('--audio_type', type=str, default="Audioset")
parser.add_argument('--modality', type=str, default="audio")
parser.add_argument('--tid', type=str, default="0")
parser.add_argument('--framework', type=str, default="triplet")
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup_epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--emo_lambda', default=1, type=float)
parser.add_argument('--is_ranksim', default=0, type=int)
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU id to use.')
parser.add_argument('--print_freq', default=50, type=int)

def main():
    args = parser.parse_args()
    main_worker(args)

def main_worker(args):
    test_dataset = get_dataset(
        split="TEST",
        modality_type=args.modality, 
        speech_type=args.speech_type
    )
    model = EmbModel(
        modality = args.modality, 
        framework = args.framework,
        emo_lambda = args.emo_lambda,
        is_ranksim = args.is_ranksim
    )
    save_dir = f"exp/{args.audio_type}_{args.speech_type}/{args.modality}_{args.framework}_{args.is_ranksim}_{args.emo_lambda}_{args.tid}"
    model = load_pretrained(save_dir, model)
    model.eval()
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    cudnn.benchmark = True
    speech_items, music_items = {}, {}
    for idx in tqdm(range(len(test_dataset.fl_speech))):
        s_fname, s_tag, s_emb, _, s_va, s_binary = test_dataset.idx_to_speech(idx)
        s_emb = s_emb.cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            s_emb = model.speech_to_emb(s_emb.unsqueeze(0))
        speech_items[s_fname] = {
            "emb": s_emb.detach().cpu(),
            "va": s_va,
            "binary": s_binary,
            "tag": s_tag
        }
    for idx in tqdm(range(len(test_dataset.fl_music))):
        m_fname, m_tag, m_emb, _, m_va, m_binary = test_dataset.idx_to_music(idx)
        m_emb = m_emb.cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            m_emb = model.music_to_emb(m_emb.unsqueeze(0))
        music_items[m_fname] = {
            "emb": m_emb.detach().cpu(),
            "va": m_va,
            "binary": m_binary,
            "tag": m_tag
        }
    results = rank_eval(args.speech_type, music_items, speech_items, topk = 5)
    with open(os.path.join(save_dir, f"results.json"), mode="w") as io:
        json.dump(results, io, indent=4)
    
    #     torch.save(music_items, os.path.join(save_dir, "music.pt"))
    #     torch.save(speech_items, os.path.join(save_dir, "speech.pt"))
        
if __name__ == '__main__':
    main()