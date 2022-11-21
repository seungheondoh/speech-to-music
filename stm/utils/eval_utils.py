import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn import metrics
from astropy.stats import jackknife
from sklearn import metrics
from astropy.stats import jackknife
from sklearn.metrics.pairwise import euclidean_distances
from stm.constants import (DATASET, AUDIOSET_TAGS, IEMOCAP_TAGS, HIKIA_TAGS, RAVDESS_TAGS, SPEECH_SAMPLE_RATE, VA_MAP)

def load_pretrained(save_dir, model):
    checkpoint= torch.load(os.path.join(save_dir,"best.pth"), map_location='cpu')
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    return model

def get_r2(y_true, y_pred):
    rv, ra = metrics.r2_score(y_true, y_pred, multioutput='raw_values')
    r2 = np.mean([rv, ra])
    return r2, rv, ra

def rank_eval(speech_dataset, music_items, speech_items, topk = 5):
    if speech_dataset == "IEMOCAP":
        speech_tags = IEMOCAP_TAGS
    elif speech_dataset == "RAVDESS":
        speech_tags = RAVDESS_TAGS
    elif speech_dataset == "HIKIA":
        speech_tags = HIKIA_TAGS
    music_tags = AUDIOSET_TAGS
    pred_items, df_pred = get_pred(music_items, speech_items)
    gt_items = get_harg_target(music_items, speech_items, speech_tags, music_tags)
    sort_target = get_soft_target(music_items, speech_items)
    P5, mrrs = [], []
    for idx, src in enumerate(gt_items):
        gt_fname = gt_items[src]
        pred_fnames = pred_items[src]
        preds = np.asarray([pred in gt_fname for pred in pred_fnames])
        rank_value = min([idx for idx, retrieval_success in enumerate(preds) if retrieval_success])
        top5 = preds[:topk]
        prec5 = top5.sum() / len(top5)
        P5.append(prec5)
        mrrs.append(1 / (rank_value + 1))
    return {
        f"mrr": np.mean(mrrs),
        f"P@{topk}": np.mean(P5),
        f"NDCG@{topk}": metrics.ndcg_score(sort_target, df_pred, k=5),
    }
    

def get_pred(music_items, speech_items):
    m_fnames = [k for k in music_items.keys()]
    s_fnames = [k for k in speech_items.keys()]
    s_emb = torch.cat([speech_items[k]['emb'] for k in s_fnames], dim=0)
    m_emb = torch.cat([music_items[k]['emb'] for k in m_fnames], dim=0)
    s_emb = nn.functional.normalize(s_emb, dim=1)
    m_emb = nn.functional.normalize(m_emb, dim=1)
    prediction = s_emb @ m_emb.T
    df_pred = pd.DataFrame(prediction.numpy(), index=s_fnames, columns=m_fnames)
    pred_items = {}
    for idx in range(len(df_pred)):
        item = df_pred.iloc[idx]
        pred_items[item.name] = list(item.sort_values(ascending=False).index)
    return pred_items, df_pred

def get_soft_target(music_items, speech_items):
    m_fnames = [k for k in music_items.keys()]
    s_fnames = [k for k in speech_items.keys()]
    s_va = np.stack([speech_items[k]['va'] for k in s_fnames])
    m_va = np.stack([music_items[k]['va'] for k in m_fnames])
    sim = 1 - euclidean_distances(s_va, m_va)
    df_va = pd.DataFrame(sim, index=s_fnames, columns=m_fnames)
    return df_va

def get_harg_target(music_items, speech_items, speech_tags, music_tags):
    tags = [tag for tag in VA_MAP.keys()]
    va_embs = np.array([VA_MAP[tag] for tag in tags])
    sim = 1 - euclidean_distances(va_embs)
    df_sim = pd.DataFrame(sim, index=tags, columns=tags)
    max_mapper = df_sim.loc[speech_tags][music_tags].idxmax(axis=1).to_dict()
    s_fnames = [k for k in speech_items.keys()]
    m_fnames = [k for k in music_items.keys()]
    tag_to_music = {i:[] for i in music_tags}
    for fname in m_fnames:
        tag = music_items[fname]['tag']
        tag_to_music[tag].append(fname)

    hard_target = {}
    for fname in s_fnames:
        stm_tag = max_mapper[speech_items[fname]['tag']]
        hard_target[fname] = tag_to_music[stm_tag]
    return hard_target


def get_scores(predict, label):
    y_pred = np.argmax(predict, axis=1)
    y_true = np.argmax(label, axis=1)
    WA = metrics.accuracy_score(y_true, y_pred)
    UA = metrics.balanced_accuracy_score(y_true, y_pred)
    roc_auc = metrics.roc_auc_score(label, predict, average='macro')
    pr_auc = metrics.average_precision_score(label, predict, average='macro')
    pre_rec_f1_macro = metrics.precision_recall_fscore_support(y_true, y_pred, average='macro')
    pre_rec_f1_micro = metrics.precision_recall_fscore_support(y_true, y_pred, average='micro')
    return WA, UA, roc_auc, pr_auc, pre_rec_f1_macro, pre_rec_f1_micro
