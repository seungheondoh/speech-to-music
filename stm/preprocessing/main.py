import os
import re
import csv
import shutil
import random
import multiprocessing
from functools import partial
from contextlib import contextmanager

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from stm.preprocessing.audio_utils import load_audio, noise
from stm.constants import (DATASET, SPEECH_SAMPLE_RATE, MUSIC_SAMPLE_RATE, STR_CH_FIRST, IEMOCAP_TAGMAP, RAVDESS_CLASS_DICT, AUDIOSET_TAGS, IEMOCAP_TAGS, RAVDESS_TAGS, HIKIA_TAGS)
# INPUT_LENGTH, AUDIOSET_TAGS, IEMOCAP_TAGS, RAVDESS_TAGS, EMOFILM_TAGS, IEMOCAP_MAP,RAVDESS_CLASS_DICT, EMOFILM_MAP

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def noise_sampling():
    train_len = 5348
    eval_len = 60
    noise_color = ['white', 'pink', 'blue', 'brown', 'violet']
    samples = []
    for idx in range(train_len + eval_len):
        color = random.choice(noise_color)
        sample = noise(N=(22050 * 10), color=color, state=None)
        sample = sample.astype(np.float32)
        sample = np.expand_dims(sample, axis=0)
        if idx < train_len:
            split = "TRAIN"
        else:
            split = "EVAL"
        _id = f"{color}_noise_{split}_{idx}"
        save_name = os.path.join(DATASET, f"feature/Audioset/npy/{_id}.npy")
        np.save(save_name, sample)
        samples.append({
            "_id": _id,
            "start" : 0,
            "end": 0,
            "split":split,
            "noise":1,
            "angry":0,
            "exciting":0,
            "funny":0,
            "happy":0,
            "sad":0,
            "scary":0,
            "tender":0
        })
    return samples
    
def get_emotions(path_to_emotions, text_name):
    results = []
    content = open(os.path.join(path_to_emotions, text_name), 'r').read()
    info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)
    info_lines = re.findall(info_line, content)
    for line in info_lines[1:]:  # the first line is a header
        start_end_time, wav_file_name, emotion, val_act_dom = line.strip().split('\t')
        start_time, end_time = start_end_time[1:-1].split('-')
        val, act, dom = val_act_dom[1:-1].split(',')
        val, act, dom = float(val), float(act), float(dom)
        start_time, end_time = float(start_time), float(end_time)
        results.append({
            "start_time" : start_time,
            "end_time" : end_time,
            "wav_file_name" : wav_file_name,
            "emotion" : emotion,
            "valence" : val,
            "activation" : act,
            "dominance" : dom,
        })
    return results

def get_transcriptions(path_to_transcriptions, text_name):
    f = open(os.path.join(path_to_transcriptions, text_name), 'r').read()
    f = np.array(f.split('\n'))
    transcription = {}
    for i in range(len(f) - 1):
        g = f[i]
        i1 = g.find(': ')
        i0 = g.find(' [')
        ind_id = g[:i0]
        ind_ts = g[i1+2:]
        transcription[ind_id] = ind_ts
    return transcription
    
def session_data(data_dir):
    results = []
    for sess in range(1, 6):
        iemocap_path = os.path.join(data_dir, f"Session{sess}/dialog/")
        path_to_wav = os.path.join(iemocap_path, 'wav')
        path_to_emotions = os.path.join(iemocap_path, 'EmoEvaluation')
        path_to_transcriptions = os.path.join(iemocap_path, 'transcriptions')
        fnames = [l for l in os.listdir(path_to_wav) if 'Ses' in l and "._Ses" not in l]
        for fname in fnames:
            if ".pk" in fname:
                continue
            text_name = fname.replace(".wav", ".txt")
            transcriptions = get_transcriptions(path_to_transcriptions, text_name)
            emotions = get_emotions(path_to_emotions, text_name)
            for utterence in emotions:
                wav_to_trans = transcriptions[utterence['wav_file_name']]
                utterence['session'] = sess
                utterence['transcription'] = wav_to_trans
            results.extend(emotions)
    return results

def music_resampler(fname, _id, start, end):
    src, _ = load_audio(
        path=fname,
        ch_format= STR_CH_FIRST,
        sample_rate= MUSIC_SAMPLE_RATE,
        downmix_to_mono= True)
    start_sample = int(start * MUSIC_SAMPLE_RATE)
    end_sample = int(end * MUSIC_SAMPLE_RATE)
    clip = src[:,start_sample:end_sample] # 10sec input
    if clip.shape[1] != 220500:
        pad = np.zeros((1,220500))
        pad[0,:clip.shape[1]] = clip[0,:]
        wav = pad
    else:
        wav = clip
    save_name = os.path.join(DATASET, f"feature/Audioset/npy/{_id}.npy")
    np.save(save_name, wav.astype(np.float32))

def speech_resampler(fname, dataset, _id):
    src, _ = load_audio(
        path=fname,
        ch_format= STR_CH_FIRST,
        sample_rate= SPEECH_SAMPLE_RATE,
        downmix_to_mono= True)
    save_name = os.path.join(DATASET, f"feature/{dataset}/npy/{_id}.npy")
    np.save(save_name, src.astype(np.float32))

def IEMOCAP_split_fn(df_filtered):
    lb = preprocessing.LabelBinarizer()
    df_filtered = df_filtered.replace({"emotion": IEMOCAP_TAGMAP})
    binary = lb.fit_transform(df_filtered['emotion'])
    df = pd.DataFrame(binary, index=df_filtered.index, columns=lb.classes_)
    x_TE = df_filtered[df_filtered['session'] == 5].index
    x_TR, x_VA, y_TR, y_VA = train_test_split(df_filtered.index, df_filtered['emotion'], stratify=df_filtered['emotion'], test_size=0.1, random_state=42)
    os.makedirs(os.path.join(DATASET, "split/IEMOCAP"), exist_ok=True)
    df.loc[x_TR].to_csv(os.path.join(DATASET, "split/IEMOCAP/train.csv"))
    df.loc[x_VA].to_csv(os.path.join(DATASET, "split/IEMOCAP/valid.csv"))
    df.loc[x_TE].to_csv(os.path.join(DATASET, "split/IEMOCAP/test.csv"))

def IEMOCAP_processor():
    os.makedirs(os.path.join(DATASET, "feature/IEMOCAP/wav"), exist_ok=True)
    os.makedirs(os.path.join(DATASET, "feature/IEMOCAP/npy"), exist_ok=True)
    data_dir = os.path.join(DATASET, "raw/IEMOCAP_full_release")
    total_item= session_data(data_dir)
    df = pd.DataFrame(total_item)
    black_list = ['xxx', 'oth', 'dis', 'fea', 'sur']
    filtered_list = [idx for idx, i in enumerate(df["emotion"]) if i not in black_list]
    df_filtered = df.iloc[filtered_list].set_index("wav_file_name")    
    IEMOCAP_split_fn(df_filtered)    

    _fnames, _dataset, _ids = [], [], []
    for idx in tqdm(range(len(df_filtered))):
        item = df_filtered.iloc[idx]
        fname = item.name
        sess = item['session']
        dialog = "_".join(fname.split("_")[:-1])
        source = os.path.join(data_dir, f"Session{sess}/sentences/wav/{dialog}/{fname}.wav")
        shutil.copy(source, os.path.join(DATASET, f"feature/IEMOCAP/wav/{fname}.wav"))
        _fnames.append(source)
        _ids.append(fname)
        _dataset.append("IEMOCAP")
    with poolcontext(processes = 20) as pool:
        pool.starmap(speech_resampler, zip(_fnames, _dataset, _ids))


def Audioset_split_fn(df_audioset):
    labels = ['happy','funny','sad','tender','exciting','angry','scary','noise']
    df_train = df_audioset[df_audioset['split'] == "TRAIN"]
    df_eval = df_audioset[df_audioset['split'] == "EVAL"]
    df_binary = df_train[labels]
    label_list = []
    for idx in range(len(df_train)):
        item = df_binary.iloc[idx]
        label_list.append(item.idxmax())
    x_TR, x_VA, _, _ = train_test_split(df_binary.index, label_list, stratify=label_list, test_size=0.1, random_state=42)
    df_binary.loc[x_TR].to_csv(os.path.join(DATASET, "split/Audioset/train.csv"))
    df_binary.loc[x_VA].to_csv(os.path.join(DATASET, "split/Audioset/valid.csv"))
    df_eval[labels].to_csv(os.path.join(DATASET, "split/Audioset/test.csv")) 

def Audioset_processor():
    os.makedirs(os.path.join(DATASET, "feature/Audioset/npy"), exist_ok=True)
    df_audioset = pd.read_csv(os.path.join(DATASET, f"raw/Audioset/audioset_mood.csv"), index_col=0)
    f_list= os.listdir(os.path.join(DATASET, f"raw/Audioset/wav"))
    save_fname = [fname.replace(".mp3","") for fname in f_list if (".mp3" in fname)]
    _fnames, _ids, _starts, _ends = [], [], [], []
    for idx in tqdm(range(len(df_audioset))):
        item = df_audioset.iloc[idx]
        fname = os.path.join(DATASET, f"raw/Audioset/wav/{item.name}.mp3")
        if item.name in save_fname:
            _fnames.append(fname)
            _ids.append(item.name)
            _starts.append(item['start'])
            _ends.append(item['end'])
    print("extract files ", len(_ids))
    with poolcontext(processes = 20) as pool:
        pool.starmap(music_resampler, zip(_fnames, _ids, _starts, _ends))

    df_audioset = df_audioset.loc[save_fname]
    samples = noise_sampling()
    noise_sample = pd.DataFrame(samples).set_index("_id")

    noise_label = [0 for i in range(len(df_audioset))]
    df_audioset['noise'] = noise_label
    df_final = pd.concat([df_audioset, noise_sample])
    df_final.to_csv(os.path.join(DATASET,"split/Audioset/annotation.csv"))
    Audioset_split_fn(df_final)


def RAVDESS_processor():
    os.makedirs(os.path.join(DATASET, "feature/RAVDESS/npy"), exist_ok=True)
    os.makedirs(os.path.join(DATASET, "split/RAVDESS/"), exist_ok=True)
    root_dir = os.path.join(DATASET, "raw/RAVDESS/")
    actors = os.listdir(root_dir)
    samples, _fnames, _dataset, _ids = [], [], [], []
    for actor_name in actors:
        for item in os.listdir(os.path.join(root_dir, actor_name)):
            _fnames.append(os.path.join(root_dir, actor_name, item))
            _dataset.append("RAVDESS")
            _ids.append(item.replace(".wav",""))
            modality, vocal_channel, emotion, intensity, statement, repetition, actor = item.replace(".wav","").split("-")
            if int(actor) < 20:
                split = "TRAIN"
            elif 20 <= int(actor) < 23:
                split = "VALID"
            else:
                split = "TEST"

            if int(actor) % 2 == 0:
                gender = "male"
            else:
                gender = "female"
            sample = {
                "fname":item.replace(".wav",""),
                "modality":RAVDESS_CLASS_DICT['modality'][modality],
                "vocal_channel":RAVDESS_CLASS_DICT['vocal_channel'][vocal_channel],
                "emotion":RAVDESS_CLASS_DICT['emotion'][emotion],
                "intensity":RAVDESS_CLASS_DICT['intensity'][intensity],
                "statement":RAVDESS_CLASS_DICT['statement'][statement],
                "repetition":RAVDESS_CLASS_DICT['repetition'][repetition],
                "actor": actor,
                "gender": gender,
                "split":split
            }
            samples.append(sample)
    df_all = pd.DataFrame(samples)
    df_all = df_all.set_index("fname")
    df_all.to_csv(os.path.join(DATASET, "split/RAVDESS", "annotation.csv"))
    lb = preprocessing.LabelBinarizer()
    binary = lb.fit_transform(df_all['emotion'])
    df_binary = pd.DataFrame(binary, index=df_all.index, columns=lb.classes_)
    print(df_binary)
    df_binary[df_all['split'] == "TRAIN"].to_csv(os.path.join(DATASET, "split/RAVDESS", "train.csv"))
    df_binary[df_all['split'] == "VALID"].to_csv(os.path.join(DATASET, "split/RAVDESS", "valid.csv"))
    df_binary[df_all['split'] == "TEST"].to_csv(os.path.join(DATASET, "split/RAVDESS", "test.csv"))
    # with poolcontext(processes = 20) as pool:
    #     pool.starmap(speech_resampler, zip(_fnames, _dataset, _ids))
    print("finish extract")


def load_glove_model(File):
    print("Loading Glove Model")
    glove_model = {}
    with open(File,'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model

def tag_embedding_extractor(): 
    total_tag = list(set(AUDIOSET_TAGS+IEMOCAP_TAGS+RAVDESS_TAGS+HIKIA_TAGS))
    glove = load_glove_model("../../../glove.42B.300d.txt")
    glove_emb= {tag:np.array(glove[tag]) for tag in total_tag}
    os.makedirs(os.path.join(DATASET,"pretrained","word"), exist_ok=True)
    torch.save(glove_emb, os.path.join(DATASET,"pretrained","word","glove.pt"))

def main():
    # Audioset_processor()
    # IEMOCAP_processor()
    # RAVDESS_processor()
    # EMOFILM_processor()
    tag_embedding_extractor()

if __name__ == '__main__':
    main()