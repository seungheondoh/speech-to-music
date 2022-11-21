import os
import torch
import random
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Processor, DistilBertTokenizer
from stm.constants import (DATASET, AUDIOSET_TAGS, IEMOCAP_TAGS, HIKIA_TAGS, RAVDESS_TAGS, SPEECH_SAMPLE_RATE, VA_MAP)

class Emotion_Dataset(Dataset):
    def __init__(self, split, modality_type, speech_type, music_type="Audioset"):
        self.split = split
        self.modality_type = modality_type
        self.speech_type = speech_type
        self.music_type = music_type
        if speech_type == "IEMOCAP":
            self.speech_tags = IEMOCAP_TAGS
        elif speech_type == "RAVDESS":
            self.speech_tags = RAVDESS_TAGS
        elif speech_type == "HIKIA":
            self.speech_tags = HIKIA_TAGS
        self.music_tags = AUDIOSET_TAGS
        self.get_mapping()
        if split == 'TRAIN':
            self.fl_speech = pd.read_csv(os.path.join(DATASET, f"split/{self.speech_type}/train.csv"), index_col=0)
            self.fl_music = pd.read_csv(os.path.join(DATASET, f"split/{self.music_type}/train.csv"), index_col=0)
        elif split == 'VALID':
            self.fl_speech = pd.read_csv(os.path.join(DATASET, f"split/{self.speech_type}/valid.csv"), index_col=0)
            self.fl_music = pd.read_csv(os.path.join(DATASET, f"split/{self.music_type}/test.csv"), index_col=0)
        elif split == 'TEST':
            self.fl_speech = pd.read_csv(os.path.join(DATASET, f"split/{self.speech_type}/test.csv"), index_col=0)
            self.fl_music = pd.read_csv(os.path.join(DATASET, f"split/{self.music_type}/test.csv"), index_col=0)

    def get_mapping(self):
        tags = [tag for tag in VA_MAP.keys()]
        va_embs = np.array([VA_MAP[tag] for tag in tags])
        sim = 1 - euclidean_distances(va_embs)
        df_sim = pd.DataFrame(sim, index=tags, columns=tags)
        self.max_mapper = df_sim.loc[self.speech_tags][self.music_tags].idxmax(axis=1).to_dict()
        self.va_mapper = VA_MAP
        self.tag_mapper = torch.load(os.path.join(DATASET, f"pretrained/word/glove.pt"))

    def get_speech_emb(self, fname):
        if self.modality_type == "audio":
            speech_emb = torch.load(os.path.join(DATASET, f"feature/{self.speech_type}/pretrained/audio/{fname}.pt"))
        elif self.modality_type == "text":
            speech_emb = torch.load(os.path.join(DATASET, f"feature/{self.speech_type}/pretrained/text/{fname}.pt"))
        elif self.modality_type == "fusion":
            audio_emb = torch.load(os.path.join(DATASET, f"feature/{self.speech_type}/pretrained/audio/{fname}.pt"))
            text_emb = torch.load(os.path.join(DATASET, f"feature/{self.speech_type}/pretrained/text/{fname}.pt"))
            speech_emb = torch.cat([audio_emb, text_emb], dim=0)
        else:
            speech_emb = None
        return speech_emb

    def idx_to_speech(self, i):
        item = self.fl_speech.iloc[i]
        s_fname = item.name
        s_emb = self.get_speech_emb(s_fname)
        s_binary = np.array(item).astype(np.float32) 
        s_tag = item.idxmax()
        s_tag_emb = np.array(self.tag_mapper[s_tag]).astype(np.float32) 
        s_va = np.array(self.va_mapper[s_tag]).astype(np.float32)
        return s_fname, s_tag, s_emb, s_tag_emb, s_va, s_binary

    def idx_to_music(self, i):
        item = self.fl_music.iloc[i]
        m_fname = item.name
        m_binary = np.array(item).astype(np.float32) 
        m_tag = item.idxmax()
        m_tag_emb = np.array(self.tag_mapper[m_tag]).astype(np.float32) 
        m_va = np.array(self.va_mapper[m_tag]).astype(np.float32)
        m_emb = torch.load(os.path.join(DATASET, f"feature/{self.music_type}/pretrained/audio/{m_fname}.pt"))
        return m_fname, m_tag, m_emb, m_tag_emb, m_va, m_binary
    
    def __getitem__(self, index):
        i = random.randrange(len(self.speech_tags))
        s_tag = self.speech_tags[i]
        s_tag_emb = np.array(self.tag_mapper[s_tag]).astype(np.float32) 
        s_va = np.array(self.va_mapper[s_tag]).astype(np.float32)
        s_item_list = self.fl_speech[s_tag]
        s_pos_pool = s_item_list[s_item_list != 0].index
        s_fname = random.choice(s_pos_pool)
        s_emb = self.get_speech_emb(s_fname)
        s_binary = self.fl_speech.loc[s_fname].to_numpy()
        # max sampling -> get positive
        m_tag = self.max_mapper[s_tag]
        m_tag_emb = np.array(self.tag_mapper[m_tag]).astype(np.float32)
        m_va = np.array(self.va_mapper[m_tag]).astype(np.float32)
        m_item_list = self.fl_music[m_tag]
        m_pos_pool = m_item_list[m_item_list != 0].index
        m_fname = random.choice(m_pos_pool)
        m_emb = torch.load(os.path.join(DATASET, f"feature/{self.music_type}/pretrained/audio/{m_fname}.pt"))
        m_binary = self.fl_music.loc[m_fname].to_numpy()
        return {
            "s_fname": s_fname, "s_tag":s_tag, "s_emb": s_emb, "s_tag_emb": s_tag_emb, "s_va":s_va, "s_binary":s_binary.astype(np.float32),
            "m_fname": m_fname, "m_tag":m_tag, "m_emb": m_emb, "m_tag_emb": m_tag_emb, "m_va":m_va,"m_binary":m_binary.astype(np.float32)
        }

    def __len__(self):
        return len(self.fl_speech)