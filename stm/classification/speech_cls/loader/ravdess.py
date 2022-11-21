import os
import torch
import random
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Processor, DistilBertTokenizer
from stm.constants import (CLSREG_DATASET, SPEECH_SAMPLE_RATE)

class RAVDESS_Dataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
        self.maxseqlen = SPEECH_SAMPLE_RATE * 16
        self.df_meta = pd.read_csv(os.path.join(CLSREG_DATASET, f"split/RAVDESS/annotation.csv"), index_col=0)
        if split == 'TRAIN':
            self.fl = pd.read_csv(os.path.join(CLSREG_DATASET, f"split/RAVDESS/train.csv"), index_col=0)
        elif split == 'VALID':
            self.fl = pd.read_csv(os.path.join(CLSREG_DATASET, f"split/RAVDESS/valid.csv"), index_col=0)
        elif split == 'TEST':
            self.fl = pd.read_csv(os.path.join(CLSREG_DATASET, f"split/RAVDESS/test.csv"), index_col=0)
        elif split == "ALL":
            tr = pd.read_csv(os.path.join(CLSREG_DATASET, f"split/RAVDESS/train.csv"), index_col=0)
            va = pd.read_csv(os.path.join(CLSREG_DATASET, f"split/RAVDESS/valid.csv"), index_col=0)
            te = pd.read_csv(os.path.join(CLSREG_DATASET, f"split/RAVDESS/test.csv"), index_col=0)
            self.fl = pd.concat([tr,va,te])
            
    def __getitem__(self, index):
        item = self.fl.iloc[index]
        fname = item.name
        binary = item.values
        text = self.df_meta.loc[fname]['statement']
        audio = np.load(os.path.join(CLSREG_DATASET, f"feature/RAVDESS/npy/{fname}.npy"), mmap_mode='r')
        audio = np.array(audio).squeeze(0)
        return {
            "fname":fname, 
            "binary":binary, 
            "audio":audio, 
            "text":text
        }

    def batch_processor(self, batch):
        audios = [item_dict['audio'] for item_dict in batch]
        texts = [item_dict['text'] for item_dict in batch]
        fnames = [item_dict['fname'] for item_dict in batch]
        binarys = [item_dict['binary'] for item_dict in batch]
        audio_encoding = self.processor(audios, padding="max_length", max_length=self.maxseqlen, truncation=True, return_tensors='pt',sampling_rate=SPEECH_SAMPLE_RATE, return_attention_mask=True)
        text_encoding = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        
        audios = audio_encoding['input_values']
        audio_mask = audio_encoding['attention_mask']
        token = text_encoding['input_ids']
        token_mask = text_encoding['attention_mask']
        binarys = torch.from_numpy(np.stack(binarys).astype(np.float32))
        return {"audios" : audios, "audio_mask": audio_mask,
                "token" : token, "token_mask": token_mask, 
                "fnames": fnames, "binarys" : binarys}

    def __len__(self):
        return len(self.fl)