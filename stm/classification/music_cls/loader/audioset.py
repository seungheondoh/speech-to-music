import os
import torch
import random
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from stm.constants import (CLSREG_DATASET)

class AUDIOSET_DATASET(Dataset):
    def __init__(self, split):
        self.split = split
        if split == 'TRAIN':
            fl_train = pd.read_csv(os.path.join(CLSREG_DATASET, "split", "Audioset", "train.csv"), index_col=0)
            fl_valid = pd.read_csv(os.path.join(CLSREG_DATASET, "split", "Audioset", "valid.csv"), index_col=0)
            self.fl = pd.concat([fl_train,fl_valid])
        elif split == 'VALID':
            self.fl = pd.read_csv(os.path.join(CLSREG_DATASET, "split", "Audioset", "test.csv"), index_col=0)
        elif split == 'TEST':
            self.fl = pd.read_csv(os.path.join(CLSREG_DATASET, "split", "Audioset", "test.csv"), index_col=0)
        elif split == 'ALL':
            fl_train = pd.read_csv(os.path.join(CLSREG_DATASET, "split", "Audioset", "train.csv"), index_col=0)
            fl_valid = pd.read_csv(os.path.join(CLSREG_DATASET, "split", "Audioset", "valid.csv"), index_col=0)
            fl_test = pd.read_csv(os.path.join(CLSREG_DATASET, "split", "Audioset", "test.csv"), index_col=0)
            self.fl = pd.concat([fl_train,fl_valid, fl_test])
            
    def __getitem__(self, index):
        item = self.fl.iloc[index]
        fname = item.name
        binary = np.array(item.values).astype(np.float32)
        audio = np.load(os.path.join(CLSREG_DATASET, "feature", "Audioset", 'npy', fname + ".npy"))
        return {"audio" : audio, "binary":binary, "fname":fname}

    def __len__(self):
        return len(self.fl)